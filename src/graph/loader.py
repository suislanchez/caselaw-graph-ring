"""Bulk data loader for Neo4j citation graph."""

import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RAW_DIR,
    PROCESSED_DIR,
    CITATIONS_DIR,
    EMBEDDINGS_DIR,
)
from src.graph.schema import create_schema


# Default embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 100


def load_cases_from_json(cases_dir: Path = RAW_DIR / "cases") -> Iterator[dict]:
    """
    Load cases from JSON files.

    Args:
        cases_dir: Directory containing case JSON files

    Yields:
        Case dictionaries
    """
    if not cases_dir.exists():
        return

    for json_file in sorted(cases_dir.glob("*.json")):
        with open(json_file) as f:
            case = json.load(f)
            yield case


def load_cases_from_parquet(parquet_path: Path = PROCESSED_DIR / "scdb_matched.parquet") -> pd.DataFrame:
    """
    Load cases from the matched SCDB parquet file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        DataFrame with case data
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def load_edges_from_csv(edges_path: Path = CITATIONS_DIR / "edges.csv") -> pd.DataFrame:
    """
    Load citation edges from CSV.

    Args:
        edges_path: Path to edges CSV file

    Returns:
        DataFrame with columns: source_case_id, target_case_id, weight, citation_text
    """
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path}")
    return pd.read_csv(edges_path)


def generate_embeddings(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Generate text embeddings using sentence-transformers.

    Args:
        texts: List of texts to embed
        model: SentenceTransformer model (loads default if not provided)
        batch_size: Batch size for encoding
        show_progress: Show progress bar

    Returns:
        List of embedding vectors
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    return [emb.tolist() for emb in embeddings]


def load_cases_to_neo4j(
    cases: list[dict],
    driver=None,
    generate_embeddings_flag: bool = True,
    embedding_model: Optional[SentenceTransformer] = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Load cases as nodes into Neo4j.

    Args:
        cases: List of case dictionaries with keys:
            - id: Unique identifier
            - cap_id: CAP case ID (optional)
            - scdb_id: SCDB case ID (optional)
            - name: Case name
            - date: Decision date
            - court: Court name
            - citation: Primary citation
            - outcome: 'petitioner' or 'respondent' (optional)
            - text: Full case text
        driver: Neo4j driver instance
        generate_embeddings_flag: Whether to generate text embeddings
        embedding_model: Pre-loaded embedding model
        batch_size: Batch size for Neo4j transactions

    Returns:
        Number of cases loaded
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    # Generate embeddings if requested
    embeddings = None
    if generate_embeddings_flag and cases:
        print("Generating text embeddings...")
        if embedding_model is None:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Use truncated text for embedding (first 512 tokens worth)
        texts = [c.get("text", "")[:4096] for c in cases]
        embeddings = generate_embeddings(texts, embedding_model)

    # Load cases in batches
    total_loaded = 0
    try:
        with driver.session() as session:
            for i in tqdm(range(0, len(cases), batch_size), desc="Loading cases"):
                batch = cases[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size] if embeddings else None

                # Prepare batch data
                batch_data = []
                for j, case in enumerate(batch):
                    node_data = {
                        "id": case.get("id", case.get("cap_id", "")),
                        "cap_id": case.get("cap_id"),
                        "scdb_id": case.get("scdb_id"),
                        "name": case.get("name", ""),
                        "date": case.get("date"),
                        "year": case.get("year"),
                        "court": case.get("court", ""),
                        "citation": case.get("citation", ""),
                        "outcome": case.get("outcome"),
                        "text": case.get("text", ""),
                        "text_length": len(case.get("text", "")),
                    }
                    if batch_embeddings:
                        node_data["embedding"] = batch_embeddings[j]
                    batch_data.append(node_data)

                # Batch insert
                result = session.run(
                    """
                    UNWIND $batch AS case
                    MERGE (c:Case {id: case.id})
                    SET c.cap_id = case.cap_id,
                        c.scdb_id = case.scdb_id,
                        c.name = case.name,
                        c.date = date(case.date),
                        c.year = case.year,
                        c.court = case.court,
                        c.citation = case.citation,
                        c.outcome = case.outcome,
                        c.text = case.text,
                        c.text_length = case.text_length,
                        c.embedding = case.embedding
                    RETURN count(c) AS loaded
                    """,
                    batch=batch_data,
                )
                total_loaded += result.single()["loaded"]

    finally:
        if close_driver:
            driver.close()

    return total_loaded


def load_edges_to_neo4j(
    edges: pd.DataFrame,
    driver=None,
    batch_size: int = BATCH_SIZE * 10,
) -> int:
    """
    Load citation edges into Neo4j.

    Args:
        edges: DataFrame with columns: source_case_id, target_case_id, weight, citation_text
        driver: Neo4j driver instance
        batch_size: Batch size for Neo4j transactions

    Returns:
        Number of edges loaded
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    total_loaded = 0
    try:
        with driver.session() as session:
            # Convert to list of dicts
            edges_list = edges.to_dict("records")

            for i in tqdm(range(0, len(edges_list), batch_size), desc="Loading edges"):
                batch = edges_list[i : i + batch_size]

                result = session.run(
                    """
                    UNWIND $batch AS edge
                    MATCH (source:Case {id: edge.source_case_id})
                    MATCH (target:Case {id: edge.target_case_id})
                    MERGE (source)-[r:CITES]->(target)
                    SET r.citation_text = edge.citation_text,
                        r.weight = coalesce(edge.weight, 1.0)
                    RETURN count(r) AS loaded
                    """,
                    batch=batch,
                )
                total_loaded += result.single()["loaded"]

    finally:
        if close_driver:
            driver.close()

    return total_loaded


def load_all(
    cases_source: str = "parquet",
    edges_path: Optional[Path] = None,
    create_schema_flag: bool = True,
    generate_embeddings_flag: bool = True,
) -> dict:
    """
    Load all data into Neo4j.

    Args:
        cases_source: 'parquet' or 'json'
        edges_path: Path to edges CSV (uses default if not provided)
        create_schema_flag: Whether to create schema first
        generate_embeddings_flag: Whether to generate text embeddings

    Returns:
        Dict with loading statistics
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Create schema
        if create_schema_flag:
            print("Creating schema...")
            create_schema(driver)

        # Load cases
        print("Loading cases...")
        if cases_source == "parquet":
            df = load_cases_from_parquet()
            cases = df.to_dict("records")
        else:
            cases = list(load_cases_from_json())

        cases_loaded = load_cases_to_neo4j(
            cases,
            driver=driver,
            generate_embeddings_flag=generate_embeddings_flag,
        )

        # Load edges
        edges_loaded = 0
        edges_path = edges_path or (CITATIONS_DIR / "edges.csv")
        if edges_path.exists():
            print("Loading edges...")
            edges_df = load_edges_from_csv(edges_path)
            edges_loaded = load_edges_to_neo4j(edges_df, driver=driver)
        else:
            print(f"No edges file found at {edges_path}, skipping edge loading.")

        return {
            "cases_loaded": cases_loaded,
            "edges_loaded": edges_loaded,
        }

    finally:
        driver.close()


def update_embeddings(
    embedding_path: Path = EMBEDDINGS_DIR / "graphsage_embeddings.pt",
    driver=None,
) -> int:
    """
    Update nodes with GraphSAGE embeddings.

    Args:
        embedding_path: Path to saved GraphSAGE embeddings
        driver: Neo4j driver instance

    Returns:
        Number of nodes updated
    """
    import torch

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embedding_path}")

    data = torch.load(embedding_path)
    node_ids = data["node_ids"]
    embeddings = data["embeddings"].numpy()

    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    total_updated = 0
    try:
        with driver.session() as session:
            for i in tqdm(range(0, len(node_ids), BATCH_SIZE), desc="Updating GraphSAGE embeddings"):
                batch_ids = node_ids[i : i + BATCH_SIZE]
                batch_embeddings = [emb.tolist() for emb in embeddings[i : i + BATCH_SIZE]]

                batch_data = [
                    {"id": nid, "embedding": emb}
                    for nid, emb in zip(batch_ids, batch_embeddings)
                ]

                result = session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.id})
                    SET c.graphsage_embedding = item.embedding
                    RETURN count(c) AS updated
                    """,
                    batch=batch_data,
                )
                total_updated += result.single()["updated"]

    finally:
        if close_driver:
            driver.close()

    return total_updated


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py [load|update-embeddings]")
        print("  load: Load cases and edges from processed data")
        print("  update-embeddings: Update nodes with GraphSAGE embeddings")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "load":
        stats = load_all()
        print(f"\nLoaded {stats['cases_loaded']} cases and {stats['edges_loaded']} edges.")
    elif command == "update-embeddings":
        updated = update_embeddings()
        print(f"\nUpdated {updated} nodes with GraphSAGE embeddings.")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
