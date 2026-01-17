"""Build citation graph edge list for Neo4j import.

Creates directed edges from citing cases to cited cases based on
extracted and linked citations.
"""

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel, Field

from src.config import CITATIONS_DIR
from .extractor import CitationExtractor, ExtractedCitation
from .linker import CitationLinker

logger = logging.getLogger(__name__)


class CitationEdge(BaseModel):
    """A directed edge representing a citation relationship.

    source_case_id cites target_case_id via citation_text.
    """

    source_case_id: str = Field(description="CAP ID of the citing case")
    target_case_id: str = Field(description="CAP ID of the cited case")
    citation_text: str = Field(description="The citation text (e.g., '347 U.S. 483')")
    weight: float = Field(1.0, description="Edge weight (default 1.0, can be TF-IDF)")
    citation_type: Optional[str] = Field(None, description="Classification of citation")
    source_year: Optional[int] = Field(None, description="Year of citing case")
    target_year: Optional[int] = Field(None, description="Year of cited case (if known)")


class EdgeBuildStats(BaseModel):
    """Statistics from edge building process."""

    total_cases: int = 0
    total_citations_extracted: int = 0
    unique_citations: int = 0
    successful_links: int = 0
    edges_created: int = 0
    self_citations_removed: int = 0
    duplicate_edges_removed: int = 0
    processing_time_seconds: float = 0.0


def compute_tf_idf_weights(
    case_citations: Dict[str, List[ExtractedCitation]],
) -> Dict[str, Dict[str, float]]:
    """Compute TF-IDF weights for citation edges.

    TF (term frequency): How often a case cites another case
    IDF (inverse document frequency): How rare a citation is across all cases

    Args:
        case_citations: Dictionary mapping case ID to extracted citations

    Returns:
        Nested dict: source_case_id -> citation -> weight
    """
    # Count document frequency for each citation
    citation_doc_freq: Dict[str, int] = defaultdict(int)
    for citations in case_citations.values():
        seen_in_case: Set[str] = set()
        for c in citations:
            if c.normalized not in seen_in_case:
                citation_doc_freq[c.normalized] += 1
                seen_in_case.add(c.normalized)

    total_docs = len(case_citations)
    weights: Dict[str, Dict[str, float]] = {}

    for case_id, citations in case_citations.items():
        weights[case_id] = {}

        # Count term frequency (citation count in this case)
        citation_tf: Dict[str, int] = defaultdict(int)
        for c in citations:
            citation_tf[c.normalized] += 1

        # Calculate TF-IDF
        for citation, tf in citation_tf.items():
            df = citation_doc_freq[citation]
            # TF-IDF formula: tf * log(N / df)
            import math
            idf = math.log(total_docs / df) if df > 0 else 0
            weights[case_id][citation] = tf * idf

    return weights


async def build_edge_list(
    cases: List["Case"],
    linker: Optional[CitationLinker] = None,
    extractor: Optional[CitationExtractor] = None,
    use_tf_idf: bool = False,
    remove_self_citations: bool = True,
    show_progress: bool = True,
) -> List[CitationEdge]:
    """Build citation edge list from cases.

    Extracts citations from each case, links them to case IDs,
    and creates directed edges.

    Args:
        cases: List of Case objects to process
        linker: CitationLinker instance (creates one if not provided)
        extractor: CitationExtractor instance (creates one if not provided)
        use_tf_idf: Whether to use TF-IDF weighting for edges
        remove_self_citations: Whether to remove self-citations
        show_progress: Whether to show progress bars

    Returns:
        List of CitationEdge objects
    """
    start_time = datetime.now()
    stats = EdgeBuildStats()
    stats.total_cases = len(cases)

    # Initialize components
    extractor = extractor or CitationExtractor()
    linker = linker or CitationLinker()

    # Create case ID to year mapping
    case_years: Dict[str, int] = {}
    for case in cases:
        case_years[case.id] = case.date.year

    # Step 1: Extract citations from all cases
    logger.info(f"Extracting citations from {len(cases)} cases...")
    case_citations = extractor.extract_batch(cases, show_progress=show_progress)

    extraction_stats = extractor.get_stats()
    stats.total_citations_extracted = extraction_stats.total_citations
    stats.unique_citations = extraction_stats.unique_citations

    # Collect all unique citations for linking
    all_citations: Set[str] = set()
    for citations in case_citations.values():
        for c in citations:
            all_citations.add(c.normalized)

    logger.info(f"Found {len(all_citations)} unique citations to link")

    # Step 2: Link citations to case IDs
    logger.info("Linking citations to case IDs...")
    citation_to_case_id = await linker.link_batch(
        list(all_citations),
        show_progress=show_progress,
    )

    linking_stats = linker.get_stats()
    stats.successful_links = linking_stats.get("successful_links", 0)

    # Step 3: Compute TF-IDF weights if requested
    weights: Optional[Dict[str, Dict[str, float]]] = None
    if use_tf_idf:
        logger.info("Computing TF-IDF weights...")
        weights = compute_tf_idf_weights(case_citations)

    # Step 4: Build edges
    logger.info("Building edge list...")
    edges: List[CitationEdge] = []
    seen_edges: Set[tuple] = set()  # For deduplication

    for source_case_id, citations in case_citations.items():
        source_year = case_years.get(source_case_id)

        for citation in citations:
            target_case_id = citation_to_case_id.get(citation.normalized)

            if not target_case_id:
                continue  # Couldn't link this citation

            # Skip self-citations if requested
            if remove_self_citations and source_case_id == target_case_id:
                stats.self_citations_removed += 1
                continue

            # Deduplicate edges (same source -> target via different citations)
            edge_key = (source_case_id, target_case_id)
            if edge_key in seen_edges:
                stats.duplicate_edges_removed += 1
                continue
            seen_edges.add(edge_key)

            # Get weight
            weight = 1.0
            if weights and source_case_id in weights:
                weight = weights[source_case_id].get(citation.normalized, 1.0)

            # Create edge
            edge = CitationEdge(
                source_case_id=source_case_id,
                target_case_id=target_case_id,
                citation_text=citation.normalized,
                weight=weight,
                citation_type=citation.classification,
                source_year=source_year,
            )
            edges.append(edge)

    stats.edges_created = len(edges)
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()

    logger.info(f"Created {len(edges)} citation edges")
    logger.info(f"Stats: {stats.model_dump_json(indent=2)}")

    return edges


def save_edges_csv(
    edges: List[CitationEdge],
    path: Optional[Path] = None,
    include_header: bool = True,
) -> Path:
    """Save edges to CSV file for Neo4j import.

    The CSV format is compatible with Neo4j's LOAD CSV command:
    source_case_id,target_case_id,citation_text,weight,citation_type

    Args:
        edges: List of CitationEdge objects
        path: Output path (defaults to CITATIONS_DIR/edges.csv)
        include_header: Whether to include CSV header row

    Returns:
        Path to the created CSV file
    """
    path = path or CITATIONS_DIR / "edges.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if include_header:
            writer.writerow([
                "source_case_id",
                "target_case_id",
                "citation_text",
                "weight",
                "citation_type",
                "source_year",
            ])

        for edge in edges:
            writer.writerow([
                edge.source_case_id,
                edge.target_case_id,
                edge.citation_text,
                edge.weight,
                edge.citation_type or "",
                edge.source_year or "",
            ])

    logger.info(f"Saved {len(edges)} edges to {path}")
    return path


def load_edges_csv(path: Path) -> List[CitationEdge]:
    """Load edges from CSV file.

    Args:
        path: Path to CSV file

    Returns:
        List of CitationEdge objects
    """
    edges = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            edge = CitationEdge(
                source_case_id=row["source_case_id"],
                target_case_id=row["target_case_id"],
                citation_text=row["citation_text"],
                weight=float(row.get("weight", 1.0)),
                citation_type=row.get("citation_type") or None,
                source_year=int(row["source_year"]) if row.get("source_year") else None,
            )
            edges.append(edge)

    return edges


def save_edge_stats(
    edges: List[CitationEdge],
    path: Optional[Path] = None,
) -> Path:
    """Save edge statistics to JSON file.

    Args:
        edges: List of CitationEdge objects
        path: Output path (defaults to CITATIONS_DIR/stats.json)

    Returns:
        Path to the created JSON file
    """
    path = path or CITATIONS_DIR / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    citation_types = defaultdict(int)
    years = []
    weights = []

    source_case_ids: Set[str] = set()
    target_case_ids: Set[str] = set()

    for edge in edges:
        source_case_ids.add(edge.source_case_id)
        target_case_ids.add(edge.target_case_id)

        if edge.citation_type:
            citation_types[edge.citation_type] += 1
        if edge.source_year:
            years.append(edge.source_year)
        weights.append(edge.weight)

    # Calculate derived stats
    avg_weight = sum(weights) / len(weights) if weights else 0
    year_range = (min(years), max(years)) if years else (None, None)

    # Out-degree distribution (how many cases each case cites)
    out_degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        out_degree[edge.source_case_id] += 1

    # In-degree distribution (how many times each case is cited)
    in_degree: Dict[str, int] = defaultdict(int)
    for edge in edges:
        in_degree[edge.target_case_id] += 1

    stats = {
        "total_edges": len(edges),
        "unique_source_cases": len(source_case_ids),
        "unique_target_cases": len(target_case_ids),
        "citation_types": dict(citation_types),
        "year_range": year_range,
        "avg_weight": avg_weight,
        "out_degree_stats": {
            "min": min(out_degree.values()) if out_degree else 0,
            "max": max(out_degree.values()) if out_degree else 0,
            "avg": sum(out_degree.values()) / len(out_degree) if out_degree else 0,
        },
        "in_degree_stats": {
            "min": min(in_degree.values()) if in_degree else 0,
            "max": max(in_degree.values()) if in_degree else 0,
            "avg": sum(in_degree.values()) / len(in_degree) if in_degree else 0,
        },
        "most_cited_cases": sorted(
            in_degree.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20],
        "timestamp": datetime.now().isoformat(),
    }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved edge statistics to {path}")
    return path


def generate_neo4j_import_script(
    edges_csv_path: Path,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a Cypher script for importing edges into Neo4j.

    Args:
        edges_csv_path: Path to the edges CSV file
        output_path: Optional path to save the script

    Returns:
        Cypher script as string
    """
    script = f"""// Neo4j Citation Import Script
// Generated by LegalGPT Citation Extractor

// First, ensure indexes exist for performance
CREATE INDEX case_id_index IF NOT EXISTS FOR (c:Case) ON (c.id);

// Import citation edges
LOAD CSV WITH HEADERS FROM 'file:///{edges_csv_path.absolute()}' AS row
MATCH (source:Case {{id: row.source_case_id}})
MATCH (target:Case {{id: row.target_case_id}})
CREATE (source)-[r:CITES {{
    citation_text: row.citation_text,
    weight: toFloat(row.weight),
    citation_type: row.citation_type,
    source_year: toInteger(row.source_year)
}}]->(target);

// Optional: Create citation statistics
MATCH (c:Case)
SET c.citation_count = size((c)-[:CITES]->());

MATCH (c:Case)
SET c.cited_by_count = size((c)<-[:CITES]-());
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(script)
        logger.info(f"Saved Neo4j import script to {output_path}")

    return script


async def build_and_save_edges(
    cases: List["Case"],
    output_dir: Optional[Path] = None,
    use_tf_idf: bool = False,
) -> Dict[str, Any]:
    """Complete pipeline to build edges and save all output files.

    Args:
        cases: List of Case objects
        output_dir: Output directory (defaults to CITATIONS_DIR)
        use_tf_idf: Whether to use TF-IDF weighting

    Returns:
        Dictionary with paths to created files and statistics
    """
    output_dir = output_dir or CITATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    linker = CitationLinker()
    extractor = CitationExtractor()

    # Build edges
    edges = await build_edge_list(
        cases,
        linker=linker,
        extractor=extractor,
        use_tf_idf=use_tf_idf,
    )

    # Save outputs
    edges_path = save_edges_csv(edges, output_dir / "edges.csv")
    stats_path = save_edge_stats(edges, output_dir / "stats.json")

    # Save unlinked citations
    linker.save_unlinked(output_dir / "unlinked.json")

    # Generate Neo4j import script
    cypher_path = output_dir / "import_edges.cypher"
    generate_neo4j_import_script(edges_path, cypher_path)

    return {
        "edges_csv": edges_path,
        "stats_json": stats_path,
        "unlinked_json": output_dir / "unlinked.json",
        "cypher_script": cypher_path,
        "total_edges": len(edges),
        "extraction_stats": extractor.get_stats().model_dump(),
        "linking_stats": linker.get_stats(),
    }


if __name__ == "__main__":
    # Test edge building with sample data
    async def test():
        from src.data.case_schema import Case, Court
        from datetime import datetime

        # Create sample cases
        sample_cases = [
            Case(
                id="case-001",
                name="Test v. Example",
                date=datetime(2020, 1, 1),
                court=Court(slug="us", name="Supreme Court"),
                text="The Court in Brown v. Board of Education, 347 U.S. 483 (1954), held that...",
            ),
            Case(
                id="case-002",
                name="Another v. Case",
                date=datetime(2021, 6, 15),
                court=Court(slug="us", name="Supreme Court"),
                text="Following Roe v. Wade, 410 U.S. 113 (1973), and Brown, 347 U.S. 483...",
            ),
        ]

        print("Building edge list from sample cases...")
        results = await build_and_save_edges(sample_cases)

        print(f"\nResults:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    asyncio.run(test())
