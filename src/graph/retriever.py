"""Case retrieval API combining GraphSAGE embeddings and citation graph."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from neo4j import GraphDatabase
from pydantic import BaseModel

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    DEFAULT_TOP_K,
    EMBEDDINGS_DIR,
)


class RetrievedCase(BaseModel):
    """A retrieved case with relevance information."""

    case_id: str
    name: str
    text: str
    similarity_score: float
    citation_distance: int  # Hops in citation graph (-1 if not connected)
    date: Optional[str] = None
    court: Optional[str] = None
    citation: Optional[str] = None
    outcome: Optional[str] = None
    retrieval_method: str = "hybrid"  # 'graphsage', 'citation', 'hybrid'

    class Config:
        """Pydantic config."""
        frozen = True


class CaseRetriever:
    """
    Retriever for finding similar cases using GraphSAGE embeddings
    and citation graph proximity.
    """

    def __init__(
        self,
        neo4j_driver=None,
        embeddings_path: Optional[Path] = None,
        embedding_weight: float = 0.6,
        citation_weight: float = 0.4,
    ):
        """
        Initialize the case retriever.

        Args:
            neo4j_driver: Neo4j driver instance
            embeddings_path: Path to GraphSAGE embeddings file
            embedding_weight: Weight for embedding similarity (0-1)
            citation_weight: Weight for citation proximity (0-1)
        """
        if neo4j_driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._own_driver = True
        else:
            self._driver = neo4j_driver
            self._own_driver = False

        self.embedding_weight = embedding_weight
        self.citation_weight = citation_weight

        # Load GraphSAGE embeddings if available
        self.embeddings_path = embeddings_path or (EMBEDDINGS_DIR / "graphsage_embeddings.pt")
        self.node_ids: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.node_id_to_idx: Dict[str, int] = {}

        if self.embeddings_path.exists():
            self._load_embeddings()

    def _load_embeddings(self):
        """Load GraphSAGE embeddings from disk."""
        data = torch.load(self.embeddings_path, map_location="cpu")
        self.node_ids = data["node_ids"]
        self.embeddings = data["embeddings"]
        self.node_id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids)}
        print(f"Loaded {len(self.node_ids)} embeddings from {self.embeddings_path}")

    def close(self):
        """Close the Neo4j driver if we own it."""
        if self._own_driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_similar_cases(
        self,
        query_case_id: str,
        k: int = DEFAULT_TOP_K,
        method: str = "hybrid",
        include_text: bool = True,
        max_text_length: int = 10000,
    ) -> List[RetrievedCase]:
        """
        Get the k most similar cases to the query case.

        Args:
            query_case_id: ID of the query case
            k: Number of cases to retrieve
            method: Retrieval method ('graphsage', 'citation', 'hybrid')
            include_text: Whether to include full case text
            max_text_length: Maximum text length to return

        Returns:
            List of RetrievedCase objects sorted by relevance
        """
        if method == "graphsage":
            return self._retrieve_by_embedding(
                query_case_id, k, include_text, max_text_length
            )
        elif method == "citation":
            return self._retrieve_by_citation(
                query_case_id, k, include_text, max_text_length
            )
        elif method == "hybrid":
            return self._retrieve_hybrid(
                query_case_id, k, include_text, max_text_length
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def _compute_embedding_similarity_local(
        self,
        query_case_id: str,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[tuple]:
        """Compute embedding similarity using locally loaded embeddings."""
        if self.embeddings is None:
            return []

        if query_case_id not in self.node_id_to_idx:
            return []

        query_idx = self.node_id_to_idx[query_case_id]
        query_emb = self.embeddings[query_idx].unsqueeze(0)

        if candidate_ids:
            candidate_indices = [
                self.node_id_to_idx[cid]
                for cid in candidate_ids
                if cid in self.node_id_to_idx and cid != query_case_id
            ]
            if not candidate_indices:
                return []
            candidate_embs = self.embeddings[candidate_indices]
            ids_to_use = [cid for cid in candidate_ids
                          if cid in self.node_id_to_idx and cid != query_case_id]
        else:
            candidate_embs = self.embeddings
            ids_to_use = self.node_ids

        similarities = torch.nn.functional.cosine_similarity(
            query_emb, candidate_embs, dim=1
        ).numpy()

        results = [
            (cid, float(sim))
            for cid, sim in zip(ids_to_use, similarities)
            if cid != query_case_id
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _retrieve_by_embedding(
        self,
        query_case_id: str,
        k: int,
        include_text: bool,
        max_text_length: int,
    ) -> List[RetrievedCase]:
        """Retrieve cases by GraphSAGE embedding similarity."""
        text_field = f"substring(c.text, 0, {max_text_length})" if include_text else "''"

        with self._driver.session() as session:
            # First check if we have GraphSAGE embeddings
            check_result = session.run("""
                MATCH (q:Case {id: $query_id})
                RETURN q.graphsage_embedding IS NOT NULL AS has_graphsage
            """, query_id=query_case_id)
            record = check_result.single()

            if record and record["has_graphsage"]:
                # Use GraphSAGE embeddings
                result = session.run(f"""
                    MATCH (q:Case {{id: $query_id}})
                    MATCH (c:Case)
                    WHERE c.id <> q.id AND c.graphsage_embedding IS NOT NULL
                    WITH q, c,
                         gds.similarity.cosine(q.graphsage_embedding, c.graphsage_embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $k
                    RETURN c.id AS id,
                           c.name AS name,
                           toString(c.date) AS date,
                           c.court AS court,
                           c.citation AS citation,
                           c.outcome AS outcome,
                           {text_field} AS text,
                           similarity AS score
                """, query_id=query_case_id, k=k)
            else:
                # Fall back to text embeddings
                result = session.run(f"""
                    MATCH (q:Case {{id: $query_id}})
                    MATCH (c:Case)
                    WHERE c.id <> q.id AND c.embedding IS NOT NULL
                    WITH q, c,
                         gds.similarity.cosine(q.embedding, c.embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $k
                    RETURN c.id AS id,
                           c.name AS name,
                           toString(c.date) AS date,
                           c.court AS court,
                           c.citation AS citation,
                           c.outcome AS outcome,
                           {text_field} AS text,
                           similarity AS score
                """, query_id=query_case_id, k=k)

            return [
                RetrievedCase(
                    case_id=r["id"],
                    name=r["name"],
                    date=r["date"],
                    court=r["court"],
                    citation=r["citation"],
                    outcome=r["outcome"],
                    text=r["text"] or "",
                    similarity_score=r["score"],
                    citation_distance=-1,
                    retrieval_method="graphsage",
                )
                for r in result
            ]

    def _retrieve_by_citation(
        self,
        query_case_id: str,
        k: int,
        include_text: bool,
        max_text_length: int,
    ) -> List[RetrievedCase]:
        """Retrieve cases by citation graph proximity."""
        text_field = f"substring(c.text, 0, {max_text_length})" if include_text else "''"

        with self._driver.session() as session:
            # Find cases within 2 hops in citation graph
            result = session.run(f"""
                MATCH (q:Case {{id: $query_id}})
                CALL {{
                    WITH q
                    MATCH (q)-[:CITES]->(c:Case)
                    RETURN c, 1 AS distance
                    UNION
                    WITH q
                    MATCH (q)<-[:CITES]-(c:Case)
                    RETURN c, 1 AS distance
                    UNION
                    WITH q
                    MATCH (q)-[:CITES]->()-[:CITES]->(c:Case)
                    WHERE c.id <> q.id
                    RETURN c, 2 AS distance
                    UNION
                    WITH q
                    MATCH (q)<-[:CITES]-()<-[:CITES]-(c:Case)
                    WHERE c.id <> q.id
                    RETURN c, 2 AS distance
                }}
                WITH c, min(distance) AS min_distance
                ORDER BY min_distance, c.date DESC
                LIMIT $k
                RETURN c.id AS id,
                       c.name AS name,
                       toString(c.date) AS date,
                       c.court AS court,
                       c.citation AS citation,
                       c.outcome AS outcome,
                       {text_field} AS text,
                       min_distance AS distance,
                       1.0 / min_distance AS score
            """, query_id=query_case_id, k=k)

            return [
                RetrievedCase(
                    case_id=r["id"],
                    name=r["name"],
                    date=r["date"],
                    court=r["court"],
                    citation=r["citation"],
                    outcome=r["outcome"],
                    text=r["text"] or "",
                    similarity_score=r["score"],
                    citation_distance=r["distance"],
                    retrieval_method="citation",
                )
                for r in result
            ]

    def _retrieve_hybrid(
        self,
        query_case_id: str,
        k: int,
        include_text: bool,
        max_text_length: int,
    ) -> List[RetrievedCase]:
        """Retrieve cases using combined embedding similarity and citation proximity."""
        # Get more candidates from each method
        candidate_k = k * 3

        embedding_results = self._retrieve_by_embedding(
            query_case_id, candidate_k, include_text, max_text_length
        )
        citation_results = self._retrieve_by_citation(
            query_case_id, candidate_k, include_text, max_text_length
        )

        # Combine scores
        case_scores: Dict[str, float] = {}
        case_data: Dict[str, RetrievedCase] = {}
        citation_distances: Dict[str, int] = {}

        for case in embedding_results:
            case_scores[case.case_id] = self.embedding_weight * case.similarity_score
            case_data[case.case_id] = case
            citation_distances[case.case_id] = case.citation_distance

        for case in citation_results:
            citation_score = self.citation_weight * case.similarity_score
            if case.case_id in case_scores:
                case_scores[case.case_id] += citation_score
                # Update citation distance if we have it
                if case.citation_distance >= 0:
                    citation_distances[case.case_id] = case.citation_distance
            else:
                case_scores[case.case_id] = citation_score
                case_data[case.case_id] = case
                citation_distances[case.case_id] = case.citation_distance

        # Sort by combined score
        sorted_ids = sorted(case_scores.keys(), key=lambda x: case_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:k]:
            base_case = case_data[cid]
            # Create new RetrievedCase with updated scores
            results.append(RetrievedCase(
                case_id=base_case.case_id,
                name=base_case.name,
                text=base_case.text,
                similarity_score=case_scores[cid],
                citation_distance=citation_distances[cid],
                date=base_case.date,
                court=base_case.court,
                citation=base_case.citation,
                outcome=base_case.outcome,
                retrieval_method="hybrid",
            ))

        return results

    def get_case_by_id(self, case_id: str, include_text: bool = True) -> Optional[RetrievedCase]:
        """
        Get a single case by ID.

        Args:
            case_id: The case ID to retrieve
            include_text: Whether to include full text

        Returns:
            RetrievedCase or None if not found
        """
        text_field = "c.text" if include_text else "''"

        with self._driver.session() as session:
            result = session.run(f"""
                MATCH (c:Case {{id: $case_id}})
                RETURN c.id AS id,
                       c.name AS name,
                       toString(c.date) AS date,
                       c.court AS court,
                       c.citation AS citation,
                       c.outcome AS outcome,
                       {text_field} AS text
            """, case_id=case_id)

            record = result.single()
            if record is None:
                return None

            return RetrievedCase(
                case_id=record["id"],
                name=record["name"],
                date=record["date"],
                court=record["court"],
                citation=record["citation"],
                outcome=record["outcome"],
                text=record["text"] or "",
                similarity_score=1.0,
                citation_distance=0,
                retrieval_method="direct",
            )

    def get_citing_cases(self, case_id: str, max_depth: int = 2) -> List[str]:
        """
        Get cases that cite the given case (directly or transitively).

        Args:
            case_id: ID of the case being cited
            max_depth: Maximum citation depth to traverse

        Returns:
            List of case IDs that cite this case
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (target:Case {{id: $case_id}})
                MATCH (citing:Case)-[:CITES*1..{max_depth}]->(target)
                WHERE citing.id <> $case_id
                RETURN DISTINCT citing.id AS id
                ORDER BY id
                """,
                case_id=case_id,
            )
            return [record["id"] for record in result]

    def get_cited_cases(self, case_id: str, max_depth: int = 2) -> List[str]:
        """
        Get cases cited by the given case (directly or transitively).

        Args:
            case_id: ID of the citing case
            max_depth: Maximum citation depth to traverse

        Returns:
            List of case IDs cited by this case
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (source:Case {{id: $case_id}})
                MATCH (source)-[:CITES*1..{max_depth}]->(cited:Case)
                WHERE cited.id <> $case_id
                RETURN DISTINCT cited.id AS id
                ORDER BY id
                """,
                case_id=case_id,
            )
            return [record["id"] for record in result]

    def get_citing_cases_detailed(self, case_id: str, limit: int = 100) -> List[RetrievedCase]:
        """Get cases that cite the given case with full details."""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (c:Case)-[r:CITES]->(target:Case {id: $case_id})
                RETURN c.id AS id,
                       c.name AS name,
                       toString(c.date) AS date,
                       c.court AS court,
                       c.citation AS citation,
                       c.outcome AS outcome
                ORDER BY c.date DESC
                LIMIT $limit
            """, case_id=case_id, limit=limit)

            return [
                RetrievedCase(
                    case_id=r["id"],
                    name=r["name"],
                    date=r["date"],
                    court=r["court"],
                    citation=r["citation"],
                    outcome=r["outcome"],
                    text="",
                    similarity_score=1.0,
                    citation_distance=1,
                    retrieval_method="citation",
                )
                for r in result
            ]

    def get_cited_cases_detailed(self, case_id: str, limit: int = 100) -> List[RetrievedCase]:
        """Get cases cited by the given case with full details."""
        with self._driver.session() as session:
            result = session.run("""
                MATCH (source:Case {id: $case_id})-[r:CITES]->(c:Case)
                RETURN c.id AS id,
                       c.name AS name,
                       toString(c.date) AS date,
                       c.court AS court,
                       c.citation AS citation,
                       c.outcome AS outcome
                ORDER BY c.date DESC
                LIMIT $limit
            """, case_id=case_id, limit=limit)

            return [
                RetrievedCase(
                    case_id=r["id"],
                    name=r["name"],
                    date=r["date"],
                    court=r["court"],
                    citation=r["citation"],
                    outcome=r["outcome"],
                    text="",
                    similarity_score=1.0,
                    citation_distance=1,
                    retrieval_method="citation",
                )
                for r in result
            ]


# Convenience function for simple usage
def get_similar_cases(
    query_case_id: str,
    k: int = DEFAULT_TOP_K,
    method: str = "hybrid",
    include_text: bool = True,
) -> List[RetrievedCase]:
    """
    Get similar cases to the query case.

    This is the main interface for Agent 4 (Model) to use.

    Args:
        query_case_id: ID of the query case
        k: Number of cases to retrieve
        method: Retrieval method ('graphsage', 'citation', 'hybrid')
        include_text: Whether to include full case text

    Returns:
        List of RetrievedCase objects sorted by relevance

    Example:
        >>> similar = get_similar_cases("12345", k=5)
        >>> for case in similar:
        ...     print(f"{case.name}: {case.similarity_score:.3f}")
    """
    with CaseRetriever() as retriever:
        return retriever.get_similar_cases(
            query_case_id, k=k, method=method, include_text=include_text
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retriever.py <case_id> [k] [method]")
        print("  case_id: ID of the query case")
        print("  k: Number of cases to retrieve (default: 10)")
        print("  method: 'graphsage', 'citation', or 'hybrid' (default: 'hybrid')")
        sys.exit(1)

    case_id = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    method = sys.argv[3] if len(sys.argv) > 3 else "hybrid"

    print(f"Retrieving top {k} similar cases to {case_id} using {method} method...\n")

    try:
        similar_cases = get_similar_cases(case_id, k=k, method=method, include_text=False)

        for i, case in enumerate(similar_cases, 1):
            print(f"{i}. {case.name}")
            print(f"   ID: {case.case_id}")
            print(f"   Citation: {case.citation}")
            print(f"   Court: {case.court}")
            print(f"   Outcome: {case.outcome}")
            print(f"   Score: {case.similarity_score:.4f}")
            if case.citation_distance >= 0:
                print(f"   Citation distance: {case.citation_distance} hops")
            print()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
