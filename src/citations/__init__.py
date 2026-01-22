"""Citation extraction and linking module.

This module provides tools for:
- Extracting legal citations from case text using regex patterns
- Linking citations to CourtListener case IDs
- Building citation graph edges for Neo4j import
- Validating extraction and linking quality
"""

from src.citations.patterns import (
    CITATION_PATTERNS,
    normalize_citation,
    extract_all_citations,
    classify_citation,
)
from src.citations.extractor import (
    CitationExtractor,
    ExtractedCitation,
    ExtractionStats,
)
from src.citations.linker import CitationLinker, LinkResult, link_citations
from src.citations.graph_edges import (
    build_edge_list as build_citation_edges,
    CitationEdge,
    save_edges_csv,
    load_edges_csv,
)
from src.citations.validation import (
    validate_extraction,
    run_full_validation,
    ValidationReport,
)

__all__ = [
    # Patterns
    "CITATION_PATTERNS",
    "normalize_citation",
    "extract_all_citations",
    "classify_citation",
    # Extractor
    "CitationExtractor",
    "ExtractedCitation",
    "ExtractionStats",
    # Linker
    "CitationLinker",
    "LinkResult",
    "link_citations",
    # Graph edges
    "build_citation_edges",
    "CitationEdge",
    "save_edges_csv",
    "load_edges_csv",
    # Validation
    "validate_extraction",
    "run_full_validation",
    "ValidationReport",
]
