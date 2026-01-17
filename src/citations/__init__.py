"""Citation extraction module for LegalGPT.

This module provides tools for extracting legal citations from case text,
linking them to case IDs via the CAP API, and building graph edge lists.

Main Components:
- patterns: Regex patterns for various citation formats
- extractor: CitationExtractor for extracting citations from text
- linker: CitationLinker for linking citations to CAP case IDs
- graph_edges: Tools for building citation graph edge lists
- validation: Quality validation and reporting

Usage:
    from src.citations import CitationExtractor, CitationLinker
    from src.citations import build_edge_list, save_edges_csv

    # Extract citations
    extractor = CitationExtractor()
    citations = extractor.extract(case_text)

    # Link to case IDs
    linker = CitationLinker(cap_client)
    case_id = await linker.link_citation("347 U.S. 483")

    # Build and save edge list
    edges = await build_edge_list(cases, linker)
    save_edges_csv(edges, Path("edges.csv"))

Output Files:
- data/citations/edges.csv: Edge list for Neo4j import
- data/citations/stats.json: Extraction and linking statistics
- data/citations/unlinked.json: Citations that couldn't be linked
- data/citations/link_cache.json: Cached citation->ID mappings
"""

from .patterns import (
    CITATION_PATTERNS,
    extract_all_citations,
    normalize_citation,
    classify_citation,
    extract_citation_components,
    is_valid_citation,
    split_parallel_citations,
)

from .extractor import (
    CitationExtractor,
    ExtractedCitation,
    ExtractionStats,
    extract_citations_from_text,
)

from .linker import (
    CitationLinker,
    LinkResult,
    LinkingStats,
    link_citations,
)

from .graph_edges import (
    CitationEdge,
    EdgeBuildStats,
    build_edge_list,
    save_edges_csv,
    load_edges_csv,
    save_edge_stats,
    build_and_save_edges,
    compute_tf_idf_weights,
    generate_neo4j_import_script,
)

from .validation import (
    ExtractionReport,
    LinkingReport,
    ValidationReport,
    validate_extraction,
    validate_linking,
    run_full_validation,
    save_validation_report,
    print_validation_summary,
)

__all__ = [
    # Patterns
    "CITATION_PATTERNS",
    "extract_all_citations",
    "normalize_citation",
    "classify_citation",
    "extract_citation_components",
    "is_valid_citation",
    "split_parallel_citations",
    # Extractor
    "CitationExtractor",
    "ExtractedCitation",
    "ExtractionStats",
    "extract_citations_from_text",
    # Linker
    "CitationLinker",
    "LinkResult",
    "LinkingStats",
    "link_citations",
    # Graph Edges
    "CitationEdge",
    "EdgeBuildStats",
    "build_edge_list",
    "save_edges_csv",
    "load_edges_csv",
    "save_edge_stats",
    "build_and_save_edges",
    "compute_tf_idf_weights",
    "generate_neo4j_import_script",
    # Validation
    "ExtractionReport",
    "LinkingReport",
    "ValidationReport",
    "validate_extraction",
    "validate_linking",
    "run_full_validation",
    "save_validation_report",
    "print_validation_summary",
]
