#!/usr/bin/env python3
"""Citation Graph Pipeline - Agent A

Extracts citations from all cases, links them to CourtListener,
and builds the citation edge list for Neo4j import.

Usage:
    python scripts/run_citation_pipeline.py
    python scripts/run_citation_pipeline.py --skip-linking  # Use cached links
    python scripts/run_citation_pipeline.py --max-cases 50  # Test with subset
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CITATIONS_DIR, PROCESSED_DIR
from src.data.storage import load_cases_parquet
from src.data.courtlistener_client import CourtListenerClient
from src.citations.extractor import CitationExtractor
from src.citations.linker import CitationLinker
from src.citations.graph_edges import (
    build_edge_list,
    save_edges_csv,
    save_edge_stats,
    generate_neo4j_import_script,
)
from src.citations.validation import (
    validate_extraction,
    save_validation_report,
    print_validation_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_citation_pipeline(
    max_cases: int | None = None,
    skip_linking: bool = False,
    use_tf_idf: bool = True,
) -> dict:
    """Run the full citation extraction and linking pipeline.

    Args:
        max_cases: Maximum cases to process (None for all)
        skip_linking: Skip API linking, use cached results only
        use_tf_idf: Use TF-IDF weighting for edges

    Returns:
        Dictionary with pipeline results
    """
    results = {
        "total_cases": 0,
        "total_citations": 0,
        "unique_citations": 0,
        "linked_citations": 0,
        "edges_created": 0,
        "timestamp": datetime.now().isoformat(),
    }

    # =========================================================
    # STEP 1: Load cases
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading cases from parquet")
    logger.info("=" * 60)

    cases_path = PROCESSED_DIR / "scdb_matched.parquet"
    if not cases_path.exists():
        logger.error(f"Cases file not found: {cases_path}")
        logger.error("Run the data pipeline first: python scripts/run_data_pipeline.py")
        return results

    cases = load_cases_parquet(cases_path)
    logger.info(f"Loaded {len(cases)} cases")

    if max_cases:
        cases = cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases")

    results["total_cases"] = len(cases)

    # =========================================================
    # STEP 2: Extract citations
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting citations from case text")
    logger.info("=" * 60)

    extractor = CitationExtractor(
        context_window=100,
        deduplicate=True,
        max_workers=4,
    )

    # Extract citations from all cases
    case_citations = extractor.extract_batch(cases, show_progress=True)

    # Get stats
    extraction_stats = extractor.get_stats()
    results["total_citations"] = extraction_stats.total_citations
    results["unique_citations"] = extraction_stats.unique_citations

    logger.info(f"Extracted {extraction_stats.total_citations} total citations")
    logger.info(f"Unique citations: {extraction_stats.unique_citations}")
    logger.info(f"Avg per case: {extraction_stats.avg_citations_per_case:.1f}")

    # Show citation type breakdown
    logger.info("Citations by type:")
    for ctype, count in sorted(extraction_stats.citations_by_type.items(), key=lambda x: -x[1]):
        logger.info(f"  {ctype}: {count}")

    # Collect all unique citations for linking
    all_citations = set()
    for citations in case_citations.values():
        for c in citations:
            all_citations.add(c.normalized)

    logger.info(f"Found {len(all_citations)} unique citations to link")

    # =========================================================
    # STEP 3: Link citations to CourtListener
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Linking citations to CourtListener case IDs")
    logger.info("=" * 60)

    CITATIONS_DIR.mkdir(parents=True, exist_ok=True)

    if skip_linking:
        logger.info("Skipping API linking (using cached results only)")
        linker = CitationLinker(client=None)
    else:
        async with CourtListenerClient() as client:
            linker = CitationLinker(client=client, max_concurrent=3)

            # Link all citations
            citation_to_case = await linker.link_batch(
                list(all_citations),
                show_progress=True,
            )

            # Get linking stats
            link_stats = linker.get_stats()
            results["linked_citations"] = link_stats.get("successful_links", 0)

            logger.info(f"Successfully linked: {link_stats.get('successful_links', 0)}")
            logger.info(f"Failed to link: {link_stats.get('failed_links', 0)}")
            logger.info(f"Success rate: {link_stats.get('success_rate', 0)*100:.1f}%")

            # Save unlinked citations
            linker.save_unlinked(CITATIONS_DIR / "unlinked.json")

    # =========================================================
    # STEP 4: Build citation edges
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Building citation edge list")
    logger.info("=" * 60)

    # Re-initialize linker with cached results for edge building
    linker = CitationLinker(client=None)

    edges = await build_edge_list(
        cases,
        linker=linker,
        extractor=extractor,
        use_tf_idf=use_tf_idf,
        remove_self_citations=True,
        show_progress=True,
    )

    results["edges_created"] = len(edges)
    logger.info(f"Created {len(edges)} citation edges")

    # =========================================================
    # STEP 5: Save outputs
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Saving outputs")
    logger.info("=" * 60)

    # Save edges CSV
    edges_path = save_edges_csv(edges, CITATIONS_DIR / "edges.csv")
    logger.info(f"Saved edges to {edges_path}")

    # Save edge statistics
    stats_path = save_edge_stats(edges, CITATIONS_DIR / "stats.json")
    logger.info(f"Saved stats to {stats_path}")

    # Generate Neo4j import script
    cypher_path = CITATIONS_DIR / "import_edges.cypher"
    generate_neo4j_import_script(edges_path, cypher_path)
    logger.info(f"Saved Neo4j import script to {cypher_path}")

    # Run validation on sample
    logger.info("Running extraction validation...")
    validation_report = validate_extraction(cases[:min(50, len(cases))], extractor)
    save_validation_report(
        type('ValidationReport', (), {'extraction': validation_report, 'linking': None, 'edge_stats': None, 'recommendations': [], 'overall_quality_score': 0.0, 'timestamp': datetime.now().isoformat(), 'model_dump': lambda self: {'extraction': self.extraction.model_dump(), 'linking': None, 'edge_stats': None, 'recommendations': [], 'overall_quality_score': 0.0, 'timestamp': self.timestamp}})(),
        CITATIONS_DIR / "validation_report.json"
    )

    # Save pipeline results
    results_path = CITATIONS_DIR / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved pipeline results to {results_path}")

    # =========================================================
    # SUMMARY
    # =========================================================
    logger.info("=" * 60)
    logger.info("CITATION PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Cases processed: {results['total_cases']}")
    logger.info(f"Citations extracted: {results['total_citations']}")
    logger.info(f"Unique citations: {results['unique_citations']}")
    logger.info(f"Citations linked: {results['linked_citations']}")
    logger.info(f"Edges created: {results['edges_created']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the citation extraction and linking pipeline"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Max cases to process (default: all)"
    )
    parser.add_argument(
        "--skip-linking",
        action="store_true",
        help="Skip API linking, use cached results only"
    )
    parser.add_argument(
        "--no-tf-idf",
        action="store_true",
        help="Don't use TF-IDF weighting for edges"
    )

    args = parser.parse_args()

    try:
        results = asyncio.run(run_citation_pipeline(
            max_cases=args.max_cases,
            skip_linking=args.skip_linking,
            use_tf_idf=not args.no_tf_idf,
        ))

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for key, value in results.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
