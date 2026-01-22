#!/usr/bin/env python3
"""Main entry point for Data Pipeline.

This script orchestrates the full data pipeline:
1. Download SCDB (Supreme Court Database) - outcome labels
2. Match SCDB cases to CourtListener for full text
3. Preprocess text
4. Create train/val/test splits
5. Save to storage

Usage:
    python scripts/run_data_pipeline.py --max-cases 50   # Dev sample
    python scripts/run_data_pipeline.py --max-cases 500  # Larger sample
    python scripts/run_data_pipeline.py                  # Full dataset (~9K)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR
from src.data.scdb_loader import SCDBLoader, create_stratified_splits
from src.data.courtlistener_client import CourtListenerClient
from src.data.preprocessing import clean_legal_text
from src.data.storage import save_cases_parquet, save_splits, get_dataset_stats
from src.data.case_schema import Case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def match_scdb_to_courtlistener(
    scdb_cases,
    client: CourtListenerClient,
    max_concurrent: int = 3
) -> list[Case]:
    """Match SCDB cases to CourtListener by citation.

    Args:
        scdb_cases: List of SCDBCase objects
        client: CourtListenerClient instance
        max_concurrent: Max concurrent requests

    Returns:
        List of Case objects with text and outcome
    """
    from tqdm.asyncio import tqdm
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    matched = []
    failed = 0

    async def match_one(scdb_case):
        nonlocal failed
        citation = scdb_case.primary_citation
        if not citation:
            return None

        async with semaphore:
            try:
                cl_case = await client.search_by_citation(citation, save_to_disk=True)
                if cl_case:
                    case = cl_case.to_case()
                    case.outcome = scdb_case.outcome
                    case.scdb_id = scdb_case.case_id
                    return case
            except Exception as e:
                logger.debug(f"Failed to match {citation}: {e}")
                failed += 1
        return None

    # Run with progress bar
    tasks = [match_one(sc) for sc in scdb_cases]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Matching to CourtListener"):
        result = await coro
        if result:
            matched.append(result)

    logger.info(f"Matched {len(matched)}/{len(scdb_cases)} cases ({failed} failures)")
    return matched


async def run_pipeline(
    max_cases: int | None = None,
    force_download: bool = False,
    skip_matching: bool = False
) -> dict:
    """Run the full data pipeline.

    Args:
        max_cases: Maximum cases to process (None for all)
        force_download: Force re-download of SCDB
        skip_matching: Skip CourtListener matching (use cached)

    Returns:
        Dictionary with pipeline results
    """
    results = {
        "scdb_total": 0,
        "cases_matched": 0,
        "cases_with_text": 0,
        "train_size": 0,
        "val_size": 0,
        "test_size": 0,
    }

    # =========================================================
    # STEP 1: Download SCDB (outcome labels)
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading Supreme Court Database (SCDB)")
    logger.info("=" * 60)

    scdb_loader = SCDBLoader()
    await scdb_loader.download(force=force_download)

    scdb_cases = scdb_loader.to_scdb_cases()
    results["scdb_total"] = len(scdb_cases)
    logger.info(f"Loaded {len(scdb_cases)} SCDB cases with outcome labels")

    if max_cases:
        scdb_cases = scdb_cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases")

    # =========================================================
    # STEP 2: Match to CourtListener for full text
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Matching SCDB to CourtListener for case text")
    logger.info("=" * 60)

    cached_path = PROCESSED_DIR / "scdb_matched.parquet"

    if cached_path.exists() and skip_matching:
        logger.info(f"Loading cached matches from {cached_path}")
        from src.data.storage import load_cases_parquet
        cases = load_cases_parquet(cached_path)
        if max_cases:
            cases = cases[:max_cases]
    else:
        async with CourtListenerClient() as client:
            cases = await match_scdb_to_courtlistener(scdb_cases, client)

    results["cases_matched"] = len(cases)

    # =========================================================
    # STEP 3: Preprocess text
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Preprocessing case text")
    logger.info("=" * 60)

    for case in cases:
        if case.text:
            case.text = clean_legal_text(case.text)

    # Filter cases with sufficient text
    cases = [c for c in cases if c.text and len(c.text) > 500]
    results["cases_with_text"] = len(cases)
    logger.info(f"{len(cases)} cases with valid text (>500 chars)")

    if not cases:
        logger.error("No cases with text! Check CourtListener API key.")
        return results

    # =========================================================
    # STEP 4: Create stratified splits
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Creating train/val/test splits")
    logger.info("=" * 60)

    splits = create_stratified_splits(cases)
    results["train_size"] = len(splits.train)
    results["val_size"] = len(splits.val)
    results["test_size"] = len(splits.test)

    # =========================================================
    # STEP 5: Save outputs
    # =========================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Saving outputs")
    logger.info("=" * 60)

    # Save all matched cases
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_cases_parquet(cases, cached_path)
    logger.info(f"Saved {len(cases)} cases to {cached_path}")

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_splits(splits)
    logger.info(f"Saved splits to {SPLITS_DIR}")

    # Compute and save statistics
    stats = get_dataset_stats(cases)
    stats["pipeline_results"] = results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / "data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved statistics to {stats_path}")

    # =========================================================
    # SUMMARY
    # =========================================================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total matched: {len(cases)}")
    logger.info(f"Train: {results['train_size']}, Val: {results['val_size']}, Test: {results['test_size']}")

    petitioner = sum(1 for c in cases if c.outcome == "petitioner")
    respondent = sum(1 for c in cases if c.outcome == "respondent")
    logger.info(f"Outcomes: petitioner={petitioner}, respondent={respondent}")

    avg_len = sum(len(c.text) for c in cases) / len(cases) if cases else 0
    logger.info(f"Avg text length: {avg_len:,.0f} chars")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the LegalGPT data pipeline"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Max cases to process (default: all ~9K)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download SCDB"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip CourtListener matching (use cached)"
    )

    args = parser.parse_args()

    try:
        results = asyncio.run(run_pipeline(
            max_cases=args.max_cases,
            force_download=args.force,
            skip_matching=args.skip_matching
        ))

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for key, value in results.items():
            print(f"  {key}: {value}")

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure COURTLISTENER_API_KEY is set in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
