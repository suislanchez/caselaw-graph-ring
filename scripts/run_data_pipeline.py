#!/usr/bin/env python3
"""Main entry point for Agent 1: Data Pipeline.

This script orchestrates the full data pipeline:
1. Download SCDB (Supreme Court Database)
2. Match SCDB cases to CAP (Caselaw Access Project) for full text
3. Preprocess text
4. Create train/val/test splits
5. Save to storage

Usage:
    python scripts/run_data_pipeline.py --max-cases 100  # Dev sample
    python scripts/run_data_pipeline.py                  # Full dataset
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
from src.data.cap_client import CAPClient
from src.data.scdb_loader import SCDBLoader, SCDBMatcher, create_splits
from src.data.preprocessing import clean_legal_text, compute_text_stats
from src.data.storage import save_to_parquet, save_splits, get_dataset_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_pipeline(
    max_cases: int | None = None,
    force_download: bool = False,
    skip_cap_match: bool = False
) -> dict:
    """Run the full data pipeline.

    Args:
        max_cases: Maximum cases to process (None for all)
        force_download: Force re-download of SCDB
        skip_cap_match: Skip CAP matching (use cached if available)

    Returns:
        Dictionary with pipeline results and statistics
    """
    results = {
        "scdb_downloaded": 0,
        "cases_matched": 0,
        "cases_preprocessed": 0,
        "train_size": 0,
        "val_size": 0,
        "test_size": 0,
    }

    # Step 1: Download SCDB
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading Supreme Court Database (SCDB)")
    logger.info("=" * 60)

    scdb_loader = SCDBLoader()
    await scdb_loader.download(force=force_download)

    scdb_cases = scdb_loader.to_scdb_cases()
    results["scdb_downloaded"] = len(scdb_cases)
    logger.info(f"Loaded {len(scdb_cases)} SCDB cases with outcome labels")

    if max_cases:
        scdb_cases = scdb_cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases for processing")

    # Step 2: Match to CAP for full text
    logger.info("=" * 60)
    logger.info("STEP 2: Matching SCDB cases to CAP for full text")
    logger.info("=" * 60)

    # Check for cached matches
    cached_path = PROCESSED_DIR / "scdb_matched.parquet"
    if cached_path.exists() and not force_download and not skip_cap_match:
        logger.info(f"Found cached matches at {cached_path}")
        from src.data.storage import load_from_parquet
        cases = load_from_parquet(cached_path)
        if max_cases:
            cases = cases[:max_cases]
    else:
        matcher = SCDBMatcher()
        cases = await matcher.match_cases(scdb_cases, max_concurrent=3)

    results["cases_matched"] = len(cases)
    logger.info(f"Matched {len(cases)} cases to CAP")

    # Step 3: Preprocess text
    logger.info("=" * 60)
    logger.info("STEP 3: Preprocessing case text")
    logger.info("=" * 60)

    for case in cases:
        if case.text:
            case.text = clean_legal_text(case.text)

    # Filter out cases with empty text
    cases = [c for c in cases if c.text and len(c.text) > 100]
    results["cases_preprocessed"] = len(cases)
    logger.info(f"Preprocessed {len(cases)} cases")

    # Step 4: Create splits
    logger.info("=" * 60)
    logger.info("STEP 4: Creating train/val/test splits")
    logger.info("=" * 60)

    splits = create_splits(cases)
    results["train_size"] = len(splits.train)
    results["val_size"] = len(splits.val)
    results["test_size"] = len(splits.test)

    # Step 5: Save outputs
    logger.info("=" * 60)
    logger.info("STEP 5: Saving outputs")
    logger.info("=" * 60)

    # Save matched cases to parquet
    save_to_parquet(cases, PROCESSED_DIR / "scdb_matched.parquet")

    # Save splits
    save_splits(splits)

    # Compute and save statistics
    stats = get_dataset_stats(cases)
    text_stats = compute_text_stats([c.text for c in cases if c.text])
    stats.update({"text_stats": text_stats})

    stats_path = RESULTS_DIR / "data_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved statistics to {stats_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total cases processed: {len(cases)}")
    logger.info(f"Train: {results['train_size']}, Val: {results['val_size']}, Test: {results['test_size']}")
    logger.info(f"Petitioner wins: {stats.get('petitioner_wins', 0)}")
    logger.info(f"Respondent wins: {stats.get('respondent_wins', 0)}")
    logger.info(f"Avg text length: {stats.get('avg_text_length', 0):.0f} chars")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the data pipeline for LegalGPT"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum cases to process (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all data"
    )
    parser.add_argument(
        "--skip-cap",
        action="store_true",
        help="Skip CAP matching (use cached data)"
    )

    args = parser.parse_args()

    results = asyncio.run(run_pipeline(
        max_cases=args.max_cases,
        force_download=args.force,
        skip_cap_match=args.skip_cap
    ))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
