"""Supreme Court Database (SCDB) loader and matcher.

Downloads SCDB data, parses case outcomes, and matches to CAP cases by citation.
Creates stratified train/val/test splits based on outcome distribution.
"""

import asyncio
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter

import aiohttp
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import (
    SCDB_URL, PROCESSED_DIR, SPLITS_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from src.data.case_schema import Case, SCDBCase, DataSplit
from src.data.cap_client import CAPClient

logger = logging.getLogger(__name__)


class SCDBLoader:
    """Load and process Supreme Court Database.

    Downloads SCDB CSV from Washington University, parses case outcomes,
    and provides methods to extract cases with valid outcome labels.
    """

    def __init__(self, scdb_url: str = SCDB_URL):
        """Initialize SCDB loader.

        Args:
            scdb_url: URL to SCDB CSV zip file
        """
        self.scdb_url = scdb_url
        self._df: Optional[pd.DataFrame] = None

    async def download(self, force: bool = False) -> pd.DataFrame:
        """Download SCDB data from Washington University.

        Args:
            force: Force re-download even if cached

        Returns:
            DataFrame with SCDB data
        """
        cache_file = PROCESSED_DIR / "scdb_raw.parquet"

        if cache_file.exists() and not force:
            logger.info(f"Loading cached SCDB from {cache_file}")
            self._df = pd.read_parquet(cache_file)
            return self._df

        logger.info(f"Downloading SCDB from {self.scdb_url}")

        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.get(self.scdb_url) as response:
                response.raise_for_status()
                content = await response.read()

                # Extract CSV from zip
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        raise ValueError("No CSV file found in SCDB zip")

                    with z.open(csv_files[0]) as f:
                        self._df = pd.read_csv(f, encoding='latin-1', low_memory=False)

        logger.info(f"Loaded {len(self._df)} SCDB records")

        # Cache the raw data
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(cache_file)

        return self._df

    def get_cases_with_outcomes(self) -> pd.DataFrame:
        """Get cases that have outcome labels.

        partyWinning values:
        - 1 = petitioner/appellant won
        - 0 = respondent/appellee won

        Returns:
            DataFrame filtered to cases with valid outcomes
        """
        if self._df is None:
            raise ValueError("Must call download() first")

        # Filter to cases with valid partyWinning values
        df = self._df[self._df['partyWinning'].isin([0, 1])].copy()

        # Select relevant columns
        columns = [
            'caseId', 'docket', 'caseName', 'dateDecision',
            'usCite', 'sctCite', 'ledCite', 'lexisCite',
            'partyWinning', 'decisionDirection', 'issueArea'
        ]

        # Only keep columns that exist
        columns = [c for c in columns if c in df.columns]
        df = df[columns].copy()

        # Log outcome distribution
        petitioner_wins = (df['partyWinning'] == 1).sum()
        respondent_wins = (df['partyWinning'] == 0).sum()
        logger.info(f"Found {len(df)} cases with outcomes: {petitioner_wins} petitioner, {respondent_wins} respondent")

        return df

    def to_scdb_cases(self, df: Optional[pd.DataFrame] = None) -> List[SCDBCase]:
        """Convert DataFrame to SCDBCase objects.

        Args:
            df: DataFrame to convert (defaults to cases with outcomes)

        Returns:
            List of SCDBCase objects
        """
        if df is None:
            df = self.get_cases_with_outcomes()

        cases = []
        for _, row in df.iterrows():
            try:
                case = SCDBCase(
                    caseId=str(row.get('caseId', '')),
                    docket=row.get('docket'),
                    caseName=row.get('caseName', 'Unknown'),
                    dateDecision=row.get('dateDecision'),
                    usCite=row.get('usCite'),
                    sctCite=row.get('sctCite'),
                    ledCite=row.get('ledCite'),
                    lexisCite=row.get('lexisCite'),
                    partyWinning=row.get('partyWinning'),
                    decisionDirection=row.get('decisionDirection'),
                    issueArea=row.get('issueArea')
                )
                cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to parse SCDB case: {e}")
                continue

        return cases


class SCDBMatcher:
    """Match SCDB cases to CAP cases by citation."""

    def __init__(self, cap_client: Optional[CAPClient] = None):
        """Initialize matcher.

        Args:
            cap_client: CAP API client
        """
        self.cap_client = cap_client or CAPClient()
        self._cache: Dict[str, Optional[Case]] = {}

    async def match_by_citation(
        self,
        scdb_case: SCDBCase
    ) -> Optional[Case]:
        """Match SCDB case to CAP case by citation.

        Tries US cite first, then S.Ct. cite, then L.Ed. cite.

        Args:
            scdb_case: SCDB case to match

        Returns:
            Matched Case or None
        """
        citation = scdb_case.primary_citation
        if not citation:
            return None

        # Check cache
        if citation in self._cache:
            return self._cache[citation]

        # Search CAP
        cap_case = await self.cap_client.get_case_by_citation(citation, save_to_disk=True)

        if cap_case:
            case = cap_case.to_case()
            case.outcome = scdb_case.outcome
            case.scdb_id = scdb_case.case_id
            self._cache[citation] = case
            return case

        self._cache[citation] = None
        return None

    async def match_cases(
        self,
        scdb_cases: List[SCDBCase],
        max_concurrent: int = 5
    ) -> List[Case]:
        """Match multiple SCDB cases to CAP cases.

        Args:
            scdb_cases: List of SCDB cases
            max_concurrent: Max concurrent API calls

        Returns:
            List of matched Case objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        matched_cases = []

        async def match_with_semaphore(scdb_case: SCDBCase) -> Optional[Case]:
            async with semaphore:
                return await self.match_by_citation(scdb_case)

        # Create tasks
        tasks = [match_with_semaphore(sc) for sc in scdb_cases]

        # Run with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Matching cases"):
            result = await coro
            results.append(result)

        matched_cases = [c for c in results if c is not None]
        logger.info(f"Matched {len(matched_cases)}/{len(scdb_cases)} cases")

        return matched_cases


def create_stratified_splits(
    cases: List[Case],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42
) -> DataSplit:
    """Create train/val/test splits stratified by outcome.

    Ensures that each split has approximately the same ratio of
    petitioner wins to respondent wins.

    Args:
        cases: List of cases to split
        train_ratio: Training set ratio (default: 0.70)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        DataSplit object with train, val, test lists
    """
    if not cases:
        logger.warning("No cases to split")
        return DataSplit(train=[], val=[], test=[])

    # Separate cases by outcome
    cases_with_outcome = [c for c in cases if c.outcome is not None]
    cases_without_outcome = [c for c in cases if c.outcome is None]

    if cases_without_outcome:
        logger.warning(f"{len(cases_without_outcome)} cases have no outcome label")

    if not cases_with_outcome:
        logger.warning("No cases with outcome labels for stratified splitting")
        # Fall back to random split
        return create_splits(cases, train_ratio, val_ratio, test_ratio, seed)

    # Extract labels for stratification
    labels = [c.outcome for c in cases_with_outcome]

    # First split: separate test set
    # val_ratio + train_ratio should equal 1 - test_ratio
    train_val_cases, test_cases, train_val_labels, _ = train_test_split(
        cases_with_outcome,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels
    )

    # Second split: separate train and val from remaining
    # val_ratio / (train_ratio + val_ratio) gives the proportion of val in train_val
    val_proportion = val_ratio / (train_ratio + val_ratio)

    train_cases, val_cases, _, _ = train_test_split(
        train_val_cases,
        train_val_labels,
        test_size=val_proportion,
        random_state=seed,
        stratify=train_val_labels
    )

    # Add cases without outcome to training set
    train_cases.extend(cases_without_outcome)

    # Log split statistics
    def get_outcome_stats(case_list):
        outcomes = Counter(c.outcome for c in case_list if c.outcome)
        return dict(outcomes)

    logger.info(f"Split statistics:")
    logger.info(f"  Train: {len(train_cases)} cases, outcomes: {get_outcome_stats(train_cases)}")
    logger.info(f"  Val: {len(val_cases)} cases, outcomes: {get_outcome_stats(val_cases)}")
    logger.info(f"  Test: {len(test_cases)} cases, outcomes: {get_outcome_stats(test_cases)}")

    return DataSplit(train=train_cases, val=val_cases, test=test_cases)


def create_splits(
    cases: List[Case],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42
) -> DataSplit:
    """Create train/val/test splits (non-stratified).

    Use create_stratified_splits() for stratified splitting by outcome.

    Args:
        cases: List of cases to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        DataSplit object
    """
    import random
    random.seed(seed)

    # Shuffle cases
    shuffled = cases.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    return DataSplit(train=train, val=val, test=test)


def match_scdb_to_cap_by_citation(
    scdb_df: pd.DataFrame,
    cap_cases: List[Case]
) -> List[Case]:
    """Match SCDB cases to CAP cases by citation (offline matching).

    Useful when you already have CAP cases downloaded and want to match
    them to SCDB outcome labels without making API calls.

    Args:
        scdb_df: DataFrame with SCDB data (must have usCite, sctCite columns)
        cap_cases: List of CAP Case objects

    Returns:
        List of Case objects with SCDB outcome labels added
    """
    # Build citation index from SCDB
    citation_to_scdb = {}
    for _, row in scdb_df.iterrows():
        for cite_col in ['usCite', 'sctCite', 'ledCite']:
            cite = row.get(cite_col)
            if pd.notna(cite) and cite:
                # Normalize citation format
                cite_normalized = str(cite).strip()
                citation_to_scdb[cite_normalized] = row

    # Match CAP cases to SCDB
    matched_cases = []
    for case in cap_cases:
        scdb_row = None

        # Try each citation
        for cite in case.citations:
            cite_normalized = str(cite).strip()
            if cite_normalized in citation_to_scdb:
                scdb_row = citation_to_scdb[cite_normalized]
                break

        if scdb_row is not None:
            # Add outcome from SCDB
            party_winning = scdb_row.get('partyWinning')
            if party_winning == 1:
                case.outcome = "petitioner"
            elif party_winning == 0:
                case.outcome = "respondent"

            case.scdb_id = str(scdb_row.get('caseId', ''))
            matched_cases.append(case)

    logger.info(f"Matched {len(matched_cases)}/{len(cap_cases)} cases to SCDB")
    return matched_cases


async def load_scdb_with_cap_text(
    max_cases: Optional[int] = None,
    force_download: bool = False,
    use_stratified: bool = True
) -> Tuple[List[Case], DataSplit]:
    """Full pipeline: load SCDB, match to CAP, create splits.

    Downloads SCDB data, matches cases to CAP API by citation,
    and creates train/val/test splits (stratified by outcome).

    Args:
        max_cases: Maximum cases to process (None for all)
        force_download: Force re-download of SCDB
        use_stratified: Use stratified splits (recommended)

    Returns:
        Tuple of (all_cases, splits)
    """
    # Load SCDB
    loader = SCDBLoader()
    await loader.download(force=force_download)
    scdb_cases = loader.to_scdb_cases()

    if max_cases:
        scdb_cases = scdb_cases[:max_cases]

    # Match to CAP
    async with CAPClient() as cap_client:
        matcher = SCDBMatcher(cap_client)
        cases = await matcher.match_cases(scdb_cases)

    # Save matched cases to parquet
    if cases:
        matched_records = []
        for case in cases:
            record = case.model_dump()
            # Flatten court object for parquet
            if 'court' in record and isinstance(record['court'], dict):
                record['court_slug'] = record['court'].get('slug')
                record['court_name'] = record['court'].get('name')
                del record['court']
            matched_records.append(record)

        matched_df = pd.DataFrame(matched_records)
        output_path = PROCESSED_DIR / "scdb_matched.parquet"
        matched_df.to_parquet(output_path)
        logger.info(f"Saved {len(cases)} matched cases to {output_path}")

    # Create splits
    if use_stratified:
        splits = create_stratified_splits(cases)
    else:
        splits = create_splits(cases)

    # Save splits to JSON
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_cases = getattr(splits, split_name)
        split_file = SPLITS_DIR / f"{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(
                [c.model_dump(mode="json") for c in split_cases],
                f,
                indent=2,
                default=str
            )
        logger.info(f"Saved {len(split_cases)} cases to {split_file}")

    return cases, splits


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load SCDB and match to CAP")
    parser.add_argument("--max-cases", type=int, help="Max cases to process")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified splits")
    parser.add_argument("--download-only", action="store_true", help="Only download SCDB, no matching")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    async def main():
        if args.download_only:
            loader = SCDBLoader()
            df = await loader.download(force=args.force)
            print(f"\nDownloaded {len(df)} SCDB records")

            cases_df = loader.get_cases_with_outcomes()
            print(f"Cases with outcomes: {len(cases_df)}")
            print(f"Petitioner wins: {(cases_df['partyWinning'] == 1).sum()}")
            print(f"Respondent wins: {(cases_df['partyWinning'] == 0).sum()}")
        else:
            cases, splits = await load_scdb_with_cap_text(
                max_cases=args.max_cases,
                force_download=args.force,
                use_stratified=not args.no_stratify
            )

            print(f"\nProcessed {len(cases)} cases")
            print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")

            # Print outcome distribution
            petitioner = sum(1 for c in cases if c.outcome == "petitioner")
            respondent = sum(1 for c in cases if c.outcome == "respondent")
            print(f"Outcomes: petitioner={petitioner}, respondent={respondent}")

    asyncio.run(main())
