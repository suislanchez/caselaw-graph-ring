"""Storage utilities for case data.

Provides functions for:
- JSON storage for individual cases
- Parquet storage for bulk case data
- Directory management
- Train/val/test split persistence
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Iterator, TypeVar, Union
from datetime import datetime

import pandas as pd

from src.config import (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CASES_DIR,
    DATA_DIR, EMBEDDINGS_DIR, CITATIONS_DATA_DIR
)
from src.data.case_schema import Case, DataSplit

logger = logging.getLogger(__name__)

T = TypeVar('T')


def ensure_dirs() -> None:
    """Create all necessary data directories.

    Creates the following directory structure:
    - data/raw/cases/
    - data/processed/
    - data/splits/
    - data/embeddings/
    - data/citations/
    """
    directories = [
        DATA_DIR,
        RAW_DIR,
        CASES_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        EMBEDDINGS_DIR,
        CITATIONS_DATA_DIR,
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

    logger.info(f"All data directories created/verified")


def save_case(case: Case, output_dir: Optional[Path] = None) -> Path:
    """Save a single case to JSON.

    Args:
        case: Case to save
        output_dir: Output directory (defaults to CASES_DIR)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or CASES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{case.id}.json"
    with open(file_path, 'w') as f:
        json.dump(case.model_dump(mode="json"), f, indent=2, default=str)

    return file_path


def load_case(case_id: str, input_dir: Optional[Path] = None) -> Optional[Case]:
    """Load a single case from JSON.

    Args:
        case_id: Case ID to load
        input_dir: Input directory (defaults to CASES_DIR)

    Returns:
        Case object or None if not found
    """
    input_dir = input_dir or CASES_DIR
    file_path = input_dir / f"{case_id}.json"

    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle datetime parsing
    if 'date' in data and isinstance(data['date'], str):
        try:
            data['date'] = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
        except ValueError:
            data['date'] = datetime.now()

    return Case(**data)


def save_cases_parquet(cases: List[Case], output_path: Union[str, Path]) -> Path:
    """Save cases to Parquet format for bulk storage.

    Args:
        cases: List of cases to save
        output_path: Output file path

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)

    # Convert to DataFrame
    records = []
    for case in cases:
        record = case.model_dump()
        # Flatten nested court object for parquet compatibility
        if 'court' in record and isinstance(record['court'], dict):
            record['court_slug'] = record['court'].get('slug')
            record['court_name'] = record['court'].get('name')
            record['court_name_abbreviation'] = record['court'].get('name_abbreviation')
            del record['court']
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path)
    logger.info(f"Saved {len(cases)} cases to {output_path}")

    return output_path


def load_cases_parquet(input_path: Union[str, Path]) -> List[Case]:
    """Load cases from Parquet format.

    Args:
        input_path: Input file path

    Returns:
        List of Case objects
    """
    input_path = Path(input_path)
    df = pd.read_parquet(input_path)

    cases = []
    for _, row in df.iterrows():
        data = row.to_dict()

        # Reconstruct court object if flattened
        if 'court_slug' in data:
            data['court'] = {
                'slug': data.pop('court_slug'),
                'name': data.pop('court_name', 'Unknown'),
                'name_abbreviation': data.pop('court_name_abbreviation', None)
            }

        # Handle NaN values
        for key, value in list(data.items()):
            if pd.isna(value):
                data[key] = None

        # Handle datetime
        if 'date' in data and data['date'] is not None:
            if isinstance(data['date'], str):
                try:
                    data['date'] = datetime.fromisoformat(data['date'].replace('Z', '+00:00'))
                except ValueError:
                    data['date'] = datetime.now()

        try:
            case = Case(**data)
            cases.append(case)
        except Exception as e:
            logger.warning(f"Failed to parse case from parquet: {e}")

    logger.info(f"Loaded {len(cases)} cases from {input_path}")
    return cases


class CaseStorage:
    """Manage case data storage and retrieval.

    Provides a class-based interface for JSON case storage with
    iteration, counting, and batch operations.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize storage.

        Args:
            base_dir: Base directory for data (defaults to RAW_DIR)
        """
        self.base_dir = base_dir or RAW_DIR
        self.cases_dir = self.base_dir / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)

    def save_case(self, case: Case) -> Path:
        """Save a single case to JSON.

        Args:
            case: Case to save

        Returns:
            Path to saved file
        """
        return save_case(case, self.cases_dir)

    def load_case(self, case_id: str) -> Optional[Case]:
        """Load a single case from JSON.

        Args:
            case_id: Case ID to load

        Returns:
            Case object or None if not found
        """
        return load_case(case_id, self.cases_dir)

    def save_cases(self, cases: List[Case], batch_name: Optional[str] = None) -> int:
        """Save multiple cases.

        Args:
            cases: List of cases to save
            batch_name: Optional batch identifier (for logging)

        Returns:
            Number of cases saved
        """
        count = 0
        for case in cases:
            try:
                self.save_case(case)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save case {case.id}: {e}")

        logger.info(f"Saved {count} cases" + (f" (batch: {batch_name})" if batch_name else ""))
        return count

    def load_all_cases(self) -> List[Case]:
        """Load all cases from storage.

        Returns:
            List of all cases
        """
        cases = []
        for file_path in self.cases_dir.glob("*.json"):
            case_id = file_path.stem
            case = self.load_case(case_id)
            if case:
                cases.append(case)

        logger.info(f"Loaded {len(cases)} cases")
        return cases

    def iter_cases(self) -> Iterator[Case]:
        """Iterate over cases without loading all into memory.

        Yields:
            Case objects
        """
        for file_path in self.cases_dir.glob("*.json"):
            case_id = file_path.stem
            case = self.load_case(case_id)
            if case:
                yield case

    def case_exists(self, case_id: str) -> bool:
        """Check if a case exists in storage.

        Args:
            case_id: Case ID to check

        Returns:
            True if exists
        """
        return (self.cases_dir / f"{case_id}.json").exists()

    def count_cases(self) -> int:
        """Count total cases in storage.

        Returns:
            Number of cases
        """
        return len(list(self.cases_dir.glob("*.json")))

    def delete_case(self, case_id: str) -> bool:
        """Delete a case from storage.

        Args:
            case_id: Case ID to delete

        Returns:
            True if deleted
        """
        file_path = self.cases_dir / f"{case_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


# Backward compatibility aliases
save_to_parquet = save_cases_parquet
load_from_parquet = load_cases_parquet


def save_splits(splits: DataSplit, output_dir: Optional[Path] = None) -> None:
    """Save train/val/test splits to JSON.

    Args:
        splits: DataSplit object
        output_dir: Output directory (defaults to SPLITS_DIR)
    """
    output_dir = output_dir or SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_cases = getattr(splits, split_name)
        output_path = output_dir / f"{split_name}.json"

        with open(output_path, 'w') as f:
            json.dump(
                [c.model_dump(mode="json") for c in split_cases],
                f,
                indent=2,
                default=str
            )

        logger.info(f"Saved {len(split_cases)} cases to {output_path}")


def load_splits(input_dir: Optional[Path] = None) -> DataSplit:
    """Load train/val/test splits from JSON.

    Args:
        input_dir: Input directory (defaults to SPLITS_DIR)

    Returns:
        DataSplit object
    """
    input_dir = input_dir or SPLITS_DIR

    splits = {}
    for split_name in ["train", "val", "test"]:
        input_path = input_dir / f"{split_name}.json"

        if not input_path.exists():
            logger.warning(f"Split file not found: {input_path}")
            splits[split_name] = []
            continue

        with open(input_path, 'r') as f:
            data = json.load(f)

        cases = []
        for record in data:
            # Handle datetime
            if 'date' in record and isinstance(record['date'], str):
                try:
                    record['date'] = datetime.fromisoformat(
                        record['date'].replace('Z', '+00:00')
                    )
                except ValueError:
                    record['date'] = datetime.now()

            try:
                case = Case(**record)
                cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to parse case: {e}")

        splits[split_name] = cases
        logger.info(f"Loaded {len(cases)} cases from {input_path}")

    return DataSplit(**splits)


def get_dataset_stats(cases: List[Case]) -> dict:
    """Compute statistics for a list of cases.

    Args:
        cases: List of cases

    Returns:
        Dictionary of statistics
    """
    if not cases:
        return {"error": "No cases provided"}

    outcomes = [c.outcome for c in cases if c.outcome]
    text_lengths = [len(c.text) for c in cases if c.text]
    citation_counts = [len(c.citations_raw) for c in cases]

    courts = {}
    for c in cases:
        court_name = c.court.name if c.court else "Unknown"
        courts[court_name] = courts.get(court_name, 0) + 1

    dates = [c.date for c in cases if c.date]

    return {
        "total_cases": len(cases),
        "cases_with_outcome": len(outcomes),
        "petitioner_wins": outcomes.count("petitioner"),
        "respondent_wins": outcomes.count("respondent"),
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "min_text_length": min(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
        "avg_citations": sum(citation_counts) / len(citation_counts) if citation_counts else 0,
        "courts": courts,
        "date_range": (min(dates).isoformat(), max(dates).isoformat()) if dates else None,
    }


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data storage utilities")
    parser.add_argument("--stats", action="store_true", help="Print dataset stats")
    parser.add_argument("--count", action="store_true", help="Count cases in storage")
    parser.add_argument("--ensure-dirs", action="store_true", help="Create all data directories")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.ensure_dirs:
        ensure_dirs()
        print("All directories created/verified")

    storage = CaseStorage()

    if args.count:
        count = storage.count_cases()
        print(f"Cases in storage: {count}")

    if args.stats:
        cases = storage.load_all_cases()
        stats = get_dataset_stats(cases)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
