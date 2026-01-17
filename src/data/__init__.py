"""Data pipeline module for LegalGPT.

This module provides:
- CAP API client for downloading cases from Caselaw Access Project
- SCDB loader for Supreme Court outcome labels
- Text preprocessing utilities for legal documents
- Storage utilities for JSON/Parquet formats

Example usage:
    from src.data import CAPClient, SCDBLoader, clean_legal_text

    # Download cases from CAP
    async with CAPClient() as client:
        cases = await client.download_cases(max_cases=100)

    # Load SCDB and create splits
    from src.data import load_scdb_with_cap_text
    cases, splits = await load_scdb_with_cap_text(max_cases=1000)

    # Preprocess text
    cleaned = clean_legal_text(case.text)
    sentences = sentence_segment(cleaned)
    stats = get_text_stats(cleaned)

    # Save/load cases
    from src.data import save_case, load_case, save_cases_parquet
    save_case(case)
    loaded = load_case(case.id)
"""

from src.data.case_schema import (
    Case,
    CAPCase,
    SCDBCase,
    CaseMetadata,
    Court,
    Opinion,
    CaseBody,
    DataSplit,
    DatasetStats,
)

from src.data.cap_client import (
    CAPClient,
    download_sample_scotus,
)

from src.data.scdb_loader import (
    SCDBLoader,
    SCDBMatcher,
    create_splits,
    create_stratified_splits,
    match_scdb_to_cap_by_citation,
    load_scdb_with_cap_text,
)

from src.data.preprocessing import (
    clean_legal_text,
    sentence_segment,
    count_tokens,
    count_tokens_batch,
    extract_case_metadata,
    compute_text_stats,
    get_text_stats,
    truncate_to_tokens,
)

from src.data.storage import (
    CaseStorage,
    ensure_dirs,
    save_case,
    load_case,
    save_cases_parquet,
    load_cases_parquet,
    save_to_parquet,  # Backward compatibility alias
    load_from_parquet,  # Backward compatibility alias
    save_splits,
    load_splits,
    get_dataset_stats,
)

__all__ = [
    # Schema models
    "Case",
    "CAPCase",
    "SCDBCase",
    "CaseMetadata",
    "Court",
    "Opinion",
    "CaseBody",
    "DataSplit",
    "DatasetStats",
    # CAP Client
    "CAPClient",
    "download_sample_scotus",
    # SCDB Loader
    "SCDBLoader",
    "SCDBMatcher",
    "create_splits",
    "create_stratified_splits",
    "match_scdb_to_cap_by_citation",
    "load_scdb_with_cap_text",
    # Preprocessing
    "clean_legal_text",
    "sentence_segment",
    "count_tokens",
    "count_tokens_batch",
    "extract_case_metadata",
    "compute_text_stats",
    "get_text_stats",
    "truncate_to_tokens",
    # Storage
    "CaseStorage",
    "ensure_dirs",
    "save_case",
    "load_case",
    "save_cases_parquet",
    "load_cases_parquet",
    "save_to_parquet",
    "load_from_parquet",
    "save_splits",
    "load_splits",
    "get_dataset_stats",
]
