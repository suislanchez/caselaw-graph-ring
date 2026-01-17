"""Data pipeline module for LegalGPT.

This module provides:
- CourtListener API client for downloading SCOTUS case text (recommended)
- CAP API client (deprecated - API disabled Sept 2024)
- SCDB loader for Supreme Court outcome labels
- Text preprocessing utilities
- Storage utilities for JSON/Parquet
"""

from src.data.case_schema import (
    Case,
    CAPCase,
    SCDBCase,
    Court,
    DataSplit,
)

from src.data.courtlistener_client import (
    CourtListenerClient,
    CourtListenerCase,
    download_sample_scotus,
)

from src.data.scdb_loader import (
    SCDBLoader,
    SCDBMatcher,
    create_splits,
    load_scdb_with_cap_text,
)

from src.data.preprocessing import (
    clean_legal_text,
    sentence_segment,
    count_tokens,
    get_text_stats,
)

from src.data.storage import (
    CaseStorage,
    save_to_parquet,
    load_from_parquet,
    save_splits,
    load_splits,
    get_dataset_stats,
)

# Backward compatibility - CAPClient now points to CourtListenerClient
CAPClient = CourtListenerClient

__all__ = [
    # Schema
    "Case",
    "CAPCase",
    "SCDBCase",
    "Court",
    "DataSplit",
    # CourtListener Client (recommended)
    "CourtListenerClient",
    "CourtListenerCase",
    "download_sample_scotus",
    # Backward compat
    "CAPClient",
    # SCDB
    "SCDBLoader",
    "SCDBMatcher",
    "create_splits",
    "load_scdb_with_cap_text",
    # Preprocessing
    "clean_legal_text",
    "sentence_segment",
    "count_tokens",
    "get_text_stats",
    # Storage
    "CaseStorage",
    "save_to_parquet",
    "load_from_parquet",
    "save_splits",
    "load_splits",
    "get_dataset_stats",
]
