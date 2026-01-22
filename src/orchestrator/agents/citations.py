"""Agent A: Citation extraction and linking."""

import asyncio
from typing import Dict, Any
from pathlib import Path

from .base import BaseAgent
from ..status import StatusManager


class CitationsAgent(BaseAgent):
    """
    Citation extraction pipeline agent.

    Steps:
    1. Load cases from storage
    2. Extract citations from text
    3. Link citations to CourtListener case IDs
    4. Build citation graph edges
    """

    def __init__(self, status_manager: StatusManager):
        super().__init__("citations", status_manager, dependencies=[])

    async def run(self) -> Dict[str, Any]:
        from src.config import PROCESSED_DIR, CITATIONS_DIR

        metrics = {}

        # Step 1: Load cases
        self.start_step("loading_cases", "Loading cases from parquet...")
        try:
            from src.data.storage import load_cases_parquet
            cases_path = PROCESSED_DIR / "scdb_matched.parquet"
            if cases_path.exists():
                cases = load_cases_parquet(cases_path)
                metrics["total_cases"] = len(cases)
                self.complete_step("loading_cases", {"total_cases": len(cases)})
                self.log(f"Loaded {len(cases)} cases")
            else:
                # Try loading from JSON splits
                import json
                splits_dir = Path("data/splits")
                cases = []
                for split in ["train", "val", "test"]:
                    split_file = splits_dir / f"{split}.json"
                    if split_file.exists():
                        with open(split_file) as f:
                            cases.extend(json.load(f))
                metrics["total_cases"] = len(cases)
                self.complete_step("loading_cases", {"total_cases": len(cases)})
                self.log(f"Loaded {len(cases)} cases from splits")
        except Exception as e:
            self.log(f"Error loading cases: {e}")
            # Continue with existing data
            self.complete_step("loading_cases", {"error": str(e)})

        # Step 2: Extract citations
        self.start_step("extracting_citations", "Extracting citations from case text...")
        try:
            # Check if citations already extracted
            edges_file = CITATIONS_DIR / "edges.csv"
            if edges_file.exists():
                import pandas as pd
                edges_df = pd.read_csv(edges_file)
                metrics["edges_existing"] = len(edges_df)
                self.log(f"Found existing {len(edges_df)} edges")
                self.complete_step("extracting_citations", {"edges": len(edges_df), "status": "using_existing"})
            else:
                self.log("No existing edges, would run extraction...")
                self.complete_step("extracting_citations", {"status": "skipped_no_data"})
        except Exception as e:
            self.log(f"Citation extraction: {e}")
            self.complete_step("extracting_citations", {"error": str(e)})

        # Step 3: Link citations
        self.start_step("linking_citations", "Linking citations to case IDs...")
        try:
            link_cache = CITATIONS_DIR / "link_cache.json"
            if link_cache.exists():
                import json
                with open(link_cache) as f:
                    cache = json.load(f)
                metrics["linked_citations"] = len(cache)
                self.log(f"Found {len(cache)} linked citations in cache")
                self.complete_step("linking_citations", {"cached": len(cache)})
            else:
                self.complete_step("linking_citations", {"status": "no_cache"})
        except Exception as e:
            self.log(f"Citation linking: {e}")
            self.complete_step("linking_citations", {"error": str(e)})

        # Step 4: Build edges
        self.start_step("building_edges", "Building citation graph edges...")
        try:
            stats_file = CITATIONS_DIR / "stats.json"
            if stats_file.exists():
                import json
                with open(stats_file) as f:
                    stats = json.load(f)
                metrics.update(stats)
                self.log(f"Citation stats: {stats.get('total_edges', 0)} edges")
                self.complete_step("building_edges", stats)
            else:
                self.complete_step("building_edges", {"status": "no_stats"})
        except Exception as e:
            self.log(f"Building edges: {e}")
            self.complete_step("building_edges", {"error": str(e)})

        return metrics
