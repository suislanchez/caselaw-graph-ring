"""Link extracted citations to case IDs.

Uses the CourtListener API to search for cases by citation and caches successful
lookups for performance.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from pydantic import BaseModel, Field

from src.config import CITATIONS_DIR
from src.data.courtlistener_client import CourtListenerClient

logger = logging.getLogger(__name__)


class LinkResult(BaseModel):
    """Result of attempting to link a citation to a case ID."""

    citation: str = Field(description="The normalized citation")
    case_id: Optional[str] = Field(None, description="Linked CAP case ID if found")
    case_name: Optional[str] = Field(None, description="Case name if found")
    confidence: float = Field(1.0, description="Confidence score (1.0 = exact match)")
    ambiguous: bool = Field(False, description="True if multiple matches found")
    match_count: int = Field(0, description="Number of potential matches")
    error: Optional[str] = Field(None, description="Error message if linking failed")


class LinkingStats(BaseModel):
    """Statistics from citation linking."""

    total_citations: int = 0
    successful_links: int = 0
    ambiguous_links: int = 0
    failed_links: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    errors: int = 0
    success_rate: float = 0.0
    error_messages: List[str] = Field(default_factory=list)

    def update_success_rate(self) -> None:
        """Calculate success rate."""
        if self.total_citations > 0:
            self.success_rate = self.successful_links / self.total_citations


class CitationLinker:
    """Links extracted citations to case IDs.

    Uses the CourtListener API to search for cases by citation. Caches successful
    lookups to avoid redundant API calls.

    Usage:
        linker = CitationLinker(client)
        case_id = await linker.link_citation("347 U.S. 483")

        # Batch linking
        results = await linker.link_batch(citations)
    """

    def __init__(
        self,
        client: Optional[CourtListenerClient] = None,
        cache_path: Optional[Path] = None,
        max_concurrent: int = 5,
        cache_ambiguous: bool = False,
    ):
        """Initialize the citation linker.

        Args:
            client: CourtListenerClient instance for API calls
            cache_path: Path to cache file (defaults to CITATIONS_DIR/link_cache.json)
            max_concurrent: Max concurrent API requests
            cache_ambiguous: Whether to cache ambiguous results
        """
        self.client = client
        self._owns_client = client is None  # Track if we created the client
        self.cache_path = cache_path or CITATIONS_DIR / "link_cache.json"
        self.max_concurrent = max_concurrent
        self.cache_ambiguous = cache_ambiguous

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = LinkingStats()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the link cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached citation links")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save the link cache to disk."""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_from_cache(self, citation: str) -> Optional[LinkResult]:
        """Get a link result from cache.

        Args:
            citation: Normalized citation string

        Returns:
            LinkResult if cached, None otherwise
        """
        if citation in self._cache:
            cached = self._cache[citation]
            self._stats.cache_hits += 1
            return LinkResult(**cached)
        return None

    def _add_to_cache(self, result: LinkResult) -> None:
        """Add a link result to cache.

        Args:
            result: LinkResult to cache
        """
        # Only cache successful or definitively failed results
        if result.case_id or (result.ambiguous and self.cache_ambiguous) or result.match_count == 0:
            self._cache[result.citation] = result.model_dump()

    async def link_citation(self, citation: str) -> Optional[str]:
        """Link a single citation to a CAP case ID.

        Args:
            citation: Normalized citation string (e.g., "347 U.S. 483")

        Returns:
            CAP case ID if found, None otherwise
        """
        result = await self.link_citation_full(citation)
        return result.case_id

    async def _ensure_client(self) -> CourtListenerClient:
        """Ensure we have a valid client, creating one if needed."""
        if self.client is None:
            self.client = CourtListenerClient()
            self._owns_client = True
        return self.client

    async def link_citation_full(self, citation: str) -> LinkResult:
        """Link a citation and return full result with metadata.

        Args:
            citation: Normalized citation string

        Returns:
            LinkResult with all linking information
        """
        # Check cache first
        cached = self._get_from_cache(citation)
        if cached:
            return cached

        result = LinkResult(citation=citation)

        try:
            async with self._semaphore:
                self._stats.api_calls += 1

                # Get or create client
                client = await self._ensure_client()

                # Search for the case by citation using CourtListener
                cl_case = await client.search_by_citation(citation, save_to_disk=False)

                if cl_case:
                    # Found a match
                    result.case_id = f"cl_{cl_case.cluster_id}"
                    result.case_name = cl_case.case_name
                    result.confidence = 1.0
                    result.match_count = 1
                    self._stats.successful_links += 1
                else:
                    # No matches found
                    result.error = "No matches found"
                    result.match_count = 0
                    self._stats.failed_links += 1

        except Exception as e:
            logger.warning(f"Error linking citation '{citation}': {e}")
            result.error = str(e)
            self._stats.errors += 1
            self._stats.error_messages.append(f"{citation}: {str(e)}")

        # Cache the result
        self._add_to_cache(result)

        return result

    async def link_batch(
        self,
        citations: List[str],
        save_cache_interval: int = 50,
        show_progress: bool = True,
    ) -> Dict[str, Optional[str]]:
        """Link multiple citations to CAP case IDs.

        Args:
            citations: List of normalized citation strings
            save_cache_interval: Save cache every N citations
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping citation to case ID (or None if not found)
        """
        results: Dict[str, Optional[str]] = {}

        # Reset stats
        self._stats = LinkingStats()
        self._stats.total_citations = len(citations)

        if not citations:
            return results

        # Deduplicate citations
        unique_citations = list(set(citations))

        # Create tasks for all citations
        tasks = []
        for citation in unique_citations:
            tasks.append(self.link_citation_full(citation))

        # Process with optional progress bar
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio
                completed = await tqdm_asyncio.gather(
                    *tasks,
                    desc="Linking citations",
                )
            except ImportError:
                completed = await asyncio.gather(*tasks)
        else:
            completed = await asyncio.gather(*tasks)

        # Build results dictionary
        for link_result in completed:
            results[link_result.citation] = link_result.case_id

            # Periodic cache save
            if len(results) % save_cache_interval == 0:
                self._save_cache()

        # Final cache save
        self._save_cache()

        # Update success rate
        self._stats.update_success_rate()

        return results

    async def link_batch_full(
        self,
        citations: List[str],
        show_progress: bool = True,
    ) -> List[LinkResult]:
        """Link multiple citations and return full results.

        Args:
            citations: List of normalized citation strings
            show_progress: Whether to show progress bar

        Returns:
            List of LinkResult objects
        """
        # Reset stats
        self._stats = LinkingStats()
        self._stats.total_citations = len(citations)

        if not citations:
            return []

        # Deduplicate citations
        unique_citations = list(set(citations))

        # Create tasks for all citations
        tasks = [self.link_citation_full(citation) for citation in unique_citations]

        # Process with optional progress bar
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio
                results = await tqdm_asyncio.gather(
                    *tasks,
                    desc="Linking citations",
                )
            except ImportError:
                results = await asyncio.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)

        # Final cache save
        self._save_cache()

        # Update success rate
        self._stats.update_success_rate()

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get linking statistics.

        Returns:
            Dictionary with linking statistics
        """
        self._stats.update_success_rate()
        return self._stats.model_dump()

    def get_unlinked_citations(self) -> List[str]:
        """Get list of citations that couldn't be linked.

        Returns:
            List of citation strings that failed to link
        """
        unlinked = []
        for citation, data in self._cache.items():
            if data.get("case_id") is None and not data.get("ambiguous"):
                unlinked.append(citation)
        return unlinked

    def save_unlinked(self, path: Optional[Path] = None) -> None:
        """Save unlinked citations to a file.

        Args:
            path: Output path (defaults to CITATIONS_DIR/unlinked.json)
        """
        path = path or CITATIONS_DIR / "unlinked.json"
        unlinked = self.get_unlinked_citations()

        with open(path, "w") as f:
            json.dump({
                "count": len(unlinked),
                "citations": unlinked,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        logger.info(f"Saved {len(unlinked)} unlinked citations to {path}")

    def clear_cache(self) -> None:
        """Clear the link cache."""
        self._cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Cleared citation link cache")


async def link_citations(citations: List[str]) -> Dict[str, Optional[str]]:
    """Convenience function to link citations.

    Args:
        citations: List of citation strings

    Returns:
        Dictionary mapping citation to case ID
    """
    async with CourtListenerClient() as client:
        linker = CitationLinker(client=client)
        return await linker.link_batch(citations)


if __name__ == "__main__":
    # Test the linker
    async def test():
        async with CourtListenerClient() as client:
            linker = CitationLinker(client=client)

            test_citations = [
                "347 U.S. 483",  # Brown v. Board of Education
                "163 U.S. 537",  # Plessy v. Ferguson
                "410 U.S. 113",  # Roe v. Wade
            ]

            print("Linking test citations...")
            results = await linker.link_batch(test_citations)

            print("\nResults:")
            for citation, case_id in results.items():
                print(f"  {citation} -> {case_id}")

            print(f"\nStats: {linker.get_stats()}")

    asyncio.run(test())
