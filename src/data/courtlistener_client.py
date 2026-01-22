"""Async client for CourtListener API.

CourtListener provides free access to US case law including Supreme Court opinions.
API Documentation: https://www.courtlistener.com/help/api/rest/

Data Model:
- courts: Court metadata (name, abbreviation)
- dockets: Case docket info (docket number, dates)
- clusters: Groups of opinions for a case
- opinions: Individual opinion text (majority, dissent, concurrence)

Rate Limit: 5,000 requests/hour for authenticated users
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

from src.config import CASES_DIR
from src.data.case_schema import Case, Court

logger = logging.getLogger(__name__)

# CourtListener API configuration
COURTLISTENER_API_BASE = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_API_KEY = os.environ.get("COURTLISTENER_API_KEY")


def extract_id_from_url_or_value(value, resource_type: str = "clusters") -> Optional[int]:
    """Extract numeric ID from a URL, dict, or int value.

    Args:
        value: Can be int, dict with 'id' key, or URL string like
               "https://.../clusters/12345/"
        resource_type: The resource type to look for in URL (e.g., "clusters")

    Returns:
        Integer ID or None if extraction fails
    """
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, dict):
        return value.get("id")

    if isinstance(value, str):
        # Extract ID from URL like "https://.../clusters/12345/"
        if f"/{resource_type}/" in value:
            try:
                # Get the part after /clusters/ and before the next /
                parts = value.rstrip("/").split(f"/{resource_type}/")
                if len(parts) >= 2:
                    id_str = parts[-1].split("/")[0]
                    return int(id_str)
            except (ValueError, IndexError):
                pass
        # Try direct conversion
        try:
            return int(value)
        except ValueError:
            pass

    return None


class CourtListenerCase:
    """Represents a case from CourtListener API."""

    def __init__(
        self,
        cluster_id: int,
        case_name: str,
        date_filed: str,
        court_id: str,
        court_name: str,
        opinions: List[Dict[str, Any]],
        docket_number: Optional[str] = None,
        citations: Optional[List[str]] = None,
        **kwargs
    ):
        self.cluster_id = cluster_id
        self.case_name = case_name
        self.date_filed = date_filed
        self.court_id = court_id
        self.court_name = court_name
        self.opinions = opinions
        self.docket_number = docket_number
        self.citations = citations or []
        self.extra = kwargs

    @property
    def full_text(self) -> str:
        """Get concatenated text from all opinions."""
        texts = []
        for op in self.opinions:
            # Try different text fields
            text = (
                op.get("plain_text") or
                op.get("html_with_citations") or
                op.get("html") or
                op.get("xml_harvard") or
                ""
            )
            # Strip HTML if present
            if text and "<" in text:
                import re
                text = re.sub(r'<[^>]+>', '', text)
            if text:
                texts.append(text)
        return "\n\n".join(texts)

    def to_case(self) -> Case:
        """Convert to our Case model."""
        try:
            date = datetime.strptime(self.date_filed, "%Y-%m-%d")
        except (ValueError, TypeError):
            date = datetime.now()

        court = Court(
            slug=self.court_id,
            name=self.court_name,
            name_abbreviation=self.court_id.upper()
        )

        return Case(
            id=f"cl_{self.cluster_id}",
            cap_id=self.cluster_id,
            name=self.case_name,
            name_abbreviation=self.case_name[:100] if self.case_name else None,
            date=date,
            court=court,
            text=self.full_text,
            docket_number=self.docket_number,
            citations=self.citations,
            citations_raw=[]
        )


class CourtListenerClient:
    """Async client for CourtListener API.

    Requires API key from https://www.courtlistener.com/sign-in/
    Set COURTLISTENER_API_KEY environment variable.

    Features:
    - Rate limiting (5,000 req/hour)
    - Automatic pagination
    - Response caching
    - Retry with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = COURTLISTENER_API_BASE,
        max_concurrent: int = 5,
        timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize CourtListener client.

        Args:
            api_key: CourtListener API token (or set COURTLISTENER_API_KEY env var)
            base_url: API base URL
            max_concurrent: Max concurrent requests
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Initial delay between retries
        """
        self.api_key = api_key or COURTLISTENER_API_KEY
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.api_key:
            raise ValueError(
                "CourtListener API key required. "
                "Sign up at https://www.courtlistener.com/sign-in/ "
                "and set COURTLISTENER_API_KEY environment variable."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "LegalGPT/1.0"
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=self.timeout
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        full_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a rate-limited API request with retry logic.

        Args:
            endpoint: API endpoint (e.g., "opinions/")
            params: Query parameters
            full_url: Full URL (for pagination)

        Returns:
            JSON response data
        """
        async with self.semaphore:
            session = await self._get_session()
            url = full_url or f"{self.base_url}/{endpoint.lstrip('/')}"

            for attempt in range(self.retry_attempts):
                try:
                    async with session.get(url, params=params if not full_url else None) as response:
                        if response.status == 401:
                            raise ValueError("Invalid API key. Check COURTLISTENER_API_KEY.")
                        if response.status == 429:
                            # Rate limited - wait and retry
                            wait_time = 60  # Wait 1 minute
                            logger.warning(f"Rate limited. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

    async def get_cluster(self, cluster_id) -> Optional[Dict[str, Any]]:
        """Get an opinion cluster by ID.

        A cluster groups related opinions (majority, dissent, concurrence).

        Args:
            cluster_id: Cluster ID (int, dict with 'id', or URL string)

        Returns:
            Cluster data or None
        """
        # Extract numeric ID from various formats
        numeric_id = extract_id_from_url_or_value(cluster_id, "clusters")
        if numeric_id is None:
            logger.warning(f"Could not extract cluster ID from: {cluster_id}")
            return None

        try:
            return await self._request(f"clusters/{numeric_id}/")
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise

    async def get_opinions_for_cluster(self, cluster_id) -> List[Dict[str, Any]]:
        """Get all opinions for a cluster.

        Args:
            cluster_id: Cluster ID (int, dict with 'id', or URL string)

        Returns:
            List of opinion objects
        """
        # Extract numeric ID from various formats
        numeric_id = extract_id_from_url_or_value(cluster_id, "clusters")
        if numeric_id is None:
            logger.warning(f"Could not extract cluster ID from: {cluster_id}")
            return []

        opinions = []
        params = {"cluster": numeric_id}

        data = await self._request("opinions/", params)
        opinions.extend(data.get("results", []))

        # Handle pagination
        while data.get("next"):
            data = await self._request("", full_url=data["next"])
            opinions.extend(data.get("results", []))

        return opinions

    async def get_case_by_cluster_id(
        self,
        cluster_id,
        save_to_disk: bool = True
    ) -> Optional[CourtListenerCase]:
        """Fetch a complete case by cluster ID.

        Args:
            cluster_id: Opinion cluster ID (int, dict with 'id', or URL string)
            save_to_disk: Save to data/raw/cases/

        Returns:
            CourtListenerCase or None
        """
        # Extract numeric ID from various formats
        numeric_id = extract_id_from_url_or_value(cluster_id, "clusters")
        if numeric_id is None:
            logger.warning(f"Could not extract cluster ID from: {cluster_id}")
            return None

        cluster = await self.get_cluster(numeric_id)
        if not cluster:
            return None

        # Get opinions
        opinions = await self.get_opinions_for_cluster(numeric_id)

        # Extract court info - handle nested objects or URLs
        docket = cluster.get("docket") or {}
        court_id = "scotus"
        court_name = "Supreme Court of the United States"
        docket_number = None

        if isinstance(docket, dict):
            court = docket.get("court") or {}
            if isinstance(court, dict):
                court_id = court.get("id", "scotus")
                court_name = court.get("full_name", "Supreme Court of the United States")
            docket_number = docket.get("docket_number")

        # Extract citations
        citations = []
        for c in cluster.get("citations", []):
            if isinstance(c, dict) and c.get("cite"):
                citations.append(c.get("cite"))
            elif isinstance(c, str):
                citations.append(c)

        case = CourtListenerCase(
            cluster_id=numeric_id,
            case_name=cluster.get("case_name", "Unknown"),
            date_filed=cluster.get("date_filed"),
            court_id=court_id,
            court_name=court_name,
            opinions=opinions,
            docket_number=docket_number,
            citations=citations
        )

        if save_to_disk:
            await self._save_case(case, cluster, opinions)

        return case

    async def search_scotus_opinions(
        self,
        date_filed_min: Optional[str] = None,
        date_filed_max: Optional[str] = None,
        max_results: Optional[int] = None,
        save_to_disk: bool = True
    ) -> AsyncGenerator[CourtListenerCase, None]:
        """Search for Supreme Court opinions.

        Args:
            date_filed_min: Minimum date (YYYY-MM-DD)
            date_filed_max: Maximum date (YYYY-MM-DD)
            max_results: Maximum results to return
            save_to_disk: Save cases to disk

        Yields:
            CourtListenerCase objects
        """
        params = {
            "cluster__docket__court": "scotus",
            "order_by": "-cluster__date_filed"
        }

        if date_filed_min:
            params["cluster__date_filed__gte"] = date_filed_min
        if date_filed_max:
            params["cluster__date_filed__lte"] = date_filed_max

        count = 0
        next_url = None

        while True:
            if max_results and count >= max_results:
                break

            try:
                if next_url:
                    data = await self._request("", full_url=next_url)
                else:
                    data = await self._request("opinions/", params)
            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch opinions: {e}")
                break

            # Group opinions by cluster
            cluster_ids_seen = set()

            for opinion in data.get("results", []):
                if max_results and count >= max_results:
                    break

                # Extract cluster ID from various formats using helper
                cluster_id = extract_id_from_url_or_value(
                    opinion.get("cluster"), "clusters"
                )
                if cluster_id is None:
                    continue

                if cluster_id not in cluster_ids_seen:
                    cluster_ids_seen.add(cluster_id)

                    try:
                        case = await self.get_case_by_cluster_id(cluster_id, save_to_disk)
                        if case:
                            yield case
                            count += 1
                    except Exception as e:
                        logger.warning(f"Failed to fetch cluster {cluster_id}: {e}")
                        continue

            # Pagination
            next_url = data.get("next")
            if not next_url:
                break

        logger.info(f"Retrieved {count} SCOTUS cases")

    async def search_by_citation(
        self,
        citation: str,
        save_to_disk: bool = True
    ) -> Optional[CourtListenerCase]:
        """Search for a case by citation.

        Args:
            citation: Case citation (e.g., "347 U.S. 483")
            save_to_disk: Save to disk

        Returns:
            First matching case or None
        """
        # Use search endpoint
        params = {
            "citation": citation,
            "type": "o",  # opinions
        }

        try:
            data = await self._request("search/", params)
            results = data.get("results", [])

            if results:
                # Extract cluster ID from various formats
                cluster_id = extract_id_from_url_or_value(
                    results[0].get("cluster_id"), "clusters"
                )
                if cluster_id:
                    return await self.get_case_by_cluster_id(cluster_id, save_to_disk)
        except Exception as e:
            logger.warning(f"Citation search failed: {e}")

        return None

    async def _save_case(
        self,
        case: CourtListenerCase,
        cluster: Dict[str, Any],
        opinions: List[Dict[str, Any]]
    ) -> Path:
        """Save case data to disk.

        Args:
            case: CourtListenerCase object
            cluster: Raw cluster data
            opinions: Raw opinion data

        Returns:
            Path to saved file
        """
        CASES_DIR.mkdir(parents=True, exist_ok=True)

        file_path = CASES_DIR / f"cl_{case.cluster_id}.json"

        data = {
            "cluster_id": case.cluster_id,
            "case_name": case.case_name,
            "date_filed": case.date_filed,
            "court_id": case.court_id,
            "court_name": case.court_name,
            "docket_number": case.docket_number,
            "citations": case.citations,
            "text": case.full_text,
            "raw_cluster": cluster,
            "raw_opinions": opinions
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return file_path

    async def download_scotus_cases(
        self,
        max_cases: int = 1000,
        save_raw: bool = True
    ) -> List[Case]:
        """Download SCOTUS cases.

        Args:
            max_cases: Maximum cases to download
            save_raw: Save raw JSON

        Returns:
            List of Case objects
        """
        cases = []

        logger.info(f"Downloading up to {max_cases} SCOTUS cases from CourtListener...")

        async for cl_case in tqdm(
            self.search_scotus_opinions(max_results=max_cases, save_to_disk=save_raw),
            total=max_cases,
            desc="Downloading SCOTUS cases"
        ):
            try:
                case = cl_case.to_case()
                cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to convert case {cl_case.cluster_id}: {e}")

        logger.info(f"Downloaded {len(cases)} cases")
        return cases


# Alias for backward compatibility
CaseClient = CourtListenerClient


async def download_sample_scotus(n: int = 100) -> List[Case]:
    """Download a sample of SCOTUS cases.

    Args:
        n: Number of cases

    Returns:
        List of Case objects
    """
    async with CourtListenerClient() as client:
        return await client.download_scotus_cases(max_cases=n)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download cases from CourtListener")
    parser.add_argument("--max-cases", type=int, default=10, help="Max cases to download")
    parser.add_argument("--citation", help="Search by citation")
    parser.add_argument("--cluster-id", type=int, help="Fetch specific cluster")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    async def main():
        try:
            async with CourtListenerClient() as client:
                if args.cluster_id:
                    case = await client.get_case_by_cluster_id(args.cluster_id)
                    if case:
                        print(f"Found: {case.case_name}")
                        print(f"Date: {case.date_filed}")
                        print(f"Text length: {len(case.full_text)} chars")
                    else:
                        print("Case not found")
                elif args.citation:
                    case = await client.search_by_citation(args.citation)
                    if case:
                        print(f"Found: {case.case_name}")
                    else:
                        print("Case not found")
                else:
                    cases = await client.download_scotus_cases(max_cases=args.max_cases)
                    print(f"\nDownloaded {len(cases)} cases")
                    if cases:
                        print(f"Sample: {cases[0].name}")
        except ValueError as e:
            print(f"Error: {e}")
            print("\nTo get an API key:")
            print("1. Sign up at https://www.courtlistener.com/sign-in/")
            print("2. Go to https://www.courtlistener.com/profile/")
            print("3. Copy your API token")
            print("4. Set: export COURTLISTENER_API_KEY='your-token'")

    asyncio.run(main())
