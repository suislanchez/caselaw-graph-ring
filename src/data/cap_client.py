"""Async client for Caselaw Access Project API.

Uses aiohttp for async HTTP requests with rate limiting and pagination handling.
API Documentation: https://case.law/api/
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any
from datetime import datetime

import aiohttp
from tqdm.asyncio import tqdm

from src.config import CAP_API_KEY, CAP_API_BASE, RAW_DIR, CASES_DIR
from src.data.case_schema import CAPCase, Case

logger = logging.getLogger(__name__)


class CAPClient:
    """Async client for the Caselaw Access Project API.

    Features:
    - Rate limiting via asyncio.Semaphore
    - Automatic pagination handling
    - Response caching to disk
    - Retry logic for transient failures
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = CAP_API_BASE,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize CAP client.

        Args:
            api_key: CAP API key (defaults to env var CAP_API_KEY)
            base_url: API base URL
            max_concurrent: Max concurrent requests (rate limiting)
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or CAP_API_KEY
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.api_key:
            logger.warning("No CAP API key provided. Some features may be limited.")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "LegalGPT/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        return headers

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
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a rate-limited API request with retry logic.

        Args:
            endpoint: API endpoint (e.g., "cases/123/")
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On request failure after retries
        """
        async with self.semaphore:
            session = await self._get_session()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"

            for attempt in range(self.retry_attempts):
                try:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

    async def get_case_by_id(
        self,
        case_id: int,
        full_case: bool = True,
        save_to_disk: bool = True
    ) -> Optional[CAPCase]:
        """Fetch a single case by its CAP ID.

        Args:
            case_id: CAP case ID
            full_case: Include full case body (text)
            save_to_disk: Save raw response to data/raw/cases/

        Returns:
            CAPCase object or None if not found
        """
        params = {"full_case": "true"} if full_case else {}

        try:
            data = await self._request(f"cases/{case_id}/", params)
            cap_case = CAPCase(**data)

            if save_to_disk:
                await self._save_raw_response(case_id, data)

            return cap_case
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.warning(f"Case {case_id} not found")
                return None
            raise

    async def get_case_by_citation(
        self,
        citation: str,
        full_case: bool = True,
        save_to_disk: bool = True
    ) -> Optional[CAPCase]:
        """Fetch a case by its citation.

        Args:
            citation: Case citation (e.g., "347 U.S. 483")
            full_case: Include full case body
            save_to_disk: Save raw response to disk

        Returns:
            First matching CAPCase or None
        """
        async for case in self.search_cases(
            cite=citation,
            full_case=full_case,
            max_results=1,
            save_to_disk=save_to_disk
        ):
            return case
        return None

    async def search_cases(
        self,
        search: Optional[str] = None,
        cite: Optional[str] = None,
        court: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        decision_date_min: Optional[str] = None,
        decision_date_max: Optional[str] = None,
        full_case: bool = True,
        page_size: int = 100,
        max_results: Optional[int] = None,
        save_to_disk: bool = True
    ) -> AsyncGenerator[CAPCase, None]:
        """Search cases with automatic pagination.

        Args:
            search: Full-text search query
            cite: Citation to search for (e.g., "123 U.S. 456")
            court: Court slug filter (e.g., "us" for Supreme Court)
            jurisdiction: Jurisdiction slug filter
            decision_date_min: Minimum decision date (YYYY-MM-DD)
            decision_date_max: Maximum decision date (YYYY-MM-DD)
            full_case: Include full case body
            page_size: Results per page (max 100)
            max_results: Maximum total results to return
            save_to_disk: Save raw responses to disk

        Yields:
            CAPCase objects
        """
        params = {
            "page_size": min(page_size, 100),
            "ordering": "-decision_date"
        }

        if full_case:
            params["full_case"] = "true"
        if search:
            params["search"] = search
        if cite:
            params["cite"] = cite
        if court:
            params["court"] = court
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        if decision_date_min:
            params["decision_date_min"] = decision_date_min
        if decision_date_max:
            params["decision_date_max"] = decision_date_max

        count = 0
        next_url = "cases/"
        current_params = params

        while next_url:
            if max_results and count >= max_results:
                break

            try:
                data = await self._request(next_url, current_params)
            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch cases: {e}")
                break

            for result in data.get("results", []):
                if max_results and count >= max_results:
                    break

                try:
                    cap_case = CAPCase(**result)

                    if save_to_disk:
                        await self._save_raw_response(cap_case.id, result)

                    yield cap_case
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse case: {e}")
                    continue

            # Handle pagination
            next_url = data.get("next")
            if next_url:
                # Extract just the path from the full URL
                next_url = next_url.replace(self.base_url, "").lstrip("/")
                current_params = None  # Params are encoded in next_url

        logger.info(f"Retrieved {count} cases")

    async def get_scotus_cases(
        self,
        decision_date_min: Optional[str] = None,
        decision_date_max: Optional[str] = None,
        full_case: bool = True,
        max_results: Optional[int] = None,
        save_to_disk: bool = True
    ) -> AsyncGenerator[CAPCase, None]:
        """Get Supreme Court cases.

        Args:
            decision_date_min: Minimum decision date (YYYY-MM-DD)
            decision_date_max: Maximum decision date (YYYY-MM-DD)
            full_case: Include full case body
            max_results: Maximum results to return
            save_to_disk: Save raw responses to disk

        Yields:
            CAPCase objects for SCOTUS cases
        """
        async for case in self.search_cases(
            court="us",
            jurisdiction="us",
            decision_date_min=decision_date_min,
            decision_date_max=decision_date_max,
            full_case=full_case,
            max_results=max_results,
            save_to_disk=save_to_disk
        ):
            yield case

    async def _save_raw_response(self, case_id: int, data: Dict[str, Any]) -> Path:
        """Save raw API response to disk.

        Args:
            case_id: Case ID for filename
            data: Raw JSON response data

        Returns:
            Path to saved file
        """
        output_dir = CASES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{case_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return file_path

    async def download_cases(
        self,
        output_dir: Optional[Path] = None,
        court: str = "us",
        max_cases: int = 1000,
        save_raw: bool = True
    ) -> List[Case]:
        """Download cases and save to disk.

        Args:
            output_dir: Directory to save cases (defaults to CASES_DIR)
            court: Court to download from
            max_cases: Maximum number of cases
            save_raw: Whether to save raw JSON files

        Returns:
            List of Case objects
        """
        output_dir = output_dir or CASES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        cases = []

        logger.info(f"Downloading up to {max_cases} cases from court '{court}'...")

        async for cap_case in tqdm(
            self.search_cases(court=court, max_results=max_cases, save_to_disk=save_raw),
            total=max_cases,
            desc="Downloading cases"
        ):
            try:
                case = cap_case.to_case()
                cases.append(case)

                if save_raw:
                    case_file = output_dir / f"{case.id}.json"
                    with open(case_file, "w") as f:
                        json.dump(case.model_dump(mode="json"), f, indent=2, default=str)

            except Exception as e:
                logger.warning(f"Failed to process case {cap_case.id}: {e}")
                continue

        logger.info(f"Downloaded {len(cases)} cases")
        return cases


async def download_sample_scotus(n: int = 100) -> List[Case]:
    """Download a sample of SCOTUS cases for testing.

    Args:
        n: Number of cases to download

    Returns:
        List of Case objects
    """
    async with CAPClient() as client:
        return await client.download_cases(court="us", max_cases=n)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download cases from CAP API")
    parser.add_argument("--court", default="us", help="Court slug (default: us for SCOTUS)")
    parser.add_argument("--max-cases", type=int, default=100, help="Max cases to download")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--citation", help="Search by citation")
    parser.add_argument("--case-id", type=int, help="Fetch specific case by ID")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    async def main():
        async with CAPClient() as client:
            if args.case_id:
                case = await client.get_case_by_id(args.case_id)
                if case:
                    print(f"Found case: {case.name}")
                else:
                    print("Case not found")
            elif args.citation:
                case = await client.get_case_by_citation(args.citation)
                if case:
                    print(f"Found case: {case.name}")
                else:
                    print("Case not found")
            else:
                cases = await client.download_cases(
                    output_dir=args.output_dir,
                    court=args.court,
                    max_cases=args.max_cases
                )
                print(f"\nDownloaded {len(cases)} cases")
                if cases:
                    print(f"Sample case: {cases[0].name}")

    asyncio.run(main())
