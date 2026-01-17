"""Main citation extraction pipeline.

Provides the CitationExtractor class for extracting citations from legal case text
and converting them into structured ExtractedCitation objects.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from .patterns import (
    extract_all_citations,
    normalize_citation,
    classify_citation,
    extract_citation_components,
    is_valid_citation,
    split_parallel_citations,
)

logger = logging.getLogger(__name__)


class ExtractedCitation(BaseModel):
    """A citation extracted from case text."""

    raw_text: str = Field(description="Original citation text as found")
    normalized: str = Field(description="Normalized citation format")
    volume: str = Field(description="Volume number")
    reporter: str = Field(description="Reporter abbreviation")
    page: str = Field(description="Starting page number")
    pin_cite: Optional[str] = Field(None, description="Pin cite page if present")
    classification: str = Field(description="Citation type classification")
    source_case_id: Optional[str] = Field(None, description="ID of case containing this citation")
    context: Optional[str] = Field(None, description="Surrounding text context")
    position: Optional[int] = Field(None, description="Character position in source text")

    def to_search_query(self) -> str:
        """Convert to a search query format for CAP API."""
        return f"{self.volume} {self.reporter} {self.page}"

    @property
    def is_supreme_court(self) -> bool:
        """Check if this is a Supreme Court citation."""
        return self.classification.startswith("supreme_court")

    @property
    def is_federal(self) -> bool:
        """Check if this is a federal court citation."""
        return self.classification.startswith("federal") or self.is_supreme_court


class ExtractionStats(BaseModel):
    """Statistics from a citation extraction run."""

    total_cases: int = 0
    cases_with_citations: int = 0
    total_citations: int = 0
    unique_citations: int = 0
    citations_by_type: Dict[str, int] = Field(default_factory=dict)
    avg_citations_per_case: float = 0.0
    extraction_errors: int = 0
    error_messages: List[str] = Field(default_factory=list)

    def add_citation_type(self, citation_type: str) -> None:
        """Increment count for a citation type."""
        self.citations_by_type[citation_type] = self.citations_by_type.get(citation_type, 0) + 1


class CitationExtractor:
    """Main extraction pipeline for legal citations.

    Extracts citations from case text, normalizes them, and classifies them
    by reporter type.

    Usage:
        extractor = CitationExtractor()
        citations = extractor.extract(case_text)

        # For batch processing
        results = extractor.extract_batch(cases)
    """

    def __init__(
        self,
        context_window: int = 100,
        deduplicate: bool = True,
        max_workers: int = 4,
        min_citation_length: int = 5,
    ):
        """Initialize the citation extractor.

        Args:
            context_window: Characters of surrounding context to capture
            deduplicate: Whether to remove duplicate citations from same case
            max_workers: Max parallel workers for batch processing
            min_citation_length: Minimum length for valid citation
        """
        self.context_window = context_window
        self.deduplicate = deduplicate
        self.max_workers = max_workers
        self.min_citation_length = min_citation_length
        self._stats = ExtractionStats()

    def extract(
        self,
        case_text: str,
        case_id: Optional[str] = None,
        include_context: bool = True,
    ) -> List[ExtractedCitation]:
        """Extract citations from a single case text.

        Args:
            case_text: Full text of the legal case
            case_id: Optional ID of the source case
            include_context: Whether to include surrounding context

        Returns:
            List of ExtractedCitation objects
        """
        if not case_text or len(case_text) < self.min_citation_length:
            return []

        try:
            raw_citations = extract_all_citations(case_text)
        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
            self._stats.extraction_errors += 1
            self._stats.error_messages.append(str(e))
            return []

        extracted = []
        seen_normalized: Set[str] = set()

        for raw in raw_citations:
            try:
                normalized = normalize_citation(raw)

                # Skip if duplicate and deduplication is enabled
                if self.deduplicate and normalized in seen_normalized:
                    continue
                seen_normalized.add(normalized)

                # Validate citation
                if not is_valid_citation(normalized):
                    continue

                # Extract components
                volume, reporter, page = extract_citation_components(normalized)

                # Classify citation type
                classification = classify_citation(normalized)

                # Find position and context
                position = None
                context = None
                if include_context:
                    # Find the citation in the original text
                    idx = case_text.find(raw)
                    if idx >= 0:
                        position = idx
                        start = max(0, idx - self.context_window)
                        end = min(len(case_text), idx + len(raw) + self.context_window)
                        context = case_text[start:end].strip()
                        # Clean up context
                        context = " ".join(context.split())

                citation = ExtractedCitation(
                    raw_text=raw,
                    normalized=normalized,
                    volume=volume,
                    reporter=reporter,
                    page=page,
                    classification=classification,
                    source_case_id=case_id,
                    context=context,
                    position=position,
                )
                extracted.append(citation)

                # Update stats
                self._stats.add_citation_type(classification)

            except Exception as e:
                logger.debug(f"Error processing citation '{raw}': {e}")
                continue

        return extracted

    def extract_from_case(self, case: "Case") -> List[ExtractedCitation]:
        """Extract citations from a Case object.

        Args:
            case: Case object with text and id

        Returns:
            List of ExtractedCitation objects
        """
        return self.extract(case.text, case_id=case.id)

    def extract_batch(
        self,
        cases: List["Case"],
        show_progress: bool = True,
    ) -> Dict[str, List[ExtractedCitation]]:
        """Extract citations from multiple cases.

        Args:
            cases: List of Case objects to process
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping case ID to list of extracted citations
        """
        results: Dict[str, List[ExtractedCitation]] = {}
        self._stats = ExtractionStats()  # Reset stats

        if not cases:
            return results

        all_unique_citations: Set[str] = set()

        # Process cases (can be parallelized for large batches)
        if len(cases) > 100 and self.max_workers > 1:
            # Parallel processing for large batches
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_case = {
                    executor.submit(self.extract_from_case, case): case
                    for case in cases
                }

                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        citations = future.result()
                        results[case.id] = citations

                        # Track unique citations
                        for c in citations:
                            all_unique_citations.add(c.normalized)

                    except Exception as e:
                        logger.warning(f"Error processing case {case.id}: {e}")
                        results[case.id] = []
                        self._stats.extraction_errors += 1
                        self._stats.error_messages.append(f"Case {case.id}: {str(e)}")
        else:
            # Sequential processing for smaller batches
            iterator = cases
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(cases, desc="Extracting citations")
                except ImportError:
                    pass

            for case in iterator:
                try:
                    citations = self.extract_from_case(case)
                    results[case.id] = citations

                    # Track unique citations
                    for c in citations:
                        all_unique_citations.add(c.normalized)

                except Exception as e:
                    logger.warning(f"Error processing case {case.id}: {e}")
                    results[case.id] = []
                    self._stats.extraction_errors += 1
                    self._stats.error_messages.append(f"Case {case.id}: {str(e)}")

        # Calculate final stats
        self._stats.total_cases = len(cases)
        self._stats.cases_with_citations = sum(1 for cites in results.values() if cites)
        self._stats.total_citations = sum(len(cites) for cites in results.values())
        self._stats.unique_citations = len(all_unique_citations)

        if self._stats.total_cases > 0:
            self._stats.avg_citations_per_case = (
                self._stats.total_citations / self._stats.total_cases
            )

        return results

    def get_stats(self) -> ExtractionStats:
        """Get extraction statistics.

        Returns:
            ExtractionStats object with current statistics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self._stats = ExtractionStats()


def extract_citations_from_text(text: str) -> List[ExtractedCitation]:
    """Convenience function to extract citations from text.

    Args:
        text: Text to extract citations from

    Returns:
        List of ExtractedCitation objects
    """
    extractor = CitationExtractor()
    return extractor.extract(text)


if __name__ == "__main__":
    # Test the extractor
    test_text = """
    The Court in Brown v. Board of Education, 347 U.S. 483, 74 S.Ct. 686 (1954),
    overruled Plessy v. Ferguson, 163 U.S. 537 (1896). The Ninth Circuit in
    Smith v. Jones, 123 F.3d 456, 460 (9th Cir. 1997), applied this principle.
    See also 789 F.Supp.2d 123 (D.D.C. 2010).
    """

    extractor = CitationExtractor()
    citations = extractor.extract(test_text, case_id="test-001")

    print(f"Found {len(citations)} citations:")
    for c in citations:
        print(f"  {c.normalized} [{c.classification}]")
        if c.context:
            print(f"    Context: ...{c.context[:80]}...")

    print(f"\nStats: {extractor.get_stats()}")
