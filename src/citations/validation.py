"""Quality validation for citation extraction and linking.

Provides sample-based accuracy checking, extraction statistics reporting,
and identification of common failure patterns.
"""

import asyncio
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple

from pydantic import BaseModel, Field

from src.config import CITATIONS_DIR
from .extractor import CitationExtractor, ExtractedCitation
from .linker import CitationLinker
from .graph_edges import CitationEdge
from .patterns import classify_citation

logger = logging.getLogger(__name__)


class ExtractionReport(BaseModel):
    """Report on citation extraction quality."""

    sample_size: int = 0
    total_citations_extracted: int = 0
    unique_citations: int = 0
    avg_citations_per_case: float = 0.0
    median_citations_per_case: float = 0.0
    max_citations_per_case: int = 0
    min_citations_per_case: int = 0
    cases_with_no_citations: int = 0

    # Distribution by type
    citations_by_type: Dict[str, int] = Field(default_factory=dict)

    # Quality metrics
    potential_false_positives: List[str] = Field(default_factory=list)
    suspicious_patterns: List[str] = Field(default_factory=list)

    # Text analysis
    avg_text_length: float = 0.0
    citations_per_1000_chars: float = 0.0

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class LinkingReport(BaseModel):
    """Report on citation linking quality."""

    total_citations: int = 0
    successful_links: int = 0
    ambiguous_links: int = 0
    failed_links: int = 0
    success_rate: float = 0.0

    # Breakdown by citation type
    success_by_type: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Failure analysis
    common_failure_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    sample_failures: List[str] = Field(default_factory=list)

    # Cache stats
    cache_hits: int = 0
    cache_hit_rate: float = 0.0

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ValidationReport(BaseModel):
    """Combined validation report."""

    extraction: ExtractionReport = Field(default_factory=ExtractionReport)
    linking: Optional[LinkingReport] = None
    edge_stats: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)
    overall_quality_score: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def validate_extraction(
    cases_sample: List["Case"],
    extractor: Optional[CitationExtractor] = None,
) -> ExtractionReport:
    """Validate citation extraction on a sample of cases.

    Analyzes extraction quality, identifies potential issues,
    and reports statistics.

    Args:
        cases_sample: Sample of Case objects to validate
        extractor: CitationExtractor instance (creates one if not provided)

    Returns:
        ExtractionReport with validation results
    """
    extractor = extractor or CitationExtractor()
    report = ExtractionReport()
    report.sample_size = len(cases_sample)

    if not cases_sample:
        return report

    # Extract citations from all cases
    case_citations = extractor.extract_batch(cases_sample, show_progress=False)

    # Collect statistics
    all_citations: Set[str] = set()
    citations_per_case: List[int] = []
    text_lengths: List[int] = []
    type_counts: Dict[str, int] = defaultdict(int)

    for case in cases_sample:
        citations = case_citations.get(case.id, [])
        count = len(citations)
        citations_per_case.append(count)
        text_lengths.append(len(case.text))

        for c in citations:
            all_citations.add(c.normalized)
            type_counts[c.classification] += 1

    # Basic stats
    report.total_citations_extracted = sum(citations_per_case)
    report.unique_citations = len(all_citations)
    report.avg_citations_per_case = (
        sum(citations_per_case) / len(citations_per_case) if citations_per_case else 0
    )
    report.max_citations_per_case = max(citations_per_case) if citations_per_case else 0
    report.min_citations_per_case = min(citations_per_case) if citations_per_case else 0
    report.cases_with_no_citations = sum(1 for c in citations_per_case if c == 0)

    # Median
    sorted_counts = sorted(citations_per_case)
    n = len(sorted_counts)
    if n > 0:
        report.median_citations_per_case = (
            sorted_counts[n // 2] if n % 2 else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
        )

    # Type distribution
    report.citations_by_type = dict(type_counts)

    # Text analysis
    report.avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    total_chars = sum(text_lengths)
    if total_chars > 0:
        report.citations_per_1000_chars = (report.total_citations_extracted / total_chars) * 1000

    # Quality checks - identify potential issues
    report.potential_false_positives = _find_potential_false_positives(case_citations)
    report.suspicious_patterns = _find_suspicious_patterns(case_citations)

    return report


def _find_potential_false_positives(
    case_citations: Dict[str, List[ExtractedCitation]],
) -> List[str]:
    """Identify citations that might be false positives.

    Args:
        case_citations: Dictionary of case ID to citations

    Returns:
        List of suspicious citation strings
    """
    suspicious = []

    for case_id, citations in case_citations.items():
        for c in citations:
            # Check for unusual patterns
            try:
                volume = int(c.volume)
                page = int(c.page)

                # Volume 0 is suspicious
                if volume == 0:
                    suspicious.append(f"{c.normalized} (zero volume)")

                # Very high page numbers relative to volume
                if page > 10000:
                    suspicious.append(f"{c.normalized} (very high page number)")

                # Volume higher than page (unusual for most reporters)
                if volume > page and page < 10:
                    suspicious.append(f"{c.normalized} (volume > page)")

            except ValueError:
                suspicious.append(f"{c.normalized} (non-numeric volume/page)")

    return suspicious[:20]  # Limit to 20 examples


def _find_suspicious_patterns(
    case_citations: Dict[str, List[ExtractedCitation]],
) -> List[str]:
    """Identify suspicious patterns in extraction.

    Args:
        case_citations: Dictionary of case ID to citations

    Returns:
        List of pattern descriptions
    """
    patterns = []

    # Count citation frequency
    citation_freq: Dict[str, int] = defaultdict(int)
    for citations in case_citations.values():
        for c in citations:
            citation_freq[c.normalized] += 1

    # Check for unusually frequent citations (might be false positives)
    total_cases = len(case_citations)
    for citation, freq in citation_freq.items():
        if freq > total_cases * 0.5 and total_cases > 10:
            patterns.append(
                f"Citation '{citation}' appears in {freq}/{total_cases} cases "
                f"({freq/total_cases*100:.1f}%) - possible false positive"
            )

    # Check for cases with abnormally high citation counts
    for case_id, citations in case_citations.items():
        if len(citations) > 100:
            patterns.append(
                f"Case {case_id} has {len(citations)} citations - "
                "unusually high, check for extraction issues"
            )

    return patterns[:10]  # Limit to 10 patterns


async def validate_linking(
    edges: List[CitationEdge],
    linker: Optional[CitationLinker] = None,
    sample_size: int = 100,
) -> LinkingReport:
    """Validate citation linking quality.

    Analyzes linking success rates, identifies failure patterns,
    and provides quality metrics.

    Args:
        edges: List of CitationEdge objects
        linker: CitationLinker instance (creates one if not provided)
        sample_size: Number of edges to sample for validation

    Returns:
        LinkingReport with validation results
    """
    linker = linker or CitationLinker()
    report = LinkingReport()

    if not edges:
        return report

    # Get linker stats
    stats = linker.get_stats()
    report.total_citations = stats.get("total_citations", 0)
    report.successful_links = stats.get("successful_links", 0)
    report.ambiguous_links = stats.get("ambiguous_links", 0)
    report.failed_links = stats.get("failed_links", 0)
    report.cache_hits = stats.get("cache_hits", 0)

    if report.total_citations > 0:
        report.success_rate = report.successful_links / report.total_citations
        report.cache_hit_rate = report.cache_hits / report.total_citations

    # Analyze by citation type
    type_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "linked": 0})

    for edge in edges:
        citation_type = edge.citation_type or "unknown"
        type_stats[citation_type]["total"] += 1
        if edge.target_case_id:
            type_stats[citation_type]["linked"] += 1

    report.success_by_type = dict(type_stats)

    # Sample unlinked citations for failure analysis
    unlinked = linker.get_unlinked_citations()
    report.sample_failures = unlinked[:20]

    # Identify common failure patterns
    report.common_failure_patterns = _analyze_failure_patterns(unlinked)

    return report


def _analyze_failure_patterns(unlinked: List[str]) -> List[Dict[str, Any]]:
    """Analyze patterns in linking failures.

    Args:
        unlinked: List of unlinked citation strings

    Returns:
        List of pattern analysis dictionaries
    """
    patterns = []

    if not unlinked:
        return patterns

    # Group by citation type
    type_failures: Dict[str, int] = defaultdict(int)
    for citation in unlinked:
        citation_type = classify_citation(citation)
        type_failures[citation_type] += 1

    # Report most common failure types
    sorted_types = sorted(type_failures.items(), key=lambda x: x[1], reverse=True)
    for citation_type, count in sorted_types[:5]:
        patterns.append({
            "pattern": f"Unlinked {citation_type} citations",
            "count": count,
            "percentage": count / len(unlinked) * 100 if unlinked else 0,
            "examples": [c for c in unlinked if classify_citation(c) == citation_type][:3],
        })

    return patterns


def run_full_validation(
    cases: List["Case"],
    edges: Optional[List[CitationEdge]] = None,
    sample_size: int = 100,
) -> ValidationReport:
    """Run complete validation on extraction and linking.

    Args:
        cases: Full list of cases
        edges: Optional pre-built edge list
        sample_size: Sample size for validation

    Returns:
        ValidationReport with all validation results
    """
    report = ValidationReport()

    # Sample cases for extraction validation
    sample = cases if len(cases) <= sample_size else random.sample(cases, sample_size)

    # Validate extraction
    logger.info(f"Validating extraction on {len(sample)} cases...")
    report.extraction = validate_extraction(sample)

    # Validate linking if edges provided
    if edges:
        logger.info(f"Validating linking on {len(edges)} edges...")
        report.linking = asyncio.run(validate_linking(edges))

        # Calculate edge stats
        report.edge_stats = _calculate_edge_stats(edges)

    # Generate recommendations
    report.recommendations = _generate_recommendations(report)

    # Calculate overall quality score
    report.overall_quality_score = _calculate_quality_score(report)

    return report


def _calculate_edge_stats(edges: List[CitationEdge]) -> Dict[str, Any]:
    """Calculate statistics about the edge list.

    Args:
        edges: List of CitationEdge objects

    Returns:
        Dictionary of edge statistics
    """
    if not edges:
        return {}

    # Basic counts
    source_cases = set(e.source_case_id for e in edges)
    target_cases = set(e.target_case_id for e in edges)

    # Degree distributions
    out_degree: Dict[str, int] = defaultdict(int)
    in_degree: Dict[str, int] = defaultdict(int)

    for e in edges:
        out_degree[e.source_case_id] += 1
        in_degree[e.target_case_id] += 1

    # Year span analysis
    years_with_data = [e.source_year for e in edges if e.source_year]

    return {
        "total_edges": len(edges),
        "unique_source_cases": len(source_cases),
        "unique_target_cases": len(target_cases),
        "avg_out_degree": sum(out_degree.values()) / len(out_degree) if out_degree else 0,
        "avg_in_degree": sum(in_degree.values()) / len(in_degree) if in_degree else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "max_in_degree": max(in_degree.values()) if in_degree else 0,
        "year_range": (min(years_with_data), max(years_with_data)) if years_with_data else None,
    }


def _generate_recommendations(report: ValidationReport) -> List[str]:
    """Generate recommendations based on validation results.

    Args:
        report: ValidationReport to analyze

    Returns:
        List of recommendation strings
    """
    recommendations = []

    ext = report.extraction

    # Low extraction rate
    if ext.avg_citations_per_case < 1.0:
        recommendations.append(
            "Low average citations per case. Consider checking if text extraction "
            "is working correctly or if pattern coverage is sufficient."
        )

    # High no-citation cases
    if ext.sample_size > 0:
        no_citation_rate = ext.cases_with_no_citations / ext.sample_size
        if no_citation_rate > 0.3:
            recommendations.append(
                f"{no_citation_rate*100:.1f}% of cases have no citations extracted. "
                "This may indicate text quality issues or pattern gaps."
            )

    # Potential false positives
    if ext.potential_false_positives:
        recommendations.append(
            f"Found {len(ext.potential_false_positives)} potential false positive citations. "
            "Review the patterns module for overly permissive patterns."
        )

    # Suspicious patterns
    if ext.suspicious_patterns:
        recommendations.append(
            f"Found {len(ext.suspicious_patterns)} suspicious extraction patterns. "
            "Manual review recommended."
        )

    # Linking recommendations
    if report.linking:
        link = report.linking
        if link.success_rate < 0.5:
            recommendations.append(
                f"Low linking success rate ({link.success_rate*100:.1f}%). "
                "Consider expanding CAP API search strategies or improving normalization."
            )

        if link.ambiguous_links > link.successful_links * 0.3:
            recommendations.append(
                "High rate of ambiguous links. Consider adding disambiguation logic "
                "using additional metadata like case name or date."
            )

        if link.common_failure_patterns:
            top_failure = link.common_failure_patterns[0]
            recommendations.append(
                f"Most common linking failure: {top_failure['pattern']} "
                f"({top_failure['count']} occurrences). Consider targeted improvements."
            )

    return recommendations


def _calculate_quality_score(report: ValidationReport) -> float:
    """Calculate overall quality score from 0 to 1.

    Args:
        report: ValidationReport to score

    Returns:
        Quality score between 0 and 1
    """
    scores = []

    ext = report.extraction

    # Extraction coverage score (target: > 5 citations per case on average)
    if ext.avg_citations_per_case >= 5:
        scores.append(1.0)
    else:
        scores.append(ext.avg_citations_per_case / 5)

    # No-citation rate penalty
    if ext.sample_size > 0:
        no_citation_rate = ext.cases_with_no_citations / ext.sample_size
        scores.append(1 - no_citation_rate)

    # False positive penalty
    if ext.total_citations_extracted > 0:
        fp_rate = len(ext.potential_false_positives) / ext.total_citations_extracted
        scores.append(1 - min(fp_rate * 10, 1))  # Heavily penalize false positives

    # Linking score
    if report.linking:
        scores.append(report.linking.success_rate)

    return sum(scores) / len(scores) if scores else 0.0


def save_validation_report(
    report: ValidationReport,
    path: Optional[Path] = None,
) -> Path:
    """Save validation report to JSON file.

    Args:
        report: ValidationReport to save
        path: Output path (defaults to CITATIONS_DIR/validation_report.json)

    Returns:
        Path to saved file
    """
    path = path or CITATIONS_DIR / "validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    logger.info(f"Saved validation report to {path}")
    return path


def print_validation_summary(report: ValidationReport) -> None:
    """Print a summary of the validation report.

    Args:
        report: ValidationReport to summarize
    """
    print("\n" + "=" * 60)
    print("CITATION VALIDATION REPORT")
    print("=" * 60)

    print("\n--- Extraction Summary ---")
    ext = report.extraction
    print(f"Sample size: {ext.sample_size} cases")
    print(f"Total citations extracted: {ext.total_citations_extracted}")
    print(f"Unique citations: {ext.unique_citations}")
    print(f"Avg citations per case: {ext.avg_citations_per_case:.2f}")
    print(f"Cases with no citations: {ext.cases_with_no_citations}")

    if ext.citations_by_type:
        print("\nBy type:")
        for ctype, count in sorted(ext.citations_by_type.items(), key=lambda x: -x[1]):
            print(f"  {ctype}: {count}")

    if report.linking:
        print("\n--- Linking Summary ---")
        link = report.linking
        print(f"Success rate: {link.success_rate*100:.1f}%")
        print(f"Successful: {link.successful_links}")
        print(f"Ambiguous: {link.ambiguous_links}")
        print(f"Failed: {link.failed_links}")

    if report.edge_stats:
        print("\n--- Edge Statistics ---")
        for key, value in report.edge_stats.items():
            print(f"{key}: {value}")

    print("\n--- Quality Assessment ---")
    print(f"Overall quality score: {report.overall_quality_score:.2f}/1.00")

    if report.recommendations:
        print("\n--- Recommendations ---")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test validation with sample data
    from src.data.case_schema import Case, Court

    sample_cases = [
        Case(
            id="case-001",
            name="Test v. Example",
            date=datetime(2020, 1, 1),
            court=Court(slug="us", name="Supreme Court"),
            text="""
            The Court in Brown v. Board of Education, 347 U.S. 483 (1954),
            overruled the doctrine established in Plessy v. Ferguson, 163 U.S. 537 (1896).
            This principle was later applied in Loving v. Virginia, 388 U.S. 1 (1967).
            """,
        ),
        Case(
            id="case-002",
            name="Another v. Case",
            date=datetime(2021, 6, 15),
            court=Court(slug="ca9", name="Ninth Circuit"),
            text="""
            Following the Supreme Court's decision in Roe v. Wade, 410 U.S. 113 (1973),
            and its subsequent clarification in Planned Parenthood v. Casey, 505 U.S. 833 (1992),
            we hold that the district court erred. See 789 F.Supp.2d 123 (D.D.C. 2010).
            """,
        ),
        Case(
            id="case-003",
            name="Empty v. Case",
            date=datetime(2022, 1, 1),
            court=Court(slug="ca9", name="Ninth Circuit"),
            text="This case contains no citations.",
        ),
    ]

    print("Running validation on sample cases...")
    report = run_full_validation(sample_cases)
    print_validation_summary(report)

    # Save report
    save_validation_report(report)
