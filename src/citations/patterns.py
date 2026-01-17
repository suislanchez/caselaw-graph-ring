"""Legal citation regex patterns for extracting citations from case text.

Covers all major citation formats:
- Federal Reporter: "123 F.3d 456", "123 F.2d 456", "123 F. 456"
- Supreme Court: "123 U.S. 456", "123 S.Ct. 456", "123 L.Ed. 456", "123 L.Ed.2d 456"
- Federal Supplement: "123 F.Supp. 456", "123 F.Supp.2d 456", "123 F.Supp.3d 456"
- State reporters (common ones)
- Parallel citations
"""

import re
from typing import List, Tuple, Set

# Volume-Reporter-Page pattern components
VOLUME = r'\d{1,4}'
PAGE = r'\d{1,5}'
YEAR = r'\d{4}'
PIN_CITE = r'(?:,?\s*\d{1,5})?'  # Optional pin cite like ", 123"

# Federal Reporter patterns
FEDERAL_REPORTER_PATTERNS = [
    # Federal Reporter (F., F.2d, F.3d, F.4th)
    rf'({VOLUME})\s+F\.(?:\s*(?:2d|3d|4th))?\s+({PAGE}){PIN_CITE}',
    # Federal Appendix (Fed. Appx., F. App'x)
    rf'({VOLUME})\s+(?:Fed\.?\s*Appx?\.?|F\.\s*App\'?x\.?)\s+({PAGE}){PIN_CITE}',
]

# Supreme Court patterns
SUPREME_COURT_PATTERNS = [
    # U.S. Reports: "123 U.S. 456"
    rf'({VOLUME})\s+U\.S\.\s+({PAGE}){PIN_CITE}',
    # Supreme Court Reporter: "123 S.Ct. 456" or "123 S. Ct. 456"
    rf'({VOLUME})\s+S\.?\s*Ct\.\s+({PAGE}){PIN_CITE}',
    # Lawyers' Edition: "123 L.Ed. 456", "123 L.Ed.2d 456"
    rf'({VOLUME})\s+L\.?\s*Ed\.(?:\s*2d)?\s+({PAGE}){PIN_CITE}',
    # Early Supreme Court reporters (by name)
    rf'({VOLUME})\s+(?:Wheat|Pet|How|Black|Wall|Dall|Cranch)\.\s+({PAGE}){PIN_CITE}',
]

# Federal Supplement patterns
FEDERAL_SUPPLEMENT_PATTERNS = [
    # Federal Supplement: "123 F.Supp. 456", "123 F.Supp.2d 456", "123 F.Supp.3d 456"
    rf'({VOLUME})\s+F\.?\s*Supp\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # Federal Rules Decisions
    rf'({VOLUME})\s+F\.R\.D\.\s+({PAGE}){PIN_CITE}',
]

# Bankruptcy patterns
BANKRUPTCY_PATTERNS = [
    # Bankruptcy Reporter
    rf'({VOLUME})\s+B\.R\.\s+({PAGE}){PIN_CITE}',
]

# State reporter patterns (most common)
STATE_REPORTER_PATTERNS = [
    # Atlantic Reporter: "123 A.2d 456", "123 A.3d 456"
    rf'({VOLUME})\s+A\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # Pacific Reporter: "123 P.2d 456", "123 P.3d 456"
    rf'({VOLUME})\s+P\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # North Eastern Reporter
    rf'({VOLUME})\s+N\.E\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # North Western Reporter
    rf'({VOLUME})\s+N\.W\.(?:\s*(?:2d))?\s+({PAGE}){PIN_CITE}',
    # South Eastern Reporter
    rf'({VOLUME})\s+S\.E\.(?:\s*(?:2d))?\s+({PAGE}){PIN_CITE}',
    # South Western Reporter
    rf'({VOLUME})\s+S\.W\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # Southern Reporter
    rf'({VOLUME})\s+So\.(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # California Reporter
    rf'({VOLUME})\s+Cal\.?\s*(?:Rptr\.?|App\.?)(?:\s*(?:2d|3d|4th|5th))?\s+({PAGE}){PIN_CITE}',
    # New York Reporter (N.Y., N.Y.2d, N.Y.3d, A.D., A.D.2d, A.D.3d)
    rf'({VOLUME})\s+(?:N\.Y\.|A\.D\.)(?:\s*(?:2d|3d))?\s+({PAGE}){PIN_CITE}',
    # Illinois Reporter
    rf'({VOLUME})\s+Ill\.(?:\s*(?:2d|App\.))?\s+({PAGE}){PIN_CITE}',
    # Texas Reporter
    rf'({VOLUME})\s+(?:Tex\.|S\.W\.)\s+({PAGE}){PIN_CITE}',
]

# Other federal court patterns
OTHER_FEDERAL_PATTERNS = [
    # Court of Federal Claims
    rf'({VOLUME})\s+Fed\.?\s*Cl\.\s+({PAGE}){PIN_CITE}',
    # Veterans Appeals
    rf'({VOLUME})\s+Vet\.?\s*App\.\s+({PAGE}){PIN_CITE}',
]

# Compile all patterns into a master list
ALL_PATTERN_STRINGS = (
    SUPREME_COURT_PATTERNS +
    FEDERAL_REPORTER_PATTERNS +
    FEDERAL_SUPPLEMENT_PATTERNS +
    BANKRUPTCY_PATTERNS +
    STATE_REPORTER_PATTERNS +
    OTHER_FEDERAL_PATTERNS
)

# Compiled regex patterns with word boundaries
CITATION_PATTERNS: List[re.Pattern] = [
    re.compile(rf'\b{pattern}\b', re.IGNORECASE)
    for pattern in ALL_PATTERN_STRINGS
]

# Master pattern for finding any citation (for quick extraction)
# This is a more general pattern that catches most citations
GENERAL_CITATION_PATTERN = re.compile(
    r'\b(\d{1,4})\s+'
    r'('
    r'U\.S\.|'
    r'S\.?\s*Ct\.|'
    r'L\.?\s*Ed\.(?:\s*2d)?|'
    r'F\.(?:\s*(?:2d|3d|4th))?|'
    r'F\.?\s*Supp\.(?:\s*(?:2d|3d))?|'
    r'Fed\.?\s*Appx?\.?|F\.\s*App\'?x\.?|'
    r'F\.R\.D\.|'
    r'B\.R\.|'
    r'A\.(?:\s*(?:2d|3d))?|'
    r'P\.(?:\s*(?:2d|3d))?|'
    r'N\.E\.(?:\s*(?:2d|3d))?|'
    r'N\.W\.(?:\s*(?:2d))?|'
    r'S\.E\.(?:\s*(?:2d))?|'
    r'S\.W\.(?:\s*(?:2d|3d))?|'
    r'So\.(?:\s*(?:2d|3d))?|'
    r'Cal\.?\s*(?:Rptr\.?|App\.?)(?:\s*(?:2d|3d|4th|5th))?|'
    r'N\.Y\.(?:\s*(?:2d|3d))?|'
    r'A\.D\.(?:\s*(?:2d|3d))?|'
    r'Ill\.(?:\s*(?:2d|App\.?))?|'
    r'Wheat\.|Pet\.|How\.|Black\.|Wall\.|Dall\.|Cranch\.'
    r')'
    r'\s+(\d{1,5})'
    r'(?:,?\s*(\d{1,5}))?',  # Optional pin cite
    re.IGNORECASE
)


def extract_all_citations(text: str) -> List[str]:
    """Extract all legal citations from text.

    Args:
        text: The text to extract citations from

    Returns:
        List of citation strings found in the text
    """
    if not text:
        return []

    citations: Set[str] = set()

    # Use the general pattern for efficient extraction
    for match in GENERAL_CITATION_PATTERN.finditer(text):
        volume = match.group(1)
        reporter = match.group(2).strip()
        page = match.group(3)
        pin_cite = match.group(4)

        # Build the citation string
        citation = f"{volume} {reporter} {page}"
        if pin_cite:
            citation += f", {pin_cite}"

        # Normalize spacing in citation
        citation = normalize_citation(citation)
        citations.add(citation)

    # Also try each specific pattern for edge cases
    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            # Get the full matched citation
            full_match = match.group(0).strip()
            normalized = normalize_citation(full_match)
            citations.add(normalized)

    return sorted(list(citations))


def normalize_citation(citation: str) -> str:
    """Normalize a citation string to a standard format.

    Standardizes spacing, punctuation, and series designations.

    Args:
        citation: Raw citation string

    Returns:
        Normalized citation string
    """
    if not citation:
        return ""

    # Strip whitespace
    citation = citation.strip()

    # Normalize multiple spaces to single space
    citation = re.sub(r'\s+', ' ', citation)

    # Normalize U.S. Reports
    citation = re.sub(r'U\.\s*S\.', 'U.S.', citation)

    # Normalize Supreme Court Reporter
    citation = re.sub(r'S\.\s*Ct\.', 'S.Ct.', citation)

    # Normalize Lawyers' Edition
    citation = re.sub(r'L\.\s*Ed\.(\s*2d)?', lambda m: f'L.Ed.{m.group(1) or ""}', citation)

    # Normalize Federal Reporter
    citation = re.sub(r'F\.\s*(2d|3d|4th)', r'F.\1', citation)

    # Normalize Federal Supplement
    citation = re.sub(r'F\.\s*Supp\.(\s*(?:2d|3d))?', lambda m: f'F.Supp.{m.group(1).strip() if m.group(1) else ""}', citation)

    # Normalize Federal Appendix
    citation = re.sub(r'Fed\.?\s*Appx?\.?|F\.\s*App\'?x\.?', 'Fed.Appx.', citation, flags=re.IGNORECASE)

    # Normalize regional reporters
    citation = re.sub(r'A\.\s*(2d|3d)', r'A.\1', citation)
    citation = re.sub(r'P\.\s*(2d|3d)', r'P.\1', citation)
    citation = re.sub(r'N\.\s*E\.\s*(2d|3d)?', lambda m: f'N.E.{m.group(1) or ""}', citation)
    citation = re.sub(r'N\.\s*W\.\s*(2d)?', lambda m: f'N.W.{m.group(1) or ""}', citation)
    citation = re.sub(r'S\.\s*E\.\s*(2d)?', lambda m: f'S.E.{m.group(1) or ""}', citation)
    citation = re.sub(r'S\.\s*W\.\s*(2d|3d)?', lambda m: f'S.W.{m.group(1) or ""}', citation)
    citation = re.sub(r'So\.\s*(2d|3d)?', lambda m: f'So.{m.group(1) or ""}', citation)

    # Normalize California Reporter
    citation = re.sub(r'Cal\.\s*Rptr\.(\s*(?:2d|3d|4th|5th))?',
                      lambda m: f'Cal.Rptr.{m.group(1).strip() if m.group(1) else ""}', citation)

    # Normalize New York reporters
    citation = re.sub(r'N\.\s*Y\.\s*(2d|3d)?', lambda m: f'N.Y.{m.group(1) or ""}', citation)
    citation = re.sub(r'A\.\s*D\.\s*(2d|3d)?', lambda m: f'A.D.{m.group(1) or ""}', citation)

    # Remove trailing commas
    citation = citation.rstrip(',')

    return citation


def classify_citation(citation: str) -> str:
    """Classify a citation by its reporter type.

    Args:
        citation: Normalized citation string

    Returns:
        Classification string (e.g., "supreme_court", "federal_appeals", etc.)
    """
    citation_upper = citation.upper()

    if "U.S." in citation_upper and "F." not in citation_upper:
        return "supreme_court_us_reports"
    elif "S.CT." in citation_upper:
        return "supreme_court_sct"
    elif "L.ED." in citation_upper:
        return "supreme_court_led"
    elif any(x in citation_upper for x in ["WHEAT.", "PET.", "HOW.", "BLACK.", "WALL.", "DALL.", "CRANCH."]):
        return "supreme_court_early"
    elif "F.SUPP." in citation_upper:
        return "federal_district"
    elif "F." in citation_upper and "SUPP" not in citation_upper and "APPX" not in citation_upper:
        return "federal_appeals"
    elif "APPX" in citation_upper or "APP'X" in citation_upper:
        return "federal_appeals_unpublished"
    elif "B.R." in citation_upper:
        return "bankruptcy"
    elif any(x in citation_upper for x in ["A.", "P.", "N.E.", "N.W.", "S.E.", "S.W.", "SO."]):
        return "state_regional"
    elif any(x in citation_upper for x in ["CAL.", "N.Y.", "A.D.", "ILL.", "TEX."]):
        return "state_specific"
    else:
        return "unknown"


def extract_citation_components(citation: str) -> Tuple[str, str, str]:
    """Extract volume, reporter, and page from a citation.

    Args:
        citation: Normalized citation string

    Returns:
        Tuple of (volume, reporter, page) or ("", "", "") if parsing fails
    """
    match = re.match(r'(\d+)\s+(.+?)\s+(\d+)', citation)
    if match:
        return match.group(1), match.group(2).strip(), match.group(3)
    return "", "", ""


def is_valid_citation(citation: str) -> bool:
    """Check if a string looks like a valid legal citation.

    Args:
        citation: Citation string to validate

    Returns:
        True if the citation appears valid
    """
    volume, reporter, page = extract_citation_components(citation)

    if not volume or not reporter or not page:
        return False

    # Check volume is reasonable (1-999)
    try:
        vol = int(volume)
        if vol < 1 or vol > 999:
            return False
    except ValueError:
        return False

    # Check page is reasonable (1-99999)
    try:
        pg = int(page)
        if pg < 1 or pg > 99999:
            return False
    except ValueError:
        return False

    return True


# Parallel citation pattern (multiple citations separated by commas/semicolons)
PARALLEL_CITATION_SEPARATOR = re.compile(r'[;,]\s*(?=\d{1,4}\s+[A-Za-z])')


def split_parallel_citations(text: str) -> List[str]:
    """Split a string containing parallel citations into individual citations.

    Parallel citations like "347 U.S. 483, 74 S.Ct. 686, 98 L.Ed. 873"
    should be split into three separate citations.

    Args:
        text: String potentially containing parallel citations

    Returns:
        List of individual citation strings
    """
    parts = PARALLEL_CITATION_SEPARATOR.split(text)
    citations = []

    for part in parts:
        part = part.strip()
        if part and is_valid_citation(normalize_citation(part)):
            citations.append(normalize_citation(part))

    return citations


if __name__ == "__main__":
    # Test the patterns
    test_text = """
    In Brown v. Board of Education, 347 U.S. 483 (1954), the Court held...
    See also Plessy v. Ferguson, 163 U.S. 537, 16 S.Ct. 1138, 41 L.Ed. 256 (1896).
    The lower court in 123 F.3d 456 (9th Cir. 1997) agreed.
    Compare with 789 F.Supp.2d 123 (D.D.C. 2010) and 456 A.2d 789 (N.J. 2005).
    """

    citations = extract_all_citations(test_text)
    print("Extracted citations:")
    for cite in citations:
        classification = classify_citation(cite)
        print(f"  {cite} [{classification}]")
