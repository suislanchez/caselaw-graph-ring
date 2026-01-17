"""Text preprocessing utilities for legal documents."""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# Common legal boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    r"^\s*Page\s+\d+\s*$",  # Page numbers
    r"^\s*\d+\s*$",  # Standalone numbers
    r"^\*+\d+\s*",  # Star page markers (*123)
    r"\[\*\*?\d+\]",  # Page references [*123] or [**123]
    r"^\s*-+\s*$",  # Horizontal rules
    r"Syllabus\s*$",  # Syllabus headers
    r"^Opinion of the Court\.?\s*$",
    r"^Opinion of\s+\w+,\s+J\.?\s*$",
    r"^Dissenting Opinion\.?\s*$",
    r"^Concurring Opinion\.?\s*$",
]

# Compile patterns for efficiency
BOILERPLATE_RE = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

# Citation patterns (for reference, main extraction in citations module)
US_CITE_PATTERN = re.compile(r'\d+\s+U\.?\s*S\.?\s+\d+')
FEDERAL_CITE_PATTERN = re.compile(r'\d+\s+F\.?\s*(?:2d|3d|4th)?\s+\d+')


def clean_legal_text(text: str) -> str:
    """Clean legal document text.

    Removes:
    - Page numbers and markers
    - Excessive whitespace
    - Common boilerplate headers

    Args:
        text: Raw legal text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove boilerplate patterns
    for pattern in BOILERPLATE_RE:
        text = pattern.sub("", text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces
    text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def sentence_segment(text: str) -> List[str]:
    """Split text into sentences.

    Handles legal-specific patterns like:
    - Citations (e.g., "123 U.S. 456")
    - Abbreviations (e.g., "v.", "J.", "Corp.")

    Args:
        text: Text to segment

    Returns:
        List of sentences
    """
    if not text:
        return []

    # Protect common abbreviations from splitting
    protected = {
        'v.': 'V_VERSUS_V',
        'J.': 'J_JUSTICE_J',
        'Jr.': 'JR_JUNIOR_JR',
        'Sr.': 'SR_SENIOR_SR',
        'Inc.': 'INC_CORP_INC',
        'Corp.': 'CORP_CORP_CORP',
        'Ltd.': 'LTD_CORP_LTD',
        'Co.': 'CO_COMPANY_CO',
        'No.': 'NO_NUMBER_NO',
        'Mr.': 'MR_MISTER_MR',
        'Mrs.': 'MRS_MISSUS_MRS',
        'Ms.': 'MS_MISS_MS',
        'Dr.': 'DR_DOCTOR_DR',
        'U.S.': 'US_UNITED_US',
        'S.Ct.': 'SCT_COURT_SCT',
        'L.Ed.': 'LED_LAWYER_LED',
        'F.2d': 'F2D_FEDERAL_F2D',
        'F.3d': 'F3D_FEDERAL_F3D',
        'F.4th': 'F4TH_FEDERAL_F4TH',
        'et al.': 'ETAL_ETAL_ETAL',
        'e.g.': 'EG_EXAMPLE_EG',
        'i.e.': 'IE_THAT_IE',
    }

    # Replace protected patterns
    for abbrev, placeholder in protected.items():
        text = text.replace(abbrev, placeholder)

    # Split on sentence boundaries
    # Match: period/question/exclamation + space + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore protected patterns
    result = []
    for sent in sentences:
        for abbrev, placeholder in protected.items():
            sent = sent.replace(placeholder, abbrev)
        sent = sent.strip()
        if sent:
            result.append(sent)

    return result


def count_tokens(text: str, tokenizer_name: str = "gpt2") -> int:
    """Count tokens in text.

    Args:
        text: Text to tokenize
        tokenizer_name: HuggingFace tokenizer name

    Returns:
        Token count
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return len(tokenizer.encode(text))
    except ImportError:
        # Fallback: rough word-based estimate
        return len(text.split()) * 1.3  # ~1.3 tokens per word for legal text


def count_tokens_batch(texts: List[str], tokenizer_name: str = "gpt2") -> List[int]:
    """Count tokens for multiple texts efficiently.

    Args:
        texts: List of texts
        tokenizer_name: HuggingFace tokenizer name

    Returns:
        List of token counts
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return [len(tokenizer.encode(t)) for t in texts]
    except ImportError:
        return [int(len(t.split()) * 1.3) for t in texts]


def extract_case_metadata(text: str) -> Dict[str, Optional[str]]:
    """Extract metadata from case text.

    Args:
        text: Full case text

    Returns:
        Dict with extracted metadata
    """
    metadata = {
        "docket_number": None,
        "argued_date": None,
        "decided_date": None,
        "judges": [],
    }

    # Docket number patterns
    docket_match = re.search(r'No\.\s*([\d\-]+)', text[:2000])
    if docket_match:
        metadata["docket_number"] = docket_match.group(1)

    # Date patterns
    argued_match = re.search(r'Argued\s+(\w+\s+\d+,?\s+\d{4})', text[:2000])
    if argued_match:
        metadata["argued_date"] = argued_match.group(1)

    decided_match = re.search(r'Decided\s+(\w+\s+\d+,?\s+\d{4})', text[:2000])
    if decided_match:
        metadata["decided_date"] = decided_match.group(1)

    # Judge names (simple extraction)
    judge_pattern = re.compile(r'(?:Justice|Judge|Chief Justice)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)')
    judges = judge_pattern.findall(text[:5000])
    metadata["judges"] = list(set(judges))

    return metadata


def compute_text_stats(texts: List[str]) -> Dict:
    """Compute statistics over a collection of texts.

    Args:
        texts: List of texts

    Returns:
        Dict with statistics
    """
    if not texts:
        return {}

    char_lengths = [len(t) for t in texts]
    word_lengths = [len(t.split()) for t in texts]

    # Token counts (sample for efficiency)
    sample_size = min(100, len(texts))
    import random
    sample = random.sample(texts, sample_size)
    token_counts = count_tokens_batch(sample)
    avg_tokens = sum(token_counts) / len(token_counts)

    return {
        "count": len(texts),
        "avg_chars": sum(char_lengths) / len(char_lengths),
        "min_chars": min(char_lengths),
        "max_chars": max(char_lengths),
        "avg_words": sum(word_lengths) / len(word_lengths),
        "min_words": min(word_lengths),
        "max_words": max(word_lengths),
        "avg_tokens_estimated": avg_tokens,
        "total_words": sum(word_lengths),
    }


def get_text_stats(text: str) -> Dict:
    """Get statistics for a single text.

    Args:
        text: Text to analyze

    Returns:
        Dict with token count, sentence count, word count, char count
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "token_count_estimated": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
        }

    words = text.split()
    sentences = sentence_segment(text)

    # Estimate token count (more accurate with actual tokenizer)
    token_count = int(len(words) * 1.3)  # Rough estimate for legal text

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "token_count_estimated": token_count,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0.0,
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0.0,
    }


def truncate_to_tokens(text: str, max_tokens: int, tokenizer_name: str = "gpt2") -> str:
    """Truncate text to maximum token count.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        tokenizer_name: Tokenizer to use

    Returns:
        Truncated text
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    except ImportError:
        # Fallback: word-based truncation
        words = text.split()
        estimated_words = int(max_tokens / 1.3)
        return ' '.join(words[:estimated_words])


# CLI interface
if __name__ == "__main__":
    # Test preprocessing
    sample_text = """
    *123 SUPREME COURT OF THE UNITED STATES

    No. 22-451

    SMITH, PETITIONER v. JONES, RESPONDENT

    ON WRIT OF CERTIORARI TO THE UNITED STATES
    COURT OF APPEALS FOR THE NINTH CIRCUIT

    [January 15, 2024]

    Opinion of the Court.

    JUSTICE ROBERTS delivered the opinion of the Court.

    This case concerns the interpretation of 42 U.S.C. 1983.
    In Brown v. Board of Education, 347 U.S. 483 (1954), we held
    that separate educational facilities are inherently unequal.
    See also Roe v. Wade, 410 U.S. 113 (1973).

    *124 The petitioner argues that...

    Page 2

    The respondent contends that...
    """

    print("Original text length:", len(sample_text))

    cleaned = clean_legal_text(sample_text)
    print("\nCleaned text length:", len(cleaned))
    print("\nCleaned text:")
    print(cleaned[:500])

    sentences = sentence_segment(cleaned)
    print(f"\nFound {len(sentences)} sentences")
    for i, sent in enumerate(sentences[:3]):
        print(f"  {i+1}. {sent[:80]}...")

    metadata = extract_case_metadata(sample_text)
    print(f"\nExtracted metadata: {metadata}")

    stats = get_text_stats(cleaned)
    print(f"\nText statistics: {stats}")
