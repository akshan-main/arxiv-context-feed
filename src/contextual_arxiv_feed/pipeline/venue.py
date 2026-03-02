"""Top venue detection from arXiv metadata.

Detects if a paper has been accepted to a top ML/AI venue
by parsing the comments and journal_ref fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Top ML/AI venues (conferences and journals)
TOP_VENUES = {
    # Top-tier conferences
    "neurips": ["neurips", "nips", "neural information processing"],
    "icml": ["icml", "international conference on machine learning"],
    "iclr": ["iclr", "international conference on learning representations"],
    "cvpr": ["cvpr", "computer vision and pattern recognition"],
    "iccv": ["iccv", "international conference on computer vision"],
    "eccv": ["eccv", "european conference on computer vision"],
    "acl": ["acl", "association for computational linguistics"],
    "emnlp": ["emnlp", "empirical methods in natural language"],
    "naacl": ["naacl", "north american chapter"],
    "aaai": ["aaai", "association for the advancement of artificial intelligence"],
    "ijcai": ["ijcai", "international joint conference on artificial intelligence"],
    "sigir": ["sigir", "special interest group on information retrieval"],
    "kdd": ["kdd", "knowledge discovery and data mining"],
    "www": ["www ", "world wide web conference"],  # space after www to avoid urls
    "coling": ["coling", "computational linguistics"],
    "interspeech": ["interspeech"],
    "icassp": ["icassp", "acoustics, speech and signal processing"],
    # Top-tier journals
    "nature": ["nature"],
    "science": ["science magazine", "science journal"],
    "jmlr": ["jmlr", "journal of machine learning research"],
    "tmlr": ["tmlr", "transactions on machine learning research"],
    "tacl": ["tacl", "transactions of the association for computational linguistics"],
    "pami": ["pami", "pattern analysis and machine intelligence"],
}

# Acceptance indicators
ACCEPTANCE_PATTERNS = [
    r"accepted\s+(?:at|to|by|for)\s+",
    r"to\s+appear\s+(?:at|in)\s+",
    r"published\s+(?:at|in)\s+",
    r"presented\s+at\s+",
    r"appearing\s+(?:at|in)\s+",
    r"camera.?ready",
    r"oral\s+presentation",
    r"spotlight",
    r"best\s+paper",
]


@dataclass
class VenueDetectionResult:
    """Result of venue detection."""

    detected: bool
    venue: str  # Normalized venue key (e.g., "neurips")
    venue_display: str  # Display name (e.g., "NeurIPS 2024")
    raw_match: str  # Original text that matched


def detect_top_venue(comments: str, journal_ref: str = "") -> VenueDetectionResult | None:
    """Detect if paper is accepted to a top venue.

    Checks comments and journal_ref fields for acceptance indicators
    combined with top venue mentions.

    Args:
        comments: arXiv comments field (e.g., "Accepted at NeurIPS 2024").
        journal_ref: arXiv journal_ref field.

    Returns:
        VenueDetectionResult if top venue detected, None otherwise.
    """
    # Combine and normalize text
    text = f"{comments} {journal_ref}".lower()

    if not text.strip():
        return None

    # Check for acceptance indicator
    has_acceptance = any(re.search(pattern, text, re.IGNORECASE) for pattern in ACCEPTANCE_PATTERNS)

    # Even without explicit acceptance, some patterns are strong signals
    # e.g., "NeurIPS 2024 camera-ready" or "ICML 2024 oral"
    strong_signal = any(
        term in text for term in ["camera ready", "camera-ready", "oral", "spotlight", "best paper"]
    )

    if not has_acceptance and not strong_signal:
        # Check if it's just a venue mention without acceptance (e.g., "submitted to")
        # Don't count these
        return None

    # Check for top venue mentions
    for venue_key, patterns in TOP_VENUES.items():
        for pattern in patterns:
            if pattern in text:
                # Extract year if present (e.g., "NeurIPS 2024" -> 2024)
                year_match = re.search(r"20\d{2}", text)
                year = year_match.group(0) if year_match else ""

                # Build display name
                display = venue_key.upper()
                if year:
                    display = f"{display} {year}"

                # Find the raw match for logging
                raw_match = _extract_raw_match(comments, journal_ref, pattern)

                return VenueDetectionResult(
                    detected=True,
                    venue=venue_key,
                    venue_display=display,
                    raw_match=raw_match,
                )

    return None


def _extract_raw_match(comments: str, journal_ref: str, pattern: str) -> str:
    """Extract the raw text that contains the pattern."""
    for text in [comments, journal_ref]:
        if pattern in text.lower():
            # Return a snippet around the match
            idx = text.lower().find(pattern)
            start = max(0, idx - 20)
            end = min(len(text), idx + len(pattern) + 30)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet
    return ""
