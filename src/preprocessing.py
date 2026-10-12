"""Text cleaning that keeps technical terms (C++, C#, Node.js, .NET, scikit-learn ...) intact."""

from __future__ import annotations

import re
import unicodedata

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Candidate phone numbers: digits with separators. Validated in _phone_replacer.
PHONE_RE = re.compile(r"\+?\d[\d ().-]{8,}\d")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")
BULLET_CHARS = "•●▪■◦‣∙·➢➤►–—"


def _phone_replacer(match: re.Match[str]) -> str:
    """Remove phone numbers but keep date ranges such as '2019 2022 2023'."""
    chunk = match.group(0)
    groups = re.findall(r"\d+", chunk)
    digit_count = sum(len(g) for g in groups)
    looks_like_years = all(len(g) == 4 and g.startswith(("19", "20")) for g in groups)
    if 10 <= digit_count <= 13 and not looks_like_years:
        return " "
    return chunk


def clean_text(text: str, remove_contact_info: bool = True) -> str:
    """Normalise raw extracted text.

    What it does:
      * Unicode normalisation (NFKC) and removal of control characters
      * fixes words split by a hyphen at a line end
      * turns bullet glyphs into '-'
      * optionally removes e-mails, URLs and phone numbers (personal data that
        must not influence ranking)
      * collapses repeated whitespace

    What it deliberately does NOT do: strip punctuation or stop-words, lowercase,
    or remove digits - so 'C++', 'C#', 'Node.js', '.NET', 'scikit-learn' survive.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HYPHEN_BREAK_RE.sub(r"\1", text)
    for glyph in BULLET_CHARS:
        text = text.replace(glyph, "-")
    if remove_contact_info:
        text = EMAIL_RE.sub(" ", text)
        text = URL_RE.sub(" ", text)
        text = PHONE_RE.sub(_phone_replacer, text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_words: int = 150, overlap: int = 30) -> list[str]:
    """Split long text into overlapping word windows.

    Needed because sentence-transformer models only read ~256 word-pieces; long
    resumes would otherwise be silently truncated.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and smaller than max_words")
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks
