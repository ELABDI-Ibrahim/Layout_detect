"""
toc_detector.py
───────────────
Detects and extracts a Table of Contents (TOC) from OCR-extracted text.

Detection pipeline (6 layers):
  1. Pre-processing   – normalize markdown noise, split collapsed TOC lines
  2. Anchor detection – find an explicit TOC header ("Table of Contents", …)
  3. Line scoring     – score every line 0–10 for TOC-likeness
  4. Block detection  – find the densest contiguous window of high-scoring lines
  5. Boundary trim    – drop leading / trailing blank or low-score lines
  6. Structured parse – convert raw lines into a hierarchy of TocEntry objects

Handles:
  ✔ Clean TOC with dot leaders and page numbers
  ✔ Numbered-only TOC (no page numbers)
  ✔ Collapsed OCR lines (multiple entries merged onto one line)
  ✔ Markdown-wrapped OCR output (## headings, ** bold, etc.)
  ✔ Missing explicit TOC header
  ✔ Multi-language headers (EN / FR / ES / DE / AR / ZH)

Usage
-----
CLI:
    python toc_detector.py <ocr_output.txt>
    python toc_detector.py <ocr_output.txt> --raw   # also dump raw lines

Python API:
    from toc_detector import detect_toc
    result = detect_toc(text)
    # result["found"], result["entries"], result["confidence"]
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════
# 1.  COMPILED PATTERNS
# ══════════════════════════════════════════════════════════

# TOC header phrases (case-insensitive, multilingual)
_TOC_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"table\s+of\s+contents?"
    r"|contents?\s*page"
    r"|contents?"
    r"|sommaire"
    r"|table\s+des\s+mati[eè]res?"
    r"|índice"
    r"|inhaltsverzeichnis"
    r"|مقدمة"
    r"|目录"
    r")\s*$",
    re.IGNORECASE,
)

# Numbered entry prefix: "1.", "1.2", "1.2.3", "A.", "I.", "(1)", "1)"
_NUMBERED_PREFIX_RE = re.compile(
    r"^\s*(?:\(?\d+[\.\)]\d*[\.\d]*|[A-Z][\.\)]|[IVXLC]+[\.\)])\s+\S"
)

# Dot / dash leaders (4+ consecutive)
_DOT_LEADER_RE = re.compile(r"[.·•\-_]{4,}")

# Trailing page number — arabic or roman numerals
_TRAILING_PAGE_RE = re.compile(
    r"[.\s]{2,}(?:[ivxlcIVXLC]{1,6}|\d{1,4})\s*$"
)

# Inline page number near end (weaker signal)
_PAGE_NUM_RE = re.compile(r"(?:[ivxlcIVXLC]{1,6}|\d{1,4})\s*$")

# Lines that are clearly NOT TOC entries
_NOISE_RE = re.compile(
    r"^\s*$"
    r"|^\s*[=\-#*~]{3,}\s*$"
    r"|https?://"
    r"|\bimport\b.*\bfrom\b"
    r"|\bdef \w+\("
    r"|^\s*!\[.*\]\("
    r"|©\s*ISO"
    r"|All rights reserved"
)

# Markdown artefacts to strip
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*")
_MD_BOLD_RE    = re.compile(r"\*{1,2}(.+?)\*{1,2}")

# Collapsed TOC split boundary
_COLLAPSED_SPLIT_RE = re.compile(
    r"(?<=[ivxlcIVXLC\d])\s+(?=\d+[\. ]|[A-Z][a-z])"
)
_MULTI_DOT_RE = re.compile(r"[.·•]{3,}")


# ══════════════════════════════════════════════════════════
# 2.  PRE-PROCESSING
# ══════════════════════════════════════════════════════════

def _clean_line(line: str) -> str:
    """Strip markdown noise while preserving structural content."""
    line = _MD_HEADING_RE.sub("", line)
    line = _MD_BOLD_RE.sub(r"\1", line)
    return line


def _split_collapsed_line(line: str) -> list:
    """
    Split a single collapsed OCR line containing multiple TOC entries.

    Example input:
        "Foreword...iv Introduction...v 1 Scope...1 2 Normative references...1"
    Example output:
        ["Foreword...iv", "Introduction...v", "1 Scope...1", "2 Normative references...1"]
    """
    stripped = line.strip()
    if len(stripped) <= 80:
        return [line]
    if len(_MULTI_DOT_RE.findall(stripped)) < 2:
        return [line]
    parts = [p.strip() for p in _COLLAPSED_SPLIT_RE.split(stripped) if p.strip()]
    return parts if len(parts) > 1 else [line]


def _preprocess(text: str) -> list:
    """Return cleaned, expanded lines ready for scoring."""
    out = []
    for raw in text.splitlines():
        cleaned  = _clean_line(raw)
        expanded = _split_collapsed_line(cleaned)
        out.extend(expanded)
    return out


# ══════════════════════════════════════════════════════════
# 3.  LINE SCORING
# ══════════════════════════════════════════════════════════

def _score_line(line: str) -> float:
    """
    Return a 0–10 heuristic score for how TOC-like a line is.

    Weights:
      +4.0  numbered prefix
      +3.0  dot / dash leaders
      +2.0  trailing page number
      +1.0  inline page number (weaker)
      +0.5  ideal length (5–80 chars)
      -1.0  very long line (>120 chars)
      -1.5  prose sentence (many words, no structural markers)
    """
    if _NOISE_RE.search(line):
        return 0.0
    stripped = line.strip()
    if not stripped:
        return 0.0

    score = 0.0

    if _NUMBERED_PREFIX_RE.match(line):
        score += 4.0
    if _DOT_LEADER_RE.search(stripped):
        score += 3.0
    if _TRAILING_PAGE_RE.search(stripped):
        score += 2.0
    elif _PAGE_NUM_RE.search(stripped):
        score += 1.0

    length = len(stripped)
    if 5 <= length <= 80:
        score += 0.5
    elif length > 120:
        score -= 1.0

    if len(stripped.split()) > 12 and score < 1.0:
        score -= 1.5

    return max(score, 0.0)


# ══════════════════════════════════════════════════════════
# 4.  ANCHOR DETECTION
# ══════════════════════════════════════════════════════════

def _find_toc_anchor(lines: list) -> Optional[int]:
    """Return the index of the first explicit TOC header, or None."""
    for i, line in enumerate(lines):
        if _TOC_HEADER_RE.match(line.strip()):
            return i
    return None


# ══════════════════════════════════════════════════════════
# 5.  BLOCK DETECTION
# ══════════════════════════════════════════════════════════

def _find_best_block(
    lines: list,
    scores: list,
    start_hint: Optional[int] = None,
    window: int = 60,
) -> tuple:
    """
    Find the contiguous region most likely to be the TOC.

    With anchor: extend forward from it, stop at 2 consecutive weak lines.
    Without anchor: sliding window picking the densest scoring region.
    Returns (start_idx, end_idx) inclusive.
    """
    n = len(lines)
    if n == 0:
        return 0, 0

    if start_hint is not None:
        e = start_hint + 1
        consecutive_low = 0
        while e < n and consecutive_low < 2:
            if scores[e] >= 1.0:
                consecutive_low = 0
                e += 1
            else:
                consecutive_low += 1
                e += 1
        return start_hint, min(e - consecutive_low, n - 1)

    best_start, best_end, best_score = 0, 0, 0.0
    for i in range(n):
        j       = min(i + window, n)
        chunk   = scores[i:j]
        nonzero = [s for s in chunk if s > 0]
        if not nonzero:
            continue
        avg      = sum(nonzero) / len(nonzero)
        density  = len(nonzero) / (j - i)
        combined = avg * density
        if combined > best_score and density >= 0.4:
            best_score = combined
            best_start = i
            best_end   = j - 1

    return best_start, min(best_end, n - 1)


# ══════════════════════════════════════════════════════════
# 6.  BOUNDARY TRIMMING
# ══════════════════════════════════════════════════════════

def _trim_block(scores: list, start: int, end: int) -> tuple:
    """Remove leading/trailing lines with score < 1.0."""
    while start <= end and scores[start] < 1.0:
        start += 1
    while end >= start and scores[end] < 1.0:
        end -= 1
    return start, end


# ══════════════════════════════════════════════════════════
# 7.  STRUCTURED PARSING
# ══════════════════════════════════════════════════════════

@dataclass
class TocEntry:
    """A single parsed TOC entry."""
    level:    int
    number:   str
    title:    str
    page:     Optional[str] = None
    children: list = field(default_factory=list)

    def __str__(self) -> str:
        indent = "  " * self.level
        num    = f"{self.number} " if self.number else ""
        page   = f"  [{self.page}]" if self.page else ""
        return f"{indent}{num}{self.title}{page}"


_ENTRY_PARSE_RE = re.compile(
    r"^\s*"
    r"(?P<num>(?:\d+\.?)+|[A-Z][\.\)]|[IVXLC]+[\.\)])?"
    r"\s*"
    r"(?P<title>.+?)"
    r"(?:[.\s·•\-_]{3,})?"
    r"(?P<page>[ivxlcIVXLC]{1,6}|\d{1,4})?"
    r"\s*$",
)


def _indent_level(line: str) -> int:
    spaces = len(line) - len(line.lstrip())
    return 0 if spaces == 0 else 1 if spaces <= 4 else 2 if spaces <= 8 else 3


def _parse_entry(line: str) -> TocEntry:
    m     = _ENTRY_PARSE_RE.match(line)
    num   = ""
    title = line.strip()
    page  = None

    if m:
        num   = (m.group("num")   or "").strip()
        title = (m.group("title") or line).strip()
        page  = m.group("page")

    level = max(0, num.count(".")) if num else _indent_level(line)
    title = re.sub(r"[.·•\-_]{3,}.*$", "", title).strip()

    return TocEntry(level=level, number=num, title=title, page=page)


# ══════════════════════════════════════════════════════════
# 8.  PUBLIC API
# ══════════════════════════════════════════════════════════

def detect_toc(text: str) -> dict:
    """
    Detect and extract the Table of Contents from OCR-extracted text.

    Parameters
    ----------
    text : str
        Raw text from OCR (may contain markdown artefacts, collapsed lines, etc.)

    Returns
    -------
    dict:
        found       : bool
        header_line : str | None
        raw_lines   : list[str]
        entries     : list[TocEntry]
        confidence  : float  (0.0 – 1.0)
    """
    lines  = _preprocess(text)
    scores = [_score_line(l) for l in lines]

    anchor      = _find_toc_anchor(lines)
    header_line = lines[anchor].strip() if anchor is not None else None

    start, end  = _find_best_block(lines, scores, start_hint=anchor)
    start, end  = _trim_block(scores, start, end)

    if start > end:
        return {"found": False, "header_line": None, "raw_lines": [], "entries": [], "confidence": 0.0}

    raw_block = lines[start : end + 1]
    toc_lines = [l for l in raw_block if not _NOISE_RE.search(l)]

    # Remove the header line from the entries list
    if toc_lines and _TOC_HEADER_RE.match(toc_lines[0].strip()):
        toc_lines = toc_lines[1:]

    scored = [_score_line(l) for l in toc_lines if l.strip()]
    if scored:
        confidence = min(sum(scored) / len(scored) / 6.0, 1.0)
        if anchor is not None:
            confidence = min(confidence + 0.2, 1.0)
    else:
        confidence = 0.0

    entries = [_parse_entry(l) for l in toc_lines if l.strip()]

    return {
        "found":       bool(entries),
        "header_line": header_line,
        "raw_lines":   toc_lines,
        "entries":     entries,
        "confidence":  confidence,
    }


# ══════════════════════════════════════════════════════════
# 9.  PRETTY PRINTER
# ══════════════════════════════════════════════════════════

def print_toc(result: dict) -> None:
    if not result["found"]:
        print("❌  No Table of Contents detected.")
        return

    conf_pct = f"{result['confidence'] * 100:.0f}%"
    print(f"✅  TOC detected  (confidence: {conf_pct})")
    print("─" * 60)

    if result["header_line"]:
        print(f"  Header : {result['header_line']}")
        print()

    print("  STRUCTURED TOC:")
    print()
    for entry in result["entries"]:
        print(f"  {entry}")

    print()
    print("─" * 60)
    print(f"  {len(result['entries'])} entries found")


# ══════════════════════════════════════════════════════════
# 10.  CLI
# ══════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python toc_detector.py <ocr_text_file> [--raw]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text   = path.read_text(encoding="utf-8", errors="replace")
    result = detect_toc(text)
    print_toc(result)

    if "--raw" in sys.argv:
        print("\n  RAW LINES:")
        for line in result["raw_lines"]:
            print(f"    {repr(line)}")


if __name__ == "__main__":
    main()