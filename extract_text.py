"""
extract_text.py

Stage 1: Extract raw text from EPUB/PDF novel files into clean per-chapter
.txt files, ready for the next stage of the pipeline.

Usage:
    1. Put all .epub / .pdf novel files in the input/ folder
       (name files so they sort in volume/chapter order, e.g.
        01_volume1.epub, 02_volume2.epub, etc.)
    2. Run:
       python extract_text.py

Output:
    text/chapter_0001.txt, text/chapter_0002.txt, etc.
    text/manifest.json  -> order & source metadata for each chapter
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("text")
MIN_CHAPTER_CHARS = 400  # drop fragments that are too short (title pages, etc.)

# Only the first & last N lines of each page are treated as candidates for
# header/footer/page-number noise (that's where this kind of noise usually
# shows up). For short pages, this zone is automatically shrunk (see
# _edge_zone_size) so it never swallows most of a page's content.
BOILERPLATE_EDGE_LINES = 3
# Minimum number of pages before repeated-pattern detection kicks in
# (in short books, "repeated" isn't statistically meaningful).
BOILERPLATE_MIN_PAGES = 8
# If a line (after normalization) appears on >= this fraction of pages,
# it's treated as boilerplate and stripped. Set fairly high (more than half
# of all pages) because genuine headers/footers/page numbers usually appear
# on ALMOST EVERY page -- a high threshold reduces the risk of accidentally
# stripping a story sentence that happens to look similar across pages.
BOILERPLATE_FREQ_THRESHOLD = 0.5


def _edge_zone_size(total_lines: int) -> int:
    """Size of the edge zone to inspect, scaled so it never exceeds
    ~25% of a page's total lines (important for short pages)."""
    return max(1, min(BOILERPLATE_EDGE_LINES, total_lines // 4))


def _normalize_line(line: str) -> str:
    """Normalize a line for comparison: collapse whitespace, replace digits with '#'."""
    s = line.strip()
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def strip_repeating_boilerplate(pages: list[str]) -> list[str]:
    """
    Detect & strip lines that repeat identically across many pages (after
    digit normalization) -- the classic pattern for page numbers, running
    headers/footers (book title, author name, etc.). Only checks a few
    lines at the top/bottom edge of each page so it never touches story
    content in the middle of a page. Lines that look like chapter markers
    (Chapter/Prologue/Epilogue) are deliberately excluded so
    split_into_chapters() can still detect them.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return pages

    chapter_marker_re = re.compile(r"\b(chapter|prologue|epilogue)\b")
    page_lines = [p.split("\n") for p in pages]
    candidate_counter = Counter()

    for lines in page_lines:
        edge_n = _edge_zone_size(len(lines))
        edge = lines[:edge_n] + lines[-edge_n:]
        seen_this_page = set()
        for line in edge:
            norm = _normalize_line(line)
            if not norm or len(norm) > 80:
                continue
            if chapter_marker_re.search(norm):
                continue
            if norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            candidate_counter[norm] += 1

    total_pages = len(pages)
    boilerplate = {
        norm for norm, count in candidate_counter.items()
        if count / total_pages >= BOILERPLATE_FREQ_THRESHOLD
    }

    if not boilerplate:
        return pages

    sample = ", ".join(list(boilerplate)[:3])
    print(f"[INFO] Detected & removed {len(boilerplate)} repeating noise pattern(s) (example: {sample})")

    cleaned_pages = []
    for lines in page_lines:
        n = len(lines)
        edge_n = _edge_zone_size(n)
        kept = [
            line for idx, line in enumerate(lines)
            if not (
                (idx < edge_n or idx >= n - edge_n)
                and _normalize_line(line) in boilerplate
            )
        ]
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


def clean_text(raw: str) -> str:
    """Clean up excess whitespace, broken line-end hyphenation, etc."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # rejoin words split by a hyphen at line end (common in PDFs)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse repeated spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_chapters(text: str) -> list[str]:
    """
    Heuristic chapter splitter: look for common patterns like
    'Chapter 1', 'Chapter One', 'Prologue', 'Epilogue', roman numerals, etc.
    Falls back to splitting into ~3000-word blocks if no clear pattern is found.
    """
    pattern = re.compile(
        r"(?:^|\n)\s*((?:Chapter|CHAPTER|Prologue|PROLOGUE|Epilogue|EPILOGUE)\b[^\n]{0,60})\n",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))

    if len(matches) >= 2:
        chapters = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapters.append(text[start:end].strip())
        return [c for c in chapters if len(c) >= MIN_CHAPTER_CHARS]

    # Fallback: no clear chapter markers -> split into word blocks
    words = text.split()
    chunk_size = 3000
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return [c for c in chunks if len(c) >= MIN_CHAPTER_CHARS]


def extract_epub(path: Path) -> list[str]:
    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            text = clean_text(text)
            if len(text) >= MIN_CHAPTER_CHARS:
                chapters.append(text)
    return chapters


def extract_pdf(path: Path) -> list[str]:
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    pages = strip_repeating_boilerplate(pages)
    joined = clean_text("\n".join(pages))
    return split_into_chapters(joined)


def main():
    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"No files found in {INPUT_DIR}/. Put your .epub or .pdf files there first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = []
    chapter_idx = 1

    source_files = sorted(
        [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".epub", ".pdf")]
    )

    if not source_files:
        print("No .epub/.pdf files found in the input/ folder.")
        sys.exit(1)

    for src in source_files:
        print(f"Processing {src.name} ...")
        if src.suffix.lower() == ".epub":
            chapters = extract_epub(src)
        else:
            chapters = extract_pdf(src)

        for chap_text in chapters:
            out_name = f"chapter_{chapter_idx:04d}.txt"
            out_path = OUTPUT_DIR / out_name
            out_path.write_text(chap_text, encoding="utf-8")
            manifest.append(
                {
                    "chapter_file": out_name,
                    "source_file": src.name,
                    "char_count": len(chap_text),
                }
            )
            chapter_idx += 1

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(manifest)} chapters extracted to {OUTPUT_DIR}/")
    print("Spot-check a few .txt files manually to confirm the extraction is clean")
    print("(especially for PDFs, since headers/footers/page numbers sometimes slip through).")


if __name__ == "__main__":
    main()
