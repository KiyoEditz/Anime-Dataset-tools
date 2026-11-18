"""
extract_text.py

Stage 1 of the Elaina Dataset Pipeline:
Extracts raw text from EPUB/PDF novel files with a two-step confirmation workflow:

Workflow:
    Step 1 (Unpack & Triage):
        python extract_text.py --step unpack   (or: python extract_text.py --step 1)
        - Reads EPUBs in input/ in chronological order.
        - Parses OPF reading spine and Table of Contents (TOC).
        - Extracts candidate story chapters into confirm_xhtml/Volume_XX/
        - Automatically isolates junk files (covers, TOC, copyright, ads, image-only pages)
          into confirm_xhtml/_excluded_junk/Volume_XX/
        - Generates confirm_xhtml/SUMMARY.md for user audit.

    User Confirmation 1:
        - Check confirm_xhtml/. Delete any junk that slipped through, or move back
          any file from _excluded_junk/ if desired.

    Step 2 (Convert Confirmed XHTML -> Clean Text):
        python extract_text.py --step text     (or: python extract_text.py --step 2)
        - Converts confirmed .xhtml files into clean plain text.
        - Generates text/chapter_0001.txt, text/chapter_0002.txt, ...
        - Adds title & volume metadata header to every chapter text.
        - Generates text/manifest.json for full traceability.

    User Confirmation 2:
        - Check text/ to verify that file names and chapter contents match.
        - Proceed to: python persona_extractor.py
"""

import argparse
import html
import json
import os
import posixpath
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

LANG = os.environ.get("DATASET_LANG", "EN").upper()
INPUT_DIR = Path("input")
CONFIRM_DIR = Path(f"confirm_xhtml-{LANG}")
JUNK_DIR = CONFIRM_DIR / "_excluded_junk"
TEXT_DIR = Path(f"text-{LANG}")


def set_language(lang: str):
    """Dynamically set the active language and target directories."""
    global LANG, CONFIRM_DIR, JUNK_DIR, TEXT_DIR
    LANG = lang.upper()
    CONFIRM_DIR = Path(f"confirm_xhtml-{LANG}")
    JUNK_DIR = CONFIRM_DIR / "_excluded_junk"
    TEXT_DIR = Path(f"text-{LANG}")


# Patterns identifying non-story pages (frontmatter, backmatter, afterwords, ads)
# Supports English and Japanese light novel structure tokens
JUNK_NAME_PATTERN = re.compile(
    r"(cover|insert|titlepage|copyright|caution|toc|nav|newsletter|advertisement|colophon|contents|"
    r"afterword|appendix|postscript|credit|author|profile|fanletter|p-99[0-9]|p-899|"
    r"表紙|口絵|扉|目次|奥付|あとがき|後書き|解説|登場人物|人物紹介|広告)",
    re.IGNORECASE,
)

# Canonical chapter prefixes in TOC titles that are always real story chapters
STORY_TOC_PREFIX = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇0-9０-９]+[章話]|Chapter\s*\d+|Prologue|Epilogue|Interlude|"
    r"プロローグ|エピローグ|幕間|投書話)",
    re.IGNORECASE,
)

# Trailing bonus metadata / author commentary attached to short stories (e.g. Vol 15 SS collection)
TRAILING_META_PATTERN = re.compile(
    r"(?:^|\n+)\s*【(?:初出情報|著者コメント)】[\s\S]*$"
)

# Full-width to half-width digit translation table for natural volume sorting
_FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


# Minimum text length (in characters) to be considered a real story chapter.
# Single illustration/image pages have 0 characters once <head>/<title> are stripped.
MIN_CHAPTER_CHARS = 200
MIN_CHAPTER_CHARS_JP = 30

# PDF boilerplate detection settings
BOILERPLATE_EDGE_LINES = 3
BOILERPLATE_MIN_PAGES = 8
BOILERPLATE_FREQ_THRESHOLD = 0.5


def sanitize_filename(name: str) -> str:
    """Sanitize string for Windows filesystem."""
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = re.sub(r"[\s\u3000_]+", "_", s)
    return s.strip("._")


def natural_epub_sort_key(epub_path: Path):
    """
    Sort EPUB files chronologically by series group and numeric volume number,
    normalizing full-width Japanese digits (０-９ -> 0-9).
    """
    norm_name = epub_path.stem.translate(_FW_DIGITS)
    if "ガクエン" in norm_name or "学園" in norm_name:
        series_group = 1
    elif "リリエールと祈りの国" in norm_name:
        series_group = 2
    elif "イノリノクニノリリエール" in norm_name or "祈りの国のリリエール" in norm_name:
        series_group = 3
    elif "マジョノタビタビ" in norm_name or "魔女の旅々" in norm_name or "Wandering Witch" in norm_name:
        series_group = 0
    else:
        series_group = 4

    nums = [int(n) for n in re.findall(r"\d+", norm_name)]
    vol_num = nums[0] if nums else 0
    return (series_group, vol_num, norm_name.lower())


class _StdlibHTMLTextParser(HTMLParser):
    """Fallback HTML-to-text parser when BeautifulSoup (bs4) is not installed."""

    SKIP_TAGS = {"head", "title", "script", "style", "meta", "link", "rt", "rp", "svg", "nav"}
    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "div", "li"}

    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.paragraphs = []
        self.current = []

    def _flush(self):
        if self.current:
            txt = "".join(self.current).strip()
            if txt:
                self.paragraphs.append(txt)
            self.current = []

    def handle_starttag(self, tag, attrs):
        tag_l = tag.lower()
        if tag_l in self.SKIP_TAGS:
            self.skip_depth += 1
        elif self.skip_depth == 0:
            if tag_l in self.BLOCK_TAGS:
                self._flush()
            elif tag_l == "br":
                self.current.append("\n")

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if tag_l in self.SKIP_TAGS:
            if self.skip_depth > 0:
                self.skip_depth -= 1
        elif self.skip_depth == 0 and tag_l in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.current.append(data)

    def get_paragraphs(self) -> list[str]:
        self._flush()
        return self.paragraphs


class _StdlibAnchorParser(HTMLParser):
    """Fallback HTML anchor (<a>) extractor for TOC documents when bs4 is not installed."""

    def __init__(self):
        super().__init__()
        self.links = []
        self._in_a = False
        self._href = ""
        self._text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attr_dict = dict(attrs)
            self._href = attr_dict.get("href", "") or ""
            self._in_a = True
            self._text_parts = []
        elif self._in_a and tag.lower() in ("rt", "rp"):
            pass

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            title = "".join(self._text_parts).strip()
            self.links.append((self._href, title))
            self._in_a = False
            self._href = ""
            self._text_parts = []

    def handle_data(self, data):
        if self._in_a:
            self._text_parts.append(data)


def find_opf_and_dir(z: zipfile.ZipFile):
    """Find the root .opf package file and its directory inside the EPUB archive."""
    container_xml = z.read("META-INF/container.xml")
    root = ET.fromstring(container_xml)
    rootfile = root.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
    if rootfile is None:
        raise ValueError("Invalid EPUB: no rootfile specified in META-INF/container.xml")
    opf_path = rootfile.attrib["full-path"]
    opf_dir = posixpath.dirname(opf_path)
    return opf_path, opf_dir


def _extract_html_links(raw_bytes: bytes) -> list[tuple[str, str]]:
    """Extract (href, title) pairs from an HTML/XHTML TOC document."""
    if BeautifulSoup is not None:
        toc_soup = BeautifulSoup(raw_bytes, "html.parser")
        for elem in toc_soup(["rt", "rp"]):
            elem.decompose()
        return [(a.get("href", ""), a.get_text().strip()) for a in toc_soup.find_all("a")]
    parser = _StdlibAnchorParser()
    parser.feed(raw_bytes.decode("utf-8", errors="replace"))
    return parser.links


def parse_epub_structure(epub_path: Path):
    """
    Parses an EPUB's true spine (reading order), manifest, and Table of Contents (TOC).
    Supports both EPUB3 navigation documents (nav.xhtml / navigation-documents.xhtml / p-toc.xhtml)
    and EPUB2 NCX files (toc.ncx).
    Returns:
        spine_items: list of archive href paths in exact reading order
        manifest: dict of manifest items
        toc_map: dict mapping archive href -> chapter title from TOC
    """
    with zipfile.ZipFile(epub_path, "r") as z:
        opf_path, opf_dir = find_opf_and_dir(z)
        opf_root = ET.fromstring(z.read(opf_path))

        manifest = {}
        for item in opf_root.findall(".//{http://www.idpf.org/2007/opf}item"):
            item_id = item.attrib.get("id", "")
            href = item.attrib.get("href", "")
            full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
            manifest[item_id] = {
                "href": full_href,
                "media_type": item.attrib.get("media-type", ""),
                "properties": item.attrib.get("properties", ""),
            }

        spine = []
        for itemref in opf_root.findall(".//{http://www.idpf.org/2007/opf}itemref"):
            idref = itemref.attrib.get("idref")
            if idref in manifest:
                spine.append(manifest[idref]["href"])

        toc_map = {}
        archive_names = set(z.namelist())

        for item_id, info in manifest.items():
            href = info["href"]
            if href not in archive_names:
                continue
            href_lower = href.lower()
            media_type = info["media_type"].lower()
            props = info["properties"].lower()

            # 1. EPUB2 NCX Table of Contents (toc.ncx)
            if media_type == "application/x-dtbncx+xml" or href_lower.endswith(".ncx"):
                try:
                    ncx_root = ET.fromstring(z.read(href))
                    for nav_point in ncx_root.findall(".//{http://www.daisy.org/z3986/2005/ncx/}navPoint"):
                        title = nav_point.findtext(
                            ".//{http://www.daisy.org/z3986/2005/ncx/}text", ""
                        ).strip()
                        content_el = nav_point.find(".//{http://www.daisy.org/z3986/2005/ncx/}content")
                        if content_el is not None:
                            target_raw = content_el.attrib.get("src", "").split("#")[0]
                            if target_raw and title:
                                target_full = posixpath.normpath(
                                    posixpath.join(posixpath.dirname(href), target_raw)
                                )
                                if target_full not in toc_map:
                                    toc_map[target_full] = title
                except ET.ParseError:
                    pass

            # 2. EPUB3 Navigation Documents & Inline HTML TOCs
            elif (
                "nav" in props
                or "toc" in item_id.lower()
                or href_lower.endswith(
                    ("toc.xhtml", "toc.html", "nav.xhtml", "navigation-documents.xhtml")
                )
            ):
                for raw_href, title in _extract_html_links(z.read(href)):
                    target_raw = raw_href.split("#")[0]
                    if not target_raw or not title:
                        continue
                    target_full = posixpath.normpath(
                        posixpath.join(posixpath.dirname(href), target_raw)
                    )
                    if target_full not in toc_map:
                        toc_map[target_full] = title

        return spine, manifest, toc_map


def clean_html_to_text(raw_html: bytes | str) -> str:
    """
    Converts novel XHTML/HTML to clean, formatted text:
    - Strips head/title/scripts/styles/meta/svg/nav
    - Strips ruby/furigana annotations (<rt>, <rp>) to avoid duplicated kanji reading
    - Strips trailing store-bonus metadata / author comments (【初出情報】 / 【著者コメント】)
    - Preserves paragraphs and dialogues
    - Normalizes unicode typography, whitespace, and dashes
    """
    if isinstance(raw_html, bytes):
        html_str = raw_html.decode("utf-8", errors="replace")
    else:
        html_str = raw_html

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_str, "html.parser")
        for elem in soup(["head", "title", "script", "style", "meta", "link", "rt", "rp", "svg", "nav"]):
            elem.decompose()

        paragraphs = []
        for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "div"]):
            if not p.find(["p", "h1", "h2", "h3", "blockquote"]):
                txt = p.get_text(strip=True)
                if txt:
                    paragraphs.append(txt)

        if not paragraphs:
            txt = soup.get_text(separator="\n", strip=True)
            paragraphs = [line.strip() for line in txt.splitlines() if line.strip()]
    else:
        parser = _StdlibHTMLTextParser()
        parser.feed(html_str)
        paragraphs = parser.get_paragraphs()

    joined = "\n\n".join(paragraphs)
    joined = joined.replace("\ufeff", "")
    joined = joined.replace("\r\n", "\n").replace("\r", "\n")
    joined = joined.replace("\xa0", " ").replace("\u3000", " ")
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = TRAILING_META_PATTERN.sub("", joined)
    return joined.strip()


def is_junk_page(base_name: str, toc_title: str, text_content: str) -> bool:
    """
    Determines whether an EPUB spine page is non-story junk (frontmatter, backmatter,
    illustrations, copyright/caution notices, inline TOC, afterword, author bio,
    fan-letter QR page, or colophon).
    """
    text_len = len(text_content)
    min_chars = MIN_CHAPTER_CHARS_JP if LANG == "JP" else MIN_CHAPTER_CHARS

    # 1. Filename or TOC title matches known junk pattern
    if JUNK_NAME_PATTERN.search(base_name):
        return True
    if toc_title and not STORY_TOC_PREFIX.search(toc_title) and JUNK_NAME_PATTERN.search(toc_title):
        return True

    # 2. Empty or below language-specific minimum character threshold
    if text_len < min_chars:
        return True

    lines = [ln.strip() for ln in text_content.splitlines() if ln.strip()]
    first_line = lines[0] if lines else ""
    head_sample = "\n".join(lines[:3])

    # 3. Digital edition caution / copyright page
    if (
        "本書に掲載されているコンテンツの著作権" in text_content
        or "無断で複製" in text_content
        or ("本電子書籍は" in text_content and "著作権" in text_content)
        or ("All rights reserved" in text_content and text_len < 3500)
    ):
        return True

    # 4. Cover / illustration credits page
    if text_len < 300 and ("カバー・口絵" in text_content or "本文イラスト" in text_content):
        return True

    # 5. Inline Table of Contents page (e.g. CONTENTS / 目次 without toc filename)
    if re.match(r"^(?:CONTENTS|TABLE OF CONTENTS|目次)\b", first_line, re.IGNORECASE):
        return True

    # 6. Afterword (あとがき / 後書き / AFTERWORD or untitled author afterword)
    if re.search(r"^(?:あとがき|後書き|AFTERWORD)\b", head_sample, re.IGNORECASE | re.MULTILINE):
        return True
    if re.search(
        r"^(?:初めまして|はじめまして|お久しぶりです|こんにちは|こんばんは)[^。\n]{0,40}白石\s*定規",
        first_line,
    ):
        return True

    # 7. Author / Illustrator profile page
    if text_len < 600 and re.match(r"^(?:AUTHOR|著者)\b", first_line, re.IGNORECASE):
        return True

    # 8. Fan-letter / questionnaire / QR code page
    if text_len < 600 and (
        "ファンレター、作品の感想をお待ちしています" in text_content
        or "本書に関するご意見・ご感想は" in text_content
    ):
        return True

    # 9. Colophon / publisher info page
    if text_len < 800 and "発行人" in text_content and "発行所" in text_content:
        return True

    # 10. Short titlepage stub
    if text_len < 60 and re.search(r"白石\s*定規", text_content):
        return True

    return False


def step_unpack_epubs():
    """
    Step 1: Unpacks EPUBs from input/ into confirm_xhtml/.
    Separates valid chapters from junk (covers, TOC, copyright, newsletters).
    """
    source_files = sorted(
        [p for p in INPUT_DIR.glob("*.epub")],
        key=natural_epub_sort_key,
    )
    if not source_files:
        print(f"[ERROR] No .epub files found in {INPUT_DIR}/.")
        sys.exit(1)

    CONFIRM_DIR.mkdir(exist_ok=True)
    JUNK_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"# EPUB Unpack & Confirmation Summary ({LANG})\n",
        "This folder contains the unpacked XHTML files in exact reading spine order.",
        f"Please review the files before running Step 2 (`python extract_text.py --step text --{LANG}`).\n",
        f"- **Kept files**: In `{CONFIRM_DIR}/Volume_XX/` (these will be converted to text chapters)",
        f"- **Filtered junk**: In `{JUNK_DIR}/Volume_XX/` (covers, TOC, copyright, etc.)",
        f"- If an unwanted file slipped into `{CONFIRM_DIR}/Volume_XX/`, simply DELETE it or move it to `_excluded_junk/`.",
        f"- If a real story file was moved to `_excluded_junk/`, simply move it back to `{CONFIRM_DIR}/Volume_XX/`.\n",
        "## Summary by Volume:\n",
    ]

    total_chapters = 0
    total_junks = 0

    print("=" * 65)
    print(f"STEP 1: UNPACKING EPUBS TO CONFIRMATION FOLDER ({CONFIRM_DIR}/)")
    print("=" * 65)

    for vol_idx, epub_path in enumerate(source_files, 1):
        vol_name = f"Volume_{vol_idx:02d}"
        vol_confirm_dir = CONFIRM_DIR / vol_name
        vol_junk_dir = JUNK_DIR / vol_name
        if vol_confirm_dir.exists():
            shutil.rmtree(vol_confirm_dir)
        if vol_junk_dir.exists():
            shutil.rmtree(vol_junk_dir)
        vol_confirm_dir.mkdir(parents=True, exist_ok=True)
        vol_junk_dir.mkdir(parents=True, exist_ok=True)


        print(f"\n[{vol_name}] Processing: {epub_path.name}")
        spine, manifest, toc_map = parse_epub_structure(epub_path)
        print(f"  Spine items in reading order: {len(spine)}")

        vol_chapters = 0
        vol_junks = 0
        last_known_title = ""
        chap_counter = 1

        summary_lines.append(f"### {vol_name} ({epub_path.name})\n")
        summary_lines.append("| Status | File Name | Est. Chars | TOC Title |")
        summary_lines.append("|---|---|---|---|")

        with zipfile.ZipFile(epub_path, "r") as z:
            for spine_idx, item_href in enumerate(spine, 1):
                if not item_href.endswith((".xhtml", ".html")):
                    continue

                raw_content = z.read(item_href)
                text_content = clean_html_to_text(raw_content)
                text_len = len(text_content)

                # Title resolution
                toc_title = toc_map.get(item_href, "")
                base_name = posixpath.basename(item_href)
                is_junk = is_junk_page(base_name, toc_title, text_content)

                if toc_title and not is_junk:
                    last_known_title = toc_title

                title_part = sanitize_filename(toc_title if toc_title else last_known_title)
                if not title_part:
                    title_part = Path(base_name).stem


                clean_stem = Path(base_name).stem

                if is_junk:
                    junk_out_name = f"vol{vol_idx:02d}_junk_{spine_idx:02d}_{clean_stem}.xhtml"
                    out_path = vol_junk_dir / junk_out_name
                    out_path.write_bytes(raw_content)
                    vol_junks += 1
                    summary_lines.append(
                        f"| EXCLUDED | `{junk_out_name}` | {text_len} | {toc_title or '-'} |"
                    )
                else:
                    chap_out_name = (
                        f"vol{vol_idx:02d}_{chap_counter:02d}_{clean_stem}_{title_part}.xhtml"
                    )
                    out_path = vol_confirm_dir / chap_out_name
                    out_path.write_bytes(raw_content)
                    vol_chapters += 1
                    chap_counter += 1
                    summary_lines.append(
                        f"| **KEPT** | `{chap_out_name}` | {text_len} | {toc_title or last_known_title or '-'} |"
                    )

        print(f"  -> Kept chapters: {vol_chapters}")
        print(f"  -> Filtered junk/inserts: {vol_junks} (saved in {vol_junk_dir.name}/)")
        total_chapters += vol_chapters
        total_junks += vol_junks
        summary_lines.append("")

    summary_file = CONFIRM_DIR / "SUMMARY.md"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"UNPACK COMPLETE: {total_chapters} chapters kept, {total_junks} junk files filtered.")
    print(f"Summary report written to: {summary_file}")
    print("=" * 65)
    print("\n[NEXT STEP]:")
    print(f"1. Open the folder: {CONFIRM_DIR}/")
    print("   Check the files in each Volume folder.")
    print("   - Delete any file you don't want (e.g. afterword if you don't want it).")
    print(f"   - Check _excluded_junk/ in {CONFIRM_DIR}/ to ensure no real chapters were filtered by accident.")
    print("2. Once satisfied, run:")
    print(f"   python extract_text.py --step text --{LANG}")
    print("=" * 65)


def step_convert_xhtml_to_text():
    """
    Step 2: Converts confirmed .xhtml files from confirm_xhtml-{LANG}/ into clean plain text
    in text-{LANG}/chapter_0001.txt, text-{LANG}/chapter_0002.txt, etc.
    """
    volume_dirs = sorted([d for d in CONFIRM_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if not volume_dirs:
        print(f"[ERROR] No confirmed volume folders found in {CONFIRM_DIR}/.")
        print(f"Run `python extract_text.py --step unpack --{LANG}` first.")
        sys.exit(1)

    # Backup / clear old text files if present
    TEXT_DIR.mkdir(exist_ok=True)
    old_files = list(TEXT_DIR.glob("chapter_*.txt"))
    if old_files:
        backup_dir = TEXT_DIR / "_old_backup"
        backup_dir.mkdir(exist_ok=True)
        for f in old_files:
            try:
                shutil.move(str(f), str(backup_dir / f.name))
            except Exception:
                pass
        print(f"[INFO] Moved {len(old_files)} old text files to {backup_dir}/ to prevent junk contamination.")

    manifest = []
    chapter_counter = 1
    min_step2_chars = MIN_CHAPTER_CHARS_JP if LANG == "JP" else 50

    print("=" * 65)
    print(f"STEP 2: CONVERTING CONFIRMED XHTML FILES TO PLAIN TEXT ({TEXT_DIR}/)")
    print("=" * 65)

    for vol_dir in volume_dirs:
        xhtml_files = sorted(
            [f for f in vol_dir.glob("*.xhtml") if not f.name.startswith("_")]
            + [f for f in vol_dir.glob("*.html") if not f.name.startswith("_")]
        )
        print(f"\nProcessing {vol_dir.name} ({len(xhtml_files)} confirmed files)...")

        for xhtml_path in xhtml_files:
            raw = xhtml_path.read_bytes()
            clean_text = clean_html_to_text(raw)
            if not clean_text or len(clean_text) < min_step2_chars:
                print(f"  [SKIP] {xhtml_path.name} is empty or under {min_step2_chars} characters.")
                continue

            # Parse readable title from filename
            # format: vol01_01_cleanstem_TitlePart.xhtml
            parts = xhtml_path.stem.split("_", 3)
            display_title = parts[3].replace("_", " ") if len(parts) >= 4 else xhtml_path.stem

            header = (
                f"# [{vol_dir.name.replace('_', ' ')}] {display_title}\n"
                f"# Source: {xhtml_path.name}\n"
                f"#{'-' * 68}\n\n"
            )

            out_filename = f"chapter_{chapter_counter:04d}.txt"
            out_path = TEXT_DIR / out_filename
            out_path.write_text(header + clean_text, encoding="utf-8")

            manifest.append(
                {
                    "chapter_file": out_filename,
                    "volume": vol_dir.name,
                    "title": display_title,
                    "source_xhtml": xhtml_path.name,
                    "char_count": len(clean_text),
                }
            )
            chapter_counter += 1

    manifest_file = TEXT_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"TEXT EXTRACTION COMPLETE: {len(manifest)} chapters extracted to {TEXT_DIR}/")
    print(f"Manifest written to: {manifest_file}")
    print("=" * 65)
    print("\n[NEXT STEP]:")
    print(f"1. Spot-check files in {TEXT_DIR}/ (e.g. chapter_0001.txt, chapter_0002.txt).")
    print("   Notice that Table of Contents and junk files are completely gone!")
    print(f"2. When ready, run the LLM persona extractor:")
    print(f"   python persona_extractor.py --{LANG}")
    print("=" * 65)


def extract_pdf_fallback():
    """Fallback extractor for PDF files in input/ using pdfplumber."""
    import pdfplumber

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        return

    print(f"[PDF] Found {len(pdf_files)} PDF file(s). Extracting with fallback...")
    TEXT_DIR.mkdir(exist_ok=True)
    manifest = []
    chapter_idx = 1

    for pdf_path in pdf_files:
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        joined = "\n\n".join(pages)
        # Split heuristic
        pattern = re.compile(
            r"(?:^|\n)\s*((?:Chapter|CHAPTER|Prologue|PROLOGUE|Epilogue|EPILOGUE|"
            r"第[一二三四五六七八九十百零〇0-9０-９]+[章話]|プロローグ|エピローグ|幕間)\b[^\n]{0,60})\n",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(joined))
        chapters = []
        if len(matches) >= 2:
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
                chapters.append(joined[start:end].strip())
        else:
            words = joined.split()
            chunk_size = 3000
            chapters = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

        for c_text in chapters:
            if len(c_text) < MIN_CHAPTER_CHARS:
                continue
            out_name = f"chapter_pdf_{chapter_idx:04d}.txt"
            (TEXT_DIR / out_name).write_text(c_text, encoding="utf-8")
            manifest.append(
                {
                    "chapter_file": out_name,
                    "source_file": pdf_path.name,
                    "char_count": len(c_text),
                }
            )
            chapter_idx += 1

    print(f"[PDF] Done. {len(manifest)} PDF chapters added.")


def main():
    parser = argparse.ArgumentParser(
        description="Two-step novel extraction tool: EPUB -> confirm_xhtml-<LANG>/ -> text-<LANG>/"
    )
    parser.add_argument(
        "--step",
        choices=["unpack", "1", "text", "2"],
        help="Run Step 1 (unpack to confirm_xhtml-<LANG>/) or Step 2 (convert confirmed XHTML to text-<LANG>/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both Step 1 and Step 2 automatically in sequence",
    )
    parser.add_argument(
        "--lang",
        choices=["EN", "JP", "ID", "en", "jp", "id"],
        default=None,
        help="Language mode: EN, JP, or ID (default: EN or DATASET_LANG env)",
    )
    parser.add_argument("--EN", dest="lang_en", action="store_true", help="Shortcut for --lang EN")
    parser.add_argument("--JP", dest="lang_jp", action="store_true", help="Shortcut for --lang JP")
    parser.add_argument("--ID", dest="lang_id", action="store_true", help="Shortcut for --lang ID")
    args = parser.parse_args()

    # Resolve active language mode
    if args.lang_jp:
        target_lang = "JP"
    elif args.lang_en:
        target_lang = "EN"
    elif args.lang_id:
        target_lang = "ID"
    elif args.lang:
        target_lang = args.lang.upper()
    else:
        target_lang = os.environ.get("DATASET_LANG", "EN").upper()

    set_language(target_lang)

    if not INPUT_DIR.exists() or not any(INPUT_DIR.iterdir()):
        print(f"[ERROR] No files found in {INPUT_DIR}/. Put your .epub files there first.")
        sys.exit(1)

    has_epub = any(INPUT_DIR.glob("*.epub"))
    has_pdf = any(INPUT_DIR.glob("*.pdf"))

    if args.all:
        if has_epub:
            step_unpack_epubs()
            step_convert_xhtml_to_text()
        if has_pdf:
            extract_pdf_fallback()
        return

    if args.step in ["unpack", "1"]:
        if has_epub:
            step_unpack_epubs()
        if has_pdf:
            extract_pdf_fallback()
        return

    if args.step in ["text", "2"]:
        step_convert_xhtml_to_text()
        return

    # Default if no arguments given:
    if not CONFIRM_DIR.exists() or not any(CONFIRM_DIR.iterdir()):
        print(f"[INFO] No {CONFIRM_DIR}/ directory found. Running Step 1 (Unpack & Triage)...")
        if has_epub:
            step_unpack_epubs()
        if has_pdf:
            extract_pdf_fallback()
    else:
        print(f"[INFO] {CONFIRM_DIR}/ already exists.")
        print("Choose an action:")
        print(f"  1. Convert confirmed files to text:  python extract_text.py --step text --{LANG}")
        print(f"  2. Re-unpack everything from EPUB:    python extract_text.py --step unpack --{LANG}")
        print(f"  3. Run both automatically:          python extract_text.py --all --{LANG}")


if __name__ == "__main__":
    main()
