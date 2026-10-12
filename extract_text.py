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
import posixpath
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

INPUT_DIR = Path("input")
CONFIRM_DIR = Path("confirm_xhtml")
JUNK_DIR = CONFIRM_DIR / "_excluded_junk"
TEXT_DIR = Path("text")

# Patterns identifying non-story pages (frontmatter, backmatter, afterwords, ads)
JUNK_NAME_PATTERN = re.compile(
    r"(cover|insert|titlepage|copyright|toc|nav|newsletter|advertisement|colophon|contents|afterword|appendix|postscript)",
    re.IGNORECASE,
)


# Minimum text length (in characters) to be considered a real story chapter.
# Single illustration/image pages usually have < 100 characters.
MIN_CHAPTER_CHARS = 200

# PDF boilerplate detection settings
BOILERPLATE_EDGE_LINES = 3
BOILERPLATE_MIN_PAGES = 8
BOILERPLATE_FREQ_THRESHOLD = 0.5


def sanitize_filename(name: str) -> str:
    """Sanitize string for Windows filesystem."""
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = re.sub(r"[\s_]+", "_", s)
    return s.strip("._")


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


def parse_epub_structure(epub_path: Path):
    """
    Parses an EPUB's true spine (reading order), manifest, and Table of Contents (TOC).
    Returns:
        spine_items: list of archive href paths in exact reading order
        toc_map: dict mapping archive href -> chapter title from TOC
    """
    with zipfile.ZipFile(epub_path, "r") as z:
        opf_path, opf_dir = find_opf_and_dir(z)
        opf_root = ET.fromstring(z.read(opf_path))

        manifest = {}
        for item in opf_root.findall(".//{http://www.idpf.org/2007/opf}item"):
            item_id = item.attrib.get("id")
            href = item.attrib.get("href")
            full_href = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
            manifest[item_id] = {
                "href": full_href,
                "media_type": item.attrib.get("media-type", ""),
            }

        spine = []
        for itemref in opf_root.findall(".//{http://www.idpf.org/2007/opf}itemref"):
            idref = itemref.attrib.get("idref")
            if idref in manifest:
                spine.append(manifest[idref]["href"])

        # Locate TOC file
        toc_map = {}
        toc_href = None
        for item_id, info in manifest.items():
            href_lower = info["href"].lower()
            if "toc" in item_id.lower() or href_lower.endswith(("toc.xhtml", "toc.html", "nav.xhtml")):
                toc_href = info["href"]
                break

        if toc_href and toc_href in z.namelist():
            toc_soup = BeautifulSoup(z.read(toc_href), "html.parser")
            for a in toc_soup.find_all("a"):
                title = a.get_text().strip()
                target_raw = a.get("href", "").split("#")[0]
                if not target_raw:
                    continue
                target_full = posixpath.normpath(
                    posixpath.join(posixpath.dirname(toc_href), target_raw)
                )
                if target_full not in toc_map and title:
                    toc_map[target_full] = title

        return spine, manifest, toc_map


def clean_html_to_text(raw_html: bytes | str) -> str:
    """
    Converts novel XHTML/HTML to clean, formatted text:
    - Strips scripts/styles/meta
    - Preserves paragraphs and dialogues
    - Normalizes unicode typography, whitespace, and dashes
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    for elem in soup(["script", "style", "meta", "link"]):
        elem.decompose()

    paragraphs = []
    # Collect block-level novel elements
    for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "div"]):
        # Skip wrapper elements that contain nested paragraphs
        if not p.find(["p", "h1", "h2", "h3", "blockquote"]):
            txt = p.get_text(strip=True)
            if txt:
                paragraphs.append(txt)

    if not paragraphs:
        txt = soup.get_text(separator="\n", strip=True)
        paragraphs = [line.strip() for line in txt.splitlines() if line.strip()]

    joined = "\n\n".join(paragraphs)
    joined = joined.replace("\r\n", "\n").replace("\r", "\n")
    joined = joined.replace("\xa0", " ").replace("\u3000", " ")
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def step_unpack_epubs():
    """
    Step 1: Unpacks EPUBs from input/ into confirm_xhtml/.
    Separates valid chapters from junk (covers, TOC, copyright, newsletters).
    """
    source_files = sorted(
        [p for p in INPUT_DIR.glob("*.epub")]
    )
    if not source_files:
        print(f"[ERROR] No .epub files found in {INPUT_DIR}/.")
        sys.exit(1)

    CONFIRM_DIR.mkdir(exist_ok=True)
    JUNK_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# EPUB Unpack & Confirmation Summary\n",
        "This folder contains the unpacked XHTML files in exact reading spine order.",
        "Please review the files before running Step 2 (`python extract_text.py --step text`).\n",
        "- **Kept files**: In `confirm_xhtml/Volume_XX/` (these will be converted to text chapters)",
        "- **Filtered junk**: In `confirm_xhtml/_excluded_junk/Volume_XX/` (covers, TOC, copyright, etc.)",
        "- If an unwanted file slipped into `confirm_xhtml/Volume_XX/`, simply DELETE it or move it to `_excluded_junk/`.",
        "- If a real story file was moved to `_excluded_junk/`, simply move it back to `confirm_xhtml/Volume_XX/`.\n",
        "## Summary by Volume:\n",
    ]

    total_chapters = 0
    total_junks = 0

    print("=" * 65)
    print("STEP 1: UNPACKING EPUBS TO CONFIRMATION FOLDER (confirm_xhtml/)")
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
                if toc_title:
                    last_known_title = toc_title

                # Determine if it's junk (checking both filename and TOC title)
                is_junk = False
                base_name = posixpath.basename(item_href)
                if JUNK_NAME_PATTERN.search(base_name) or (toc_title and JUNK_NAME_PATTERN.search(toc_title)):
                    is_junk = True
                elif text_len < MIN_CHAPTER_CHARS:
                    # Very short page, almost certainly illustration/insert/cover
                    is_junk = True

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
    print("1. Open the folder: confirm_xhtml/")
    print("   Check the files in each Volume folder.")
    print("   - Delete any file you don't want (e.g. afterword if you don't want it).")
    print("   - Check _excluded_junk/ to ensure no real chapters were filtered by accident.")
    print("2. Once satisfied, run:")
    print("   python extract_text.py --step text")
    print("=" * 65)


def step_convert_xhtml_to_text():
    """
    Step 2: Converts confirmed .xhtml files from confirm_xhtml/ into clean plain text
    in text/chapter_0001.txt, text/chapter_0002.txt, etc.
    """
    volume_dirs = sorted([d for d in CONFIRM_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if not volume_dirs:
        print(f"[ERROR] No confirmed volume folders found in {CONFIRM_DIR}/.")
        print("Run `python extract_text.py --step unpack` first.")
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

    print("=" * 65)
    print("STEP 2: CONVERTING CONFIRMED XHTML FILES TO PLAIN TEXT (text/)")
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
            if not clean_text or len(clean_text) < 50:
                print(f"  [SKIP] {xhtml_path.name} is empty or under 50 characters.")
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
    print("2. When ready, run the LLM persona extractor:")
    print("   python persona_extractor.py")
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
            r"(?:^|\n)\s*((?:Chapter|CHAPTER|Prologue|PROLOGUE|Epilogue|EPILOGUE)\b[^\n]{0,60})\n",
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
        description="Two-step novel extraction tool: EPUB -> confirm_xhtml/ -> text/"
    )
    parser.add_argument(
        "--step",
        choices=["unpack", "1", "text", "2"],
        help="Run Step 1 (unpack to confirm_xhtml/) or Step 2 (convert confirmed XHTML to text/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both Step 1 and Step 2 automatically in sequence",
    )
    args = parser.parse_args()

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
        print("[INFO] No confirm_xhtml/ directory found. Running Step 1 (Unpack & Triage)...")
        if has_epub:
            step_unpack_epubs()
        if has_pdf:
            extract_pdf_fallback()
    else:
        print("[INFO] confirm_xhtml/ already exists.")
        print("Choose an action:")
        print("  1. Convert confirmed files to text:  python extract_text.py --step text")
        print("  2. Re-unpack everything from EPUB:    python extract_text.py --step unpack")
        print("  3. Run both automatically:          python extract_text.py --all")


if __name__ == "__main__":
    main()
