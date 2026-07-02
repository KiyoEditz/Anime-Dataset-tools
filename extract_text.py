"""
extract_text.py

Tahap 1: Ekstraksi teks mentah dari file EPUB/PDF novel menjadi
file .txt per-bab yang bersih, siap diproses oleh tahap berikutnya.

Cara pakai:
    1. Taruh semua file .epub / .pdf novel di folder input/
       (urutkan nama file agar sesuai urutan volume/bab, misal:
        01_volume1.epub, 02_volume2.epub, dst.)
    2. Jalankan:
       python extract_text.py

Hasil:
    text/chapter_0001.txt, text/chapter_0002.txt, dst.
    text/manifest.json  -> metadata urutan & sumber tiap bab
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
MIN_CHAPTER_CHARS = 400  # buang fragmen terlalu pendek (halaman judul, dll)

# Hanya N baris pertama & terakhir tiap halaman yang dianggap kandidat
# header/footer/nomor halaman (di situlah noise jenis ini biasanya muncul).
# Untuk halaman pendek, zona ini otomatis diperkecil (lihat _edge_zone_size)
# supaya tidak pernah menelan sebagian besar isi halaman.
BOILERPLATE_EDGE_LINES = 3
# Minimal jumlah halaman sebelum deteksi pola berulang dijalankan
# (di buku pendek, "berulang" tidak cukup bermakna secara statistik).
BOILERPLATE_MIN_PAGES = 8
# Kalau sebuah baris (setelah dinormalisasi) muncul di >= proporsi ini
# dari total halaman, dianggap boilerplate dan dibuang. Diset cukup tinggi
# (lebih dari separuh halaman) karena header/footer/nomor halaman asli
# biasanya muncul di HAMPIR SEMUA halaman -- ambang tinggi mengurangi risiko
# salah membuang kalimat cerita yang kebetulan mirip.
BOILERPLATE_FREQ_THRESHOLD = 0.5


def _edge_zone_size(total_lines: int) -> int:
    """Ukuran zona tepi yang diperiksa, diskalakan agar tidak pernah melebihi
    ~25% dari total baris halaman (penting untuk halaman pendek)."""
    return max(1, min(BOILERPLATE_EDGE_LINES, total_lines // 4))


def _normalize_line(line: str) -> str:
    """Normalisasi baris untuk perbandingan: rapikan spasi, ganti angka jadi '#'."""
    s = line.strip()
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def strip_repeating_boilerplate(pages: list[str]) -> list[str]:
    """
    Deteksi & buang baris yang berulang di banyak halaman secara identik
    (setelah angka dinormalisasi) -- pola khas nomor halaman, running
    header/footer (judul buku, nama penulis, dst). Hanya memeriksa
    beberapa baris di tepi atas/bawah tiap halaman supaya tidak pernah
    menyentuh isi cerita di tengah halaman. Baris yang menyerupai penanda
    bab (Chapter/Prologue/Epilogue) sengaja dikecualikan agar
    split_into_chapters() tetap bisa mendeteksinya.
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
    print(f"[INFO] {len(boilerplate)} pola noise berulang terdeteksi & dibuang (contoh: {sample})")

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
    """Bersihkan whitespace berlebih, hyphenation patah baris, dsb."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # gabungkan kata yang terpenggal tanda hubung di akhir baris (umum di PDF)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # rapikan baris kosong berlebih
    text = re.sub(r"\n{3,}", "\n\n", text)
    # rapikan spasi ganda
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_chapters(text: str) -> list[str]:
    """
    Heuristik pemisah bab: cari pola umum seperti
    'Chapter 1', 'Chapter One', 'Prologue', 'Epilogue', angka romawi, dll.
    Kalau tidak ditemukan pola jelas, fallback ke pemotongan per ~3000 kata.
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

    # Fallback: tidak ada penanda bab yang jelas -> potong per blok kata
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
        print(f"Tidak ada file di {INPUT_DIR}/. Taruh file .epub atau .pdf di sana dulu.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = []
    chapter_idx = 1

    source_files = sorted(
        [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".epub", ".pdf")]
    )

    if not source_files:
        print("Tidak ditemukan file .epub/.pdf di folder input/.")
        sys.exit(1)

    for src in source_files:
        print(f"Memproses {src.name} ...")
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

    print(f"\nSelesai. {len(manifest)} bab diekstrak ke {OUTPUT_DIR}/")
    print("Cek beberapa file .txt secara manual untuk memastikan hasil ekstraksi bersih")
    print("(terutama untuk PDF, karena header/footer/nomor halaman kadang ikut terekstrak).")


if __name__ == "__main__":
    main()
