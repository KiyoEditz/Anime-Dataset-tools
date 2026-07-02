"""
dialogue_extractor.py

Tahap 3: Baca tiap bab, cari adegan di mana Elaina berbicara, lalu ubah
jadi pasangan percakapan mentah: konteks/pertanyaan -> respons Elaina.

Jalankan setelah persona_extractor.py:
    python dialogue_extractor.py

Hasil:
    output/raw_dialogue_pairs.jsonl
    Tiap baris: {"chapter": "...", "scenario": "...", "user_message": "...", "elaina_response": "..."}
"""

import json
import os
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json

TEXT_DIR = Path("text")
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "raw_dialogue_pairs.jsonl"

# Naikkan kalau model kamu punya context window besar (mis. 128k via NVIDIA
# Build) -> konteks adegan lebih utuh & lebih sedikit panggilan API.
# Untuk model 40k seperti Hermes-4-14B, nilai default ini aman (sisa ruang
# besar untuk system prompt + output JSON).
DIALOGUE_CHUNK_CHARS = int(os.environ.get("DIALOGUE_CHUNK_CHARS", "4000"))

# Checkpoint: daftar nama bab yang SUDAH SELESAI diproses (semua chunk-nya).
# Kalau script terhenti di tengah jalan, rerun berikutnya melanjutkan dari
# bab yang belum selesai -- tanpa duplikasi baris di raw_dialogue_pairs.jsonl.
CHECKPOINT_FILE = OUTPUT_DIR / "dialogue_checkpoint.json"


def load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return set(data.get("completed_chapters", []))
    except json.JSONDecodeError:
        print("[WARN] Checkpoint rusak/tidak terbaca, mulai dari awal.")
        return set()


def save_checkpoint(completed_chapters: set):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed_chapters": sorted(completed_chapters)}, ensure_ascii=False),
        encoding="utf-8",
    )

SYSTEM_PROMPT = """Kamu membantu menyiapkan data fine-tuning chatbot untuk tokoh Elaina, \
dari novel "Majo no Tabitabi / The Journey of Elaina".

Dari cuplikan bab yang diberikan, cari SEMUA adegan di mana Elaina berbicara \
(baik dialog langsung maupun monolog batin yang menunjukkan kepribadiannya). \
Untuk tiap adegan, hasilkan satu entri dengan struktur:

{
  "scenario": "ringkasan singkat (1-2 kalimat) situasi/latar adegan ini",
  "user_message": "pesan/pertanyaan dari lawan bicara Elaina (atau narasi situasi jika tidak ada lawan bicara langsung) yang memicu respons Elaina, ditulis ulang dalam bahasa Inggris natural",
  "elaina_response": "respons Elaina dalam bahasa Inggris, pertahankan kepribadian, nada, dan gaya bicaranya seakurat mungkin"
}

Balas dengan JSON array berisi entri-entri tersebut. Kalau tidak ada adegan Elaina \
berbicara di cuplikan ini, balas dengan array kosong []. JANGAN menyertakan adegan \
yang tidak melibatkan Elaina berbicara/bereaksi.

Balas HANYA dengan JSON array, tanpa teks lain, tanpa markdown code fence.
"""


def main():
    chapter_files = sorted(TEXT_DIR.glob("chapter_*.txt"))
    if not chapter_files:
        raise SystemExit("Tidak ada file bab di text/. Jalankan extract_text.py dulu.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    completed = load_checkpoint()

    if completed:
        chapter_names = {p.name for p in chapter_files}
        stale = completed - chapter_names
        if stale:
            print(
                f"[WARN] Checkpoint menyebut {len(stale)} bab yang tidak lagi ada di "
                f"text/ (mungkin daftar bab berubah). Memproses ULANG SEMUA bab dari "
                f"awal supaya konsisten. Hapus output/dialogue_checkpoint.json dan "
                f"output/raw_dialogue_pairs.jsonl dulu kalau ini tidak diinginkan."
            )
            completed = set()

    remaining = [p for p in chapter_files if p.name not in completed]

    if completed:
        print(
            f"[INFO] Melanjutkan dari checkpoint: {len(completed)} bab sudah "
            f"selesai sebelumnya, sisa {len(remaining)} bab."
        )
    if not remaining:
        print("[INFO] Semua bab sudah selesai diproses (sesuai checkpoint).")
        return

    # Append kalau melanjutkan dari checkpoint (baris lama dipertahankan),
    # overwrite kalau ini run pertama dari awal.
    file_mode = "a" if completed else "w"
    total_pairs = 0

    with open(OUTPUT_FILE, file_mode, encoding="utf-8") as out_f:
        for chap_path in tqdm(remaining, desc="Ekstraksi dialog Elaina"):
            chapter_text = chap_path.read_text(encoding="utf-8")

            # proses per potongan supaya aman utk context window model yang dipakai
            chunk_size = DIALOGUE_CHUNK_CHARS
            chunks = [
                chapter_text[i : i + chunk_size]
                for i in range(0, len(chapter_text), chunk_size)
            ]

            # Kumpulkan hasil satu bab dulu di memori, baru ditulis ke file
            # SETELAH semua chunk bab ini selesai. Ini penting supaya kalau
            # bab ini gagal di tengah (mis. rate limit harian), tidak ada
            # baris setengah-jadi yang tertulis -- bab ini akan diulang UTUH
            # saat resume, tanpa risiko duplikasi.
            chapter_records = []

            for chunk in chunks:
                if len(chunk.strip()) < 200:
                    continue
                try:
                    pairs = chat_json(SYSTEM_PROMPT, chunk)
                except ValueError as e:
                    print(f"[WARN] Lewati potongan di {chap_path.name}: {e}")
                    continue

                if not isinstance(pairs, list):
                    continue

                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    required = {"scenario", "user_message", "elaina_response"}
                    if not required.issubset(pair.keys()):
                        continue
                    if not pair["elaina_response"].strip():
                        continue

                    chapter_records.append(
                        {
                            "chapter": chap_path.name,
                            "scenario": pair["scenario"].strip(),
                            "user_message": pair["user_message"].strip(),
                            "elaina_response": pair["elaina_response"].strip(),
                        }
                    )

            for record in chapter_records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            total_pairs += len(chapter_records)

            completed.add(chap_path.name)
            save_checkpoint(completed)

    print(f"\nSelesai. {total_pairs} pasangan dialog BARU ditambahkan ke {OUTPUT_FILE}")
    print(f"Total bab selesai: {len(completed)}/{len(chapter_files)}")


if __name__ == "__main__":
    main()
