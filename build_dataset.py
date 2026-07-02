"""
build_dataset.py

Tahap 5: Gabungkan system prompt + pasangan dialog mentah (dari novel) +
pasangan sintetis (opsional) menjadi satu file JSONL final, format standar
fine-tuning chat (field "messages" dengan role system/user/assistant).

Jalankan PALING TERAKHIR, setelah:
    1. extract_text.py
    2. persona_extractor.py
    3. dialogue_extractor.py
    4. augment_synthetic.py   (opsional)

    python build_dataset.py

Hasil:
    output/elaina_finetune_dataset.jsonl
"""

import json
from pathlib import Path

OUTPUT_DIR = Path("output")
SYSTEM_PROMPT_FILE = OUTPUT_DIR / "elaina_system_prompt.txt"
RAW_DIALOGUE_FILE = OUTPUT_DIR / "raw_dialogue_pairs.jsonl"
SYNTHETIC_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"
FINAL_FILE = OUTPUT_DIR / "elaina_finetune_dataset.jsonl"

MIN_RESPONSE_CHARS = 5


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main():
    if not SYSTEM_PROMPT_FILE.exists():
        raise SystemExit("output/elaina_system_prompt.txt tidak ditemukan. Jalankan persona_extractor.py dulu.")

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    raw_pairs = load_jsonl(RAW_DIALOGUE_FILE)
    synthetic_pairs = load_jsonl(SYNTHETIC_FILE)

    print(f"Pasangan dari novel: {len(raw_pairs)}")
    print(f"Pasangan sintetis  : {len(synthetic_pairs)}")

    seen_responses = set()
    final_records = []

    for pair in raw_pairs + synthetic_pairs:
        user_msg = pair.get("user_message", "").strip()
        response = pair.get("elaina_response", "").strip()

        if len(response) < MIN_RESPONSE_CHARS or not user_msg:
            continue

        # dedup sederhana berdasarkan respons (case-insensitive)
        dedup_key = response.lower()
        if dedup_key in seen_responses:
            continue
        seen_responses.add(dedup_key)

        final_records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": response},
                ]
            }
        )

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSelesai. {len(final_records)} contoh fine-tuning final tersimpan di {FINAL_FILE}")
    if len(final_records) < 100:
        print("Catatan: jumlah contoh masih sedikit untuk fine-tuning yang stabil.")
        print("Pertimbangkan menambah jumlah batch di augment_synthetic.py, atau cek apakah")
        print("dialogue_extractor.py melewatkan banyak chunk (lihat pesan [WARN] saat run).")


if __name__ == "__main__":
    main()
