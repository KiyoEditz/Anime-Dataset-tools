"""
validate_dataset.py

Sanity check terakhir untuk output/elaina_finetune_dataset.jsonl:
- semua baris adalah JSON valid dengan struktur "messages" yang benar
- statistik dasar: jumlah contoh, panjang rata-rata, contoh acak untuk dicek mata

Jalankan:
    python validate_dataset.py
"""

import json
import random
import statistics
from pathlib import Path

DATASET_FILE = Path("output/elaina_finetune_dataset.jsonl")


def main():
    if not DATASET_FILE.exists():
        raise SystemExit(f"{DATASET_FILE} tidak ditemukan. Jalankan build_dataset.py dulu.")

    records = []
    errors = 0

    with open(DATASET_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[ERROR] Baris {i}: bukan JSON valid")
                errors += 1
                continue

            msgs = rec.get("messages")
            if not isinstance(msgs, list) or len(msgs) != 3:
                print(f"[ERROR] Baris {i}: struktur 'messages' tidak sesuai (harus 3 turn)")
                errors += 1
                continue

            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                print(f"[ERROR] Baris {i}: urutan role salah -> {roles}")
                errors += 1
                continue

            records.append(rec)

    if not records:
        raise SystemExit("Tidak ada record valid ditemukan.")

    response_lengths = [len(r["messages"][2]["content"]) for r in records]
    user_lengths = [len(r["messages"][1]["content"]) for r in records]

    print(f"\nTotal contoh valid : {len(records)}")
    print(f"Total error         : {errors}")
    print(f"Panjang user msg    : avg {statistics.mean(user_lengths):.0f} karakter, "
          f"min {min(user_lengths)}, max {max(user_lengths)}")
    print(f"Panjang respons     : avg {statistics.mean(response_lengths):.0f} karakter, "
          f"min {min(response_lengths)}, max {max(response_lengths)}")

    print("\n--- 3 contoh acak untuk dicek manual ---\n")
    for rec in random.sample(records, min(3, len(records))):
        print("USER     :", rec["messages"][1]["content"][:200])
        print("ELAINA   :", rec["messages"][2]["content"][:200])
        print("-" * 60)


if __name__ == "__main__":
    main()
