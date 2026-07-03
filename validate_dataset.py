"""
validate_dataset.py

Final sanity check for output/elaina_finetune_dataset.jsonl:
- every line is valid JSON with the correct "messages" structure
- basic stats: example count, average length, random samples to eyeball

Run:
    python validate_dataset.py
"""

import json
import random
import statistics
from pathlib import Path

DATASET_FILE = Path("output/elaina_finetune_dataset.jsonl")


def main():
    if not DATASET_FILE.exists():
        raise SystemExit(f"{DATASET_FILE} not found. Run build_dataset.py first.")

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
                print(f"[ERROR] Line {i}: not valid JSON")
                errors += 1
                continue

            msgs = rec.get("messages")
            if not isinstance(msgs, list) or len(msgs) != 3:
                print(f"[ERROR] Line {i}: 'messages' structure is wrong (must have 3 turns)")
                errors += 1
                continue

            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                print(f"[ERROR] Line {i}: wrong role order -> {roles}")
                errors += 1
                continue

            records.append(rec)

    if not records:
        raise SystemExit("No valid records found.")

    response_lengths = [len(r["messages"][2]["content"]) for r in records]
    user_lengths = [len(r["messages"][1]["content"]) for r in records]

    print(f"\nTotal valid examples : {len(records)}")
    print(f"Total errors         : {errors}")
    print(f"User message length  : avg {statistics.mean(user_lengths):.0f} chars, "
          f"min {min(user_lengths)}, max {max(user_lengths)}")
    print(f"Response length      : avg {statistics.mean(response_lengths):.0f} chars, "
          f"min {min(response_lengths)}, max {max(response_lengths)}")

    print("\n--- 3 random examples for a manual check ---\n")
    for rec in random.sample(records, min(3, len(records))):
        print("USER     :", rec["messages"][1]["content"][:200])
        print("ELAINA   :", rec["messages"][2]["content"][:200])
        print("-" * 60)


if __name__ == "__main__":
    main()
