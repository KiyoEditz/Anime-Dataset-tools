"""
build_dataset.py

Stage 5: Combine the system prompt + raw dialogue pairs (from the novel) +
synthetic pairs (optional) into one final JSONL file, in the standard
chat fine-tuning format ("messages" field with system/user/assistant roles).

Run LAST, after:
    1. extract_text.py
    2. persona_extractor.py
    3. dialogue_extractor.py
    4. augment_synthetic.py   (optional)

    python build_dataset.py

Output:
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
        raise SystemExit("output/elaina_system_prompt.txt not found. Run persona_extractor.py first.")

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    raw_pairs = load_jsonl(RAW_DIALOGUE_FILE)
    synthetic_pairs = load_jsonl(SYNTHETIC_FILE)

    print(f"Pairs from the novel: {len(raw_pairs)}")
    print(f"Synthetic pairs     : {len(synthetic_pairs)}")

    seen_responses = set()
    final_records = []

    for pair in raw_pairs + synthetic_pairs:
        user_msg = pair.get("user_message", "").strip()
        response = pair.get("elaina_response", "").strip()

        if len(response) < MIN_RESPONSE_CHARS or not user_msg:
            continue

        # simple dedup based on the response (case-insensitive)
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

    print(f"\nDone. {len(final_records)} final fine-tuning example(s) saved to {FINAL_FILE}")
    if len(final_records) < 100:
        print("Note: this is still a small number of examples for stable fine-tuning.")
        print("Consider increasing the batch count in augment_synthetic.py, or check whether")
        print("dialogue_extractor.py skipped a lot of chunks (see [WARN] messages during the run).")


if __name__ == "__main__":
    main()
