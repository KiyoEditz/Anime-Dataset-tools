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

import argparse
import json
import os
from pathlib import Path

LANG = os.environ.get("DATASET_LANG", "EN").upper()
OUTPUT_DIR = Path(f"output-{LANG}")
SYSTEM_PROMPT_FILE = OUTPUT_DIR / "elaina_system_prompt.txt"
RAW_DIALOGUE_FILE = OUTPUT_DIR / "raw_dialogue_pairs.jsonl"
SYNTHETIC_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"
FINAL_FILE = OUTPUT_DIR / "elaina_finetune_dataset.jsonl"

MIN_RESPONSE_CHARS = 5


def set_language(lang: str):
    """Dynamically set the active language and target directories."""
    global LANG, OUTPUT_DIR, SYSTEM_PROMPT_FILE, RAW_DIALOGUE_FILE, SYNTHETIC_FILE, FINAL_FILE
    LANG = lang.upper()
    OUTPUT_DIR = Path(f"output-{LANG}")
    SYSTEM_PROMPT_FILE = OUTPUT_DIR / "elaina_system_prompt.txt"
    RAW_DIALOGUE_FILE = OUTPUT_DIR / "raw_dialogue_pairs.jsonl"
    SYNTHETIC_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"
    FINAL_FILE = OUTPUT_DIR / "elaina_finetune_dataset.jsonl"


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
    parser = argparse.ArgumentParser(description="Assemble final fine-tune dataset JSONL")
    parser.add_argument("--lang", choices=["EN", "JP", "ID", "en", "jp", "id"], default=None)
    parser.add_argument("--EN", dest="lang_en", action="store_true", help="Shortcut for --lang EN")
    parser.add_argument("--JP", dest="lang_jp", action="store_true", help="Shortcut for --lang JP")
    parser.add_argument("--ID", dest="lang_id", action="store_true", help="Shortcut for --lang ID")
    args = parser.parse_args()

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

    if not SYSTEM_PROMPT_FILE.exists():
        raise SystemExit(f"{SYSTEM_PROMPT_FILE} not found. Run persona_extractor.py --{LANG} first.")

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    raw_pairs = load_jsonl(RAW_DIALOGUE_FILE)
    synthetic_pairs = load_jsonl(SYNTHETIC_FILE)

    print(f"[{LANG}] Pairs from the novel: {len(raw_pairs)}")
    print(f"[{LANG}] Synthetic pairs     : {len(synthetic_pairs)}")

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

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(final_records)} final fine-tuning example(s) saved to {FINAL_FILE}")
    if len(final_records) < 100:
        print("Note: this is still a small number of examples for stable fine-tuning.")
        print(f"Consider increasing the batch count in augment_synthetic.py --{LANG}, or check whether")
        print(f"dialogue_extractor.py skipped a lot of chunks (see [WARN] messages during the run).")


if __name__ == "__main__":
    main()
