"""
augment_synthetic.py

Stage 4 (optional but recommended): generate ADDITIONAL, original
question-answer pairs (not quoted from the novel), based on the Elaina
profile already built. This helps the model generalize (answer questions
that AREN'T verbatim in the novel) and reduces reliance on verbatim
original text.

Run after persona_extractor.py:
    python augment_synthetic.py [batch_count]

Output:
    output/synthetic_pairs.jsonl
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json

OUTPUT_DIR = Path("output")
PROFILE_FILE = OUTPUT_DIR / "elaina_profile.json"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"

PAIRS_PER_BATCH = 8

SYSTEM_PROMPT = """You are helping prepare chatbot fine-tuning data for the character Elaina \
from the novel "Majo no Tabitabi / The Journey of Elaina", based on the character \
bible provided below.

Create {n} NEW and ORIGINAL question-answer pairs (not quotes/scenes taken \
directly from the novel) that test whether someone can consistently roleplay \
as Elaina. Vary the type of question: about her past, her opinion on \
something, her reaction to a new hypothetical situation, small everyday \
things, etc. Answers MUST be consistent with the personality, values, \
speech style, and memories in the following character bible:

{profile}

Format for each entry:
{{"user_message": "a question/situation in English", "elaina_response": "Elaina's answer in English, matching her speech style"}}

Reply with a JSON array containing {n} such entries. ONLY JSON, no other text.
"""


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not PROFILE_FILE.exists():
        raise SystemExit("output/elaina_profile.json not found. Run persona_extractor.py first.")

    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    profile_str = json.dumps(profile, ensure_ascii=False)

    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for _ in tqdm(range(n_batches), desc="Generating synthetic Q&A"):
            prompt = SYSTEM_PROMPT.format(n=PAIRS_PER_BATCH, profile=profile_str)
            try:
                pairs = chat_json(prompt, "Generate the next batch, avoid repeating themes from previous batches.", temperature=0.8)
            except ValueError as e:
                print(f"[WARN] Skipping a batch: {e}")
                continue

            if not isinstance(pairs, list):
                continue

            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if not {"user_message", "elaina_response"}.issubset(pair.keys()):
                    continue
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nDone. {total} synthetic pair(s) saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
