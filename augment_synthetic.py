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

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json

LANG = os.environ.get("DATASET_LANG", "EN").upper()
OUTPUT_DIR = Path(f"output-{LANG}")
PROFILE_FILE = OUTPUT_DIR / "elaina_profile.json"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"

PAIRS_PER_BATCH = 8


def set_language(lang: str):
    """Dynamically set the active language and target directories."""
    global LANG, OUTPUT_DIR, PROFILE_FILE, OUTPUT_FILE
    LANG = lang.upper()
    OUTPUT_DIR = Path(f"output-{LANG}")
    PROFILE_FILE = OUTPUT_DIR / "elaina_profile.json"
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"


SYSTEM_PROMPT_EN = """You are helping prepare chatbot fine-tuning data for the character Elaina \
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

SYSTEM_PROMPT_JP = """あなたは小説『魔女の旅々』の主人公・灰の魔女イレイナのチャットボット用ファインチューニングデータを準備するアシスタントです。

以下のキャラクター設定（キャラクターバイブル）に基づいて、イレイナとしてのロールプレイの一貫性を高めるための、新しくオリジナルの質問・回答ペアを {n} 個作成してください（原作の丸写しではなく、日常会話や仮定の質問、意見を尋ねるものなど多様に作成）。
回答は必ずイレイナの性格、価値観、口調（一人称「私」、丁寧だが毒舌・ナルシストな語尾）、思い出に忠実である必要があります：

{profile}

各エントリーのフォーマット（必ず日本語で記述）：
{{"user_message": "日本語での質問やシチュエーション", "elaina_response": "イレイナの口調に合わせた日本語での返答"}}

{n} 個のエントリーを含むJSON配列のみを出力してください。
"""

SYSTEM_PROMPT_ID = """Kamu membantu menyiapkan data fine-tuning chatbot untuk karakter Elaina dari novel "Majo no Tabitabi / The Journey of Elaina".

Berdasarkan character bible di bawah, buat {n} pasangan tanya-jawab BARU dan ORISINAL dalam bahasa Indonesia untuk menguji roleplay Elaina:

{profile}

Format untuk setiap entri (dalam bahasa Indonesia):
{{"user_message": "pertanyaan/situasi dalam bahasa Indonesia", "elaina_response": "jawaban Elaina dalam bahasa Indonesia, sesuai gaya bicaranya"}}

Balas dengan array JSON yang berisi {n} entri tersebut. HANYA JSON.
"""


def get_system_prompt() -> str:
    if LANG == "JP":
        return SYSTEM_PROMPT_JP
    if LANG == "ID":
        return SYSTEM_PROMPT_ID
    return SYSTEM_PROMPT_EN


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A pairs from character profile")
    parser.add_argument("batches", nargs="?", type=int, default=10, help="Number of synthetic Q&A batches (default: 10)")
    parser.add_argument("--batches", dest="named_batches", type=int, default=None, help="Alternative flag for batch count")
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

    n_batches = args.named_batches if args.named_batches is not None else args.batches

    if not PROFILE_FILE.exists():
        raise SystemExit(f"{PROFILE_FILE} not found. Run persona_extractor.py --{LANG} first.")

    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    profile_str = json.dumps(profile, ensure_ascii=False)

    total = 0
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for _ in tqdm(range(n_batches), desc=f"Generating synthetic Q&A ({LANG})"):
            prompt = get_system_prompt().format(n=PAIRS_PER_BATCH, profile=profile_str)
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
