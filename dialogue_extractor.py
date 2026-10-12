"""
dialogue_extractor.py

Stage 3: Read each chapter, find scenes where Elaina speaks, and turn
them into raw conversation pairs: context/question -> Elaina's response.

Run after persona_extractor.py:
    python dialogue_extractor.py

Output:
    output/raw_dialogue_pairs.jsonl
    Each line: {"chapter": "...", "scenario": "...", "user_message": "...", "elaina_response": "..."}
"""

import json
import os
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json

TEXT_DIR = Path("text")
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "raw_dialogue_pairs.jsonl"

# How many characters of chapter text to process per call.
# Set to 0 to send the ENTIRE chapter in a single call (ideal for Qwen 3.6 / 262k context).
# Or set to a positive integer (e.g. 35000) to process in larger scene-preserving chunks.
DIALOGUE_CHUNK_CHARS = int(os.environ.get("DIALOGUE_CHUNK_CHARS", "35000"))

# Checkpoint: list of chapter names that are FULLY done (all their chunks
# processed). If the script stops mid-run, the next run resumes from the
# unfinished chapters -- without duplicating lines in raw_dialogue_pairs.jsonl.
CHECKPOINT_FILE = OUTPUT_DIR / "dialogue_checkpoint.json"


def load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return set(data.get("completed_chapters", []))
    except json.JSONDecodeError:
        print("[WARN] Checkpoint is corrupted/unreadable, starting from scratch.")
        return set()


def save_checkpoint(completed_chapters: set):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed_chapters": sorted(completed_chapters)}, ensure_ascii=False),
        encoding="utf-8",
    )

SYSTEM_PROMPT = """You are helping prepare chatbot fine-tuning data for the character Elaina, \
from the novel "Majo no Tabitabi / The Journey of Elaina".

From the given chapter excerpt, find ALL scenes where Elaina speaks (either \
direct dialogue or inner monologue that reveals her personality). For each \
scene, produce one entry with the structure:

{
  "scenario": "a brief summary (1-2 sentences) of this scene's situation/setting",
  "user_message": "the message/question from Elaina's conversation partner (or a narration of the situation if there's no direct interlocutor) that prompts Elaina's response, rewritten in natural English",
  "elaina_response": "Elaina's response in English, preserving her personality, tone, and speech style as accurately as possible"
}

Reply with a JSON array containing these entries. If there are no scenes of \
Elaina speaking in this excerpt, reply with an empty array []. DO NOT include \
scenes that don't involve Elaina speaking/reacting.

Reply with ONLY a JSON array, no other text, no markdown code fence.
"""


def main():
    chapter_files = sorted(TEXT_DIR.glob("chapter_*.txt"))
    if not chapter_files:
        raise SystemExit("No chapter files found in text/. Run extract_text.py first.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    completed = load_checkpoint()

    if completed:
        chapter_names = {p.name for p in chapter_files}
        stale = completed - chapter_names
        if stale:
            print(
                f"[WARN] Checkpoint references {len(stale)} chapter(s) that no longer "
                f"exist in text/ (the chapter list may have changed). Reprocessing ALL "
                f"chapters from scratch for consistency. Delete "
                f"output/dialogue_checkpoint.json and output/raw_dialogue_pairs.jsonl "
                f"first if this isn't what you want."
            )
            completed = set()

    remaining = [p for p in chapter_files if p.name not in completed]

    if completed:
        print(
            f"[INFO] Resuming from checkpoint: {len(completed)} chapter(s) already "
            f"completed previously, {len(remaining)} remaining."
        )
    if not remaining:
        print("[INFO] All chapters already processed (per checkpoint).")
        return

    # Append if resuming from a checkpoint (keep old lines), overwrite if
    # this is a fresh first run.
    file_mode = "a" if completed else "w"
    total_pairs = 0

    with open(OUTPUT_FILE, file_mode, encoding="utf-8") as out_f:
        for chap_path in tqdm(remaining, desc="Extracting Elaina's dialogue"):
            chapter_text = chap_path.read_text(encoding="utf-8")

            # process in chunks to stay within the context window of whichever model is used
            # If DIALOGUE_CHUNK_CHARS <= 0, the entire chapter is processed in 1 single call
            chunk_size = len(chapter_text) if DIALOGUE_CHUNK_CHARS <= 0 else DIALOGUE_CHUNK_CHARS
            chunks = [
                chapter_text[i : i + chunk_size]
                for i in range(0, len(chapter_text), chunk_size)
            ]

            # Buffer this chapter's results in memory first, and only write
            # to the file AFTER all of this chapter's chunks are done. This
            # matters because if this chapter fails partway (e.g. a daily
            # rate limit), no half-finished lines get written -- this
            # chapter will be retried in FULL on resume, with no risk of
            # duplicates.
            chapter_records = []

            for chunk in chunks:
                if len(chunk.strip()) < 200:
                    continue
                try:
                    pairs = chat_json(SYSTEM_PROMPT, chunk)
                except ValueError as e:
                    print(f"[WARN] Skipping a chunk in {chap_path.name}: {e}")
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

    print(f"\nDone. {total_pairs} NEW dialogue pair(s) added to {OUTPUT_FILE}")
    print(f"Total chapters completed: {len(completed)}/{len(chapter_files)}")


if __name__ == "__main__":
    main()
