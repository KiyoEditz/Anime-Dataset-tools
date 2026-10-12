"""
persona_extractor.py

Stage 2: Incrementally build Elaina's "character bible" (a rolling
profile), reading chapter by chapter and updating the profile whenever
new information appears (personality, key memories/events,
relationships with other characters, speech patterns, values/worldview).

Run after extract_text.py:
    python persona_extractor.py

Output:
    output/elaina_profile.json      -> structured profile (can be audited manually)
    output/elaina_system_prompt.txt -> ready-to-use system prompt for fine-tuning
"""

import json
import os
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json, chat

TEXT_DIR = Path("text")
OUTPUT_DIR = Path("output")

# How many characters of chapter text to send per call.
# Set to 0 to send the ENTIRE chapter at once (recommended for large context models
# like Qwen 3.6 / Qwen 2.5 with 128k-262k context window on Colab A100 or Kaggle T4x2).
# Or set to a positive integer (e.g. 50000) to cap chapter length.
PERSONA_CHUNK_CHARS = int(os.environ.get("PERSONA_CHUNK_CHARS", "0"))

# The JSON profile keeps accumulating each chapter and is sent IN FULL on
# every call. For large context models (262k context), this can safely grow
# up to 50k+ chars before needing compression.
MAX_PROFILE_CHARS = int(os.environ.get("MAX_PROFILE_CHARS", "40000"))

# Checkpoint: stores the profile + the name of the last chapter completed,
# so if the script stops mid-run (error, daily rate limit, power outage,
# etc.), the next run RESUMES from where it left off instead of
# reprocessing every chapter from scratch (which wastes API quota).
CHECKPOINT_FILE = OUTPUT_DIR / "persona_checkpoint.json"

PROFILE_SCHEMA_HINT = """
The JSON format you MUST produce (a single object):
{
  "personality_traits": ["..."],          // list of traits, merge/update existing ones
  "speech_patterns": ["..."],              // distinctive way of speaking, favorite phrases, tone
  "values_and_worldview": ["..."],         // life principles, attitude toward the world/others
  "key_memories": [                        // important events she experienced/remembers
    {"event": "...", "emotional_impact": "..."}
  ],
  "relationships": [
    {"name": "...", "description": "..."}
  ],
  "physical_description": "...",
  "notes": "..."                           // anything else relevant that doesn't fit above
}
"""

SYSTEM_PROMPT = f"""You are a literary research assistant helping to build a "character bible" \
for the character Elaina from the novel "Majo no Tabitabi / The Journey of Elaina".

Your task: read the following chapter excerpt, then UPDATE the existing character \
profile (if there's new information) with details from this chapter. Do not delete \
still-valid old information; merge/tidy up if there's duplication or contradiction \
(use the more specific/recent version). Focus ONLY on the character Elaina (not \
other characters), except for the "relationships" section, which should describe \
Elaina's relationship with that character.

{PROFILE_SCHEMA_HINT}

Reply with ONLY JSON, no opening/closing text, no markdown code fence.
"""


COMPRESSION_SYSTEM_PROMPT = """You are helping to compress Elaina's character bible (from the novel \
'Majo no Tabitabi / The Journey of Elaina') that has grown long, WITHOUT losing \
important information. Merge similar or redundant traits/memories into a single, \
more concise entry, drop pure duplicates, but PRESERVE every event/memory and trait \
that has distinct meaning. Keep the exact same JSON schema structure as the input.

Reply with ONLY JSON, no other text.
"""


def compress_profile(profile: dict) -> dict:
    """Compress the profile if it has grown too large, so it doesn't balloon into the context window."""
    current_size = len(json.dumps(profile, ensure_ascii=False))
    if current_size <= MAX_PROFILE_CHARS:
        return profile

    try:
        compressed = chat_json(
            COMPRESSION_SYSTEM_PROMPT,
            json.dumps(profile, ensure_ascii=False),
            temperature=0.2,
        )
        if isinstance(compressed, dict):
            new_size = len(json.dumps(compressed, ensure_ascii=False))
            print(f"[INFO] Profile compressed: {current_size} -> {new_size} characters")
            return compressed
    except ValueError as e:
        print(f"[WARN] Failed to compress profile ({e}), keeping the previous version.")

    return profile


def load_checkpoint():
    """Return (last_completed_chapter_name, profile) from the checkpoint, or (None, {}) if none exists."""
    if not CHECKPOINT_FILE.exists():
        return None, {}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return data.get("last_completed_chapter"), data.get("profile", {})
    except json.JSONDecodeError:
        print("[WARN] Checkpoint is corrupted/unreadable, starting from scratch.")
        return None, {}


def save_checkpoint(last_completed_chapter: str, profile: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {"last_completed_chapter": last_completed_chapter, "profile": profile},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def merge_profile(existing: dict, update: dict) -> dict:
    """Merge the existing profile with one chapter's update."""
    if not existing:
        return update

    merged = dict(existing)
    for list_key in ("personality_traits", "speech_patterns", "values_and_worldview"):
        merged[list_key] = list(
            dict.fromkeys(existing.get(list_key, []) + update.get(list_key, []))
        )

    merged["key_memories"] = existing.get("key_memories", []) + update.get("key_memories", [])

    # relationships: merge by name, most recent description wins
    rel_map = {r["name"]: r for r in existing.get("relationships", [])}
    for r in update.get("relationships", []):
        rel_map[r["name"]] = r
    merged["relationships"] = list(rel_map.values())

    merged["physical_description"] = (
        update.get("physical_description") or existing.get("physical_description", "")
    )
    merged["notes"] = (existing.get("notes", "") + "\n" + update.get("notes", "")).strip()

    return merged


def build_profile():
    chapter_files = sorted(TEXT_DIR.glob("chapter_*.txt"))
    if not chapter_files:
        raise SystemExit("No chapter files found in text/. Run extract_text.py first.")

    last_completed, profile = load_checkpoint()

    if last_completed:
        chapter_names = [p.name for p in chapter_files]
        if last_completed in chapter_names:
            resume_idx = chapter_names.index(last_completed)
            skipped = chapter_files[: resume_idx + 1]
            chapter_files = chapter_files[resume_idx + 1 :]
            print(
                f"[INFO] Resuming from checkpoint: {len(skipped)} chapter(s) already "
                f"completed previously (last: {last_completed}), "
                f"{len(chapter_files)} remaining."
            )
        else:
            print(
                f"[WARN] Checkpoint references '{last_completed}' but that file no "
                f"longer exists in text/ (the chapter list may have changed due to "
                f"adding/editing a volume). Reprocessing ALL chapters from scratch "
                f"for consistency. Delete output/persona_checkpoint.json first if "
                f"this isn't what you want."
            )
            profile = {}

    if not chapter_files:
        print("[INFO] All chapters already processed (per checkpoint).")

    for chap_path in tqdm(chapter_files, desc="Building Elaina's profile"):
        chapter_text = chap_path.read_text(encoding="utf-8")

        # If PERSONA_CHUNK_CHARS <= 0, send the full chapter text
        chapter_content = (
            chapter_text
            if PERSONA_CHUNK_CHARS <= 0 or PERSONA_CHUNK_CHARS >= len(chapter_text)
            else chapter_text[:PERSONA_CHUNK_CHARS]
        )

        user_prompt = (
            "Profile built so far (JSON, may be empty if this is the first chapter):\n"
            f"{json.dumps(profile, ensure_ascii=False)}\n\n"
            "Next chapter text:\n"
            f"{chapter_content}"
        )

        try:
            update = chat_json(SYSTEM_PROMPT, user_prompt)
            if isinstance(update, dict):
                profile = merge_profile(profile, update)
                profile = compress_profile(profile)
        except ValueError as e:
            print(f"[WARN] Skipping {chap_path.name}: {e}")

        # Save the checkpoint AFTER every chapter (whether it succeeded or was
        # skipped due to a ValueError) so that if the NEXT chapter fails with
        # an unhandled error (e.g. a daily rate limit), progress up to this
        # chapter stays safe.
        save_checkpoint(chap_path.name, profile)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "elaina_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print("Profile saved to output/elaina_profile.json")
    return profile


def write_system_prompt(profile: dict):
    """Ask the model to turn the JSON profile into a ready-to-use system prompt."""
    summarizer_prompt = (
        "Here is a structured character bible for Elaina from "
        "'Majo no Tabitabi / The Journey of Elaina', extracted from the novel:\n\n"
        f"{json.dumps(profile, indent=2, ensure_ascii=False)}\n\n"
        "Rewrite this into ONE system prompt (3-6 paragraphs, in English) that "
        "instructs an AI to consistently roleplay as Elaina: her personality, "
        "her speech style, the values she holds, and the important "
        "memories/background she should remember. Write it from the "
        "perspective of instructions TO the AI (\"You are Elaina...\"), not "
        "from Elaina's own point of view."
    )
    system_prompt_text = chat(
        "You are an assistant skilled at writing detailed, consistent AI character system prompts.",
        summarizer_prompt,
        temperature=0.3,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "elaina_system_prompt.txt").write_text(system_prompt_text.strip(), encoding="utf-8")
    print("System prompt saved to output/elaina_system_prompt.txt")


if __name__ == "__main__":
    final_profile = build_profile()
    write_system_prompt(final_profile)
