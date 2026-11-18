#!/usr/bin/env python3
"""
prepare_dataset.py

Build a leakage-free, language-aware SFT dataset for the Elaina persona from the
raw extraction output, in the same `{"messages": [system, user, assistant]}`
schema that output/elaina_finetune_dataset.jsonl uses.

Why this exists (build_dataset.py is not enough):
  * every record there repeats the full 8.8 KB system prompt (~97% of all tokens)
  * there is no train/val split, and `chapter` is dropped, so a split built from
    that file would leak novel text across train and val
  * user turns mix verbatim dialogue with third-person narration and nothing
    marks which is which
  * the dataset is going multi-language (EN / ID / JP) and needs one stable slot
    per language rather than a rewrite later

Outputs (under --out):
    train.jsonl / val.jsonl          {"messages": [...]}  <- feed straight to TRL
    train_meta.jsonl / val_meta.jsonl  parallel sidecar, one object per line:
                                      {lang, chapter, kind, prompt, source}
    manifest.json                    counts, split chapters, params, output hashes
    stats.json                       length distributions, token estimates

Usage:
    python training/prepare_dataset.py --sources-file training/sources.json --check
    python training/prepare_dataset.py --sources-file training/sources.json

    # or spell the slots out by hand
    python training/prepare_dataset.py \
        --source en=output/raw_dialogue_pairs.jsonl:output/synthetic_pairs.jsonl \
        --source id=training/data_id/raw_dialogue_pairs.jsonl

A missing source file is a warning, not an error: the ID/JP slots stay empty
until their files exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# defaults (mirror build_dataset.py where the semantics are the same)
# ---------------------------------------------------------------------------

MIN_RESPONSE_CHARS = 8      # build_dataset.py used 5; 8 keeps "Shoot." but drops noise
MAX_RESPONSE_CHARS = 1500
MIN_USER_CHARS = 2
MAX_USER_CHARS = 800
VAL_FRAC = 0.06
SEED = 42
FUZZY_THRESHOLD = 92        # SequenceMatcher ratio * 100
LONG_PROMPT_FRAC = 0.10
MIN_TOTAL_VAL = 100
MIN_VAL_PER_LANG = 12
SYNTHETIC_BLOCK = 50        # synthetic rows per pseudo-chapter
SCENE_PREFIX = "[Scene] "

QUOTE_CHARS = "\"'`“”‘’«»「」『』"
NARRATION_OPEN = re.compile(r"^(She|He|The|A|An|Elaina|It|They|Someone|Her|His)\b")
NARRATION_VERB = re.compile(
    r"\b(explains?|says?|asks?|tells?|notices?|sees?|watches?|wonders?|"
    r"recalls?|remembers?|thinks?|replies?|answers?|adds?|murmurs?|whispers?)\b",
    re.IGNORECASE,
)
NARRATION_OPEN_ID = re.compile(
    r"^(Dia|Ia|Elaina|Mereka|Seseorang|Sang|Seorang|Gadis|Penyihir|Tiba-tiba|Kemudian|Saat|Ketika)\b",
    re.IGNORECASE,
)
NARRATION_VERB_ID = re.compile(
    r"\b(menjelaskan|berkata|bertanya|memberitahu|melihat|memperhatikan|menatap|"
    r"berpikir|mengingat|menjawab|membalas|bergumam|berbisik|tersenyum|menghela)\b",
    re.IGNORECASE,
)
NARRATION_JP = re.compile(
    r"(イレイナは|彼女は|少女は|魔女の|言いました|尋ねました|思いました|見ました|答えました|呟きました|頷きました)"
)
QUOTE_SEGMENT = re.compile(r"[" + re.escape(QUOTE_CHARS) + r"][^" + re.escape(QUOTE_CHARS) + r"]{2,}")


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Same forgiving contract as build_dataset.py: missing file -> [], bad lines skipped."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Dedup key only. NFKC + casefold + punctuation folded to spaces (keeps CJK)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def classify_user_turn(text: str, lang: str = "en") -> str:
    """'dialogue' or 'scene'. Conservative: only clear narration is flagged (EN/ID/JP)."""
    s = text.strip()
    if not s:
        return "dialogue"
    if s[0] in QUOTE_CHARS or s[-1] in QUOTE_CHARS:
        return "dialogue"
    if QUOTE_SEGMENT.search(s):
        return "dialogue"
    if lang == "id":
        if NARRATION_OPEN_ID.match(s) or NARRATION_VERB_ID.search(s):
            return "scene"
    elif lang == "jp":
        if "「" not in s and "『" not in s and NARRATION_JP.search(s):
            return "scene"
    if NARRATION_OPEN.match(s) or NARRATION_VERB.search(s):
        return "scene"
    return "dialogue"


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# record model
# ---------------------------------------------------------------------------

@dataclass
class Record:
    lang: str
    chapter: str
    kind: str           # "dialogue" | "scene" | "synthetic"
    user: str
    response: str
    source: str         # file the pair came from
    index: int          # position within that file

    @property
    def is_synthetic(self) -> bool:
        return self.kind == "synthetic"

    def pair_key(self) -> str:
        return f"{self.lang}\x00{normalize(self.user)}\x00{normalize(self.response)}"

    def response_key(self) -> str:
        return f"{self.lang}\x00{normalize(self.response)}"


def source_specs(args) -> list[tuple[str, list[Path]]]:
    """Collect (lang, files) slots from --sources-file and/or repeated --source."""
    slots: dict[str, list[Path]] = {}

    if args.sources_file:
        cfg_path = Path(args.sources_file)
        if not cfg_path.exists():
            raise SystemExit(f"--sources-file not found: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for entry in cfg.get("languages", []):
            lang = entry.get("lang")
            if not lang:
                continue
            if entry.get("status", "ready") != "ready":
                print(f"[INFO] Skipping '{lang}' slot (status={entry.get('status')})")
                continue
            slots.setdefault(lang, []).extend(Path(p) for p in entry.get("files", []))

    for spec in args.source or []:
        if "=" not in spec:
            raise SystemExit(f"--source needs lang=path1,path2 — got: {spec}")
        lang, raw_paths = spec.split("=", 1)
        paths = [Path(p) for p in raw_paths.split(",") if p.strip()]
        slots.setdefault(lang.strip(), []).extend(paths)

    if not slots:
        raise SystemExit("No sources. Pass --sources-file or at least one --source lang=path")

    return sorted(slots.items())


# ---------------------------------------------------------------------------
# stage: load
# ---------------------------------------------------------------------------

def load_records(slots: list[tuple[str, list[Path]]], fallback_out_dir: Path | None = None) -> tuple[list[Record], dict]:
    records: list[Record] = []
    report = {"slots": {}, "missing_files": [], "dropped_malformed": 0}

    for lang, paths in slots:
        kept_lang = 0
        per_file = {}
        for path in paths:
            if not path.exists():
                report["missing_files"].append(str(path))
                per_file[str(path)] = 0
                print(f"[WARN] {lang}: source not found, skipping -> {path}")
                continue

            is_synthetic = "synthetic" in path.name.lower()
            rows = load_jsonl(path)
            kept = 0
            for i, row in enumerate(rows):
                user = str(row.get("user_message", "")).strip()
                response = str(row.get("elaina_response", "")).strip()
                if not user or not response:
                    report["dropped_malformed"] += 1
                    continue

                if is_synthetic:
                    chapter = f"synthetic_{lang}_{i // SYNTHETIC_BLOCK:03d}"
                    kind = "synthetic"
                else:
                    chapter = str(row.get("chapter") or "").strip()
                    if not chapter:
                        chapter = f"unknown_{lang}_{i // SYNTHETIC_BLOCK:03d}"
                    kind = classify_user_turn(user, lang=lang)

                records.append(
                    Record(
                        lang=lang,
                        chapter=chapter,
                        kind=kind,
                        user=user,
                        response=response,
                        source=str(path),
                        index=i,
                    )
                )
                kept += 1
            per_file[str(path)] = kept
            kept_lang += kept

        report["slots"][lang] = {"files": per_file, "loaded": kept_lang}
        print(f"[LOAD] {lang}: {kept_lang} pair(s) from {len(paths)} file(s)")

    if not records and fallback_out_dir is not None:
        tr_p = fallback_out_dir / "train.jsonl"
        tr_m = fallback_out_dir / "train_meta.jsonl"
        va_p = fallback_out_dir / "val.jsonl"
        va_m = fallback_out_dir / "val_meta.jsonl"
        if tr_p.exists() and tr_m.exists() and va_p.exists() and va_m.exists():
            print(f"[INFO] Raw source files missing; reconstructing records from existing {fallback_out_dir}/")
            idx = 0
            for msg_path, meta_path in ((tr_p, tr_m), (va_p, va_m)):
                msgs_list = load_jsonl(msg_path)
                meta_list = load_jsonl(meta_path)
                for m_row, meta_row in zip(msgs_list, meta_list):
                    msgs = m_row.get("messages", [])
                    if len(msgs) < 3:
                        continue
                    user_txt = str(msgs[1].get("content", "")).strip()
                    if user_txt.startswith(SCENE_PREFIX):
                        user_txt = user_txt[len(SCENE_PREFIX):]
                    resp_txt = str(msgs[2].get("content", "")).strip()
                    lang = str(meta_row.get("lang", "en"))
                    chapter = str(meta_row.get("chapter", f"unknown_{lang}_000"))
                    kind = str(meta_row.get("kind") or classify_user_turn(user_txt, lang=lang))
                    src = str(meta_row.get("source", str(msg_path)))
                    records.append(
                        Record(
                            lang=lang,
                            chapter=chapter,
                            kind=kind,
                            user=user_txt,
                            response=resp_txt,
                            source=src,
                            index=idx,
                        )
                    )
                    idx += 1
            report["reconstructed_from_out"] = len(records)

    return records, report


# ---------------------------------------------------------------------------
# stage: filter
# ---------------------------------------------------------------------------

def length_filter(records: list[Record], args) -> tuple[list[Record], Counter]:
    dropped = Counter()
    kept = []
    for r in records:
        if not r.user.strip() or not r.response.strip():
            dropped["empty"] += 1
            continue
        if not (args.min_response_chars <= len(r.response) <= args.max_response_chars):
            dropped["response_length"] += 1
            continue
        if not (args.min_user_chars <= len(r.user) <= args.max_user_chars):
            dropped["user_length"] += 1
            continue
        if args.drop_scene and r.kind == "scene":
            dropped["scene"] += 1
            continue
        kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------------------
# stage: dedup
# ---------------------------------------------------------------------------

def _preference(r: Record):
    """Sort key for which duplicate to keep. Lower sorts first (kept)."""
    return (r.is_synthetic, -len(r.response), r.index)


def dedup_exact(records: list[Record]) -> tuple[list[Record], Counter]:
    """Drop duplicates on the user+response pair and on the response alone.

    Keys are language-scoped, so a Japanese "Hmm?" never cancels an English one.
    """
    dropped = Counter()
    seen_pairs: set[str] = set()
    seen_responses: set[str] = set()
    kept: list[Record] = []

    for r in sorted(records, key=_preference):
        pair = r.pair_key()
        resp = r.response_key()
        if pair in seen_pairs:
            dropped["exact_pair"] += 1
            continue
        if resp in seen_responses:
            dropped["exact_response"] += 1
            continue
        seen_pairs.add(pair)
        seen_responses.add(resp)
        kept.append(r)

    kept.sort(key=lambda r: (r.lang, r.source, r.index))
    return kept, dropped


def _bucket_key(r: Record) -> tuple:
    words = normalize(r.response).split()
    return (r.lang, " ".join(words[:3]), len(r.response) // 40)


def dedup_fuzzy(
    records: list[Record], threshold: int, max_bucket: int = 200
) -> tuple[list[Record], Counter]:
    """Near-duplicate removal without new dependencies.

    Candidates are bucketed by the first three normalized words plus a coarse
    length band, so only a handful of pairs are compared per record.
    """
    dropped = Counter()
    buckets: dict[tuple, list[Record]] = defaultdict(list)
    for r in sorted(records, key=_preference):
        buckets[_bucket_key(r)].append(r)

    kept: list[Record] = []
    for group in buckets.values():
        survivors: list[Record] = []
        if len(group) > max_bucket:
            # Pathological bucket: keep it linear rather than O(n^2).
            kept.extend(group)
            continue
        for candidate in group:
            norm = normalize(candidate.response)
            duplicate = False
            for other in survivors:
                ratio = SequenceMatcher(None, norm, normalize(other.response)).ratio() * 100
                if ratio >= threshold:
                    duplicate = True
                    break
            if duplicate:
                dropped["fuzzy"] += 1
            else:
                survivors.append(candidate)
        kept.extend(survivors)

    kept.sort(key=lambda r: (r.lang, r.source, r.index))
    return kept, dropped


# ---------------------------------------------------------------------------
# stage: split
# ---------------------------------------------------------------------------

def chapter_split(
    records: list[Record], val_frac: float, seed: int
) -> tuple[list[Record], list[Record], dict]:
    """Hold out whole chapters globally across all languages so no chapter leaks across languages."""
    by_lang: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_lang[r.lang].append(r)

    # 1. Global chapter pool across all languages (excluding synthetic pseudo-chapters)
    non_synth_chapters = sorted({r.chapter for r in records if not r.is_synthetic})
    chapter_counts = Counter(r.chapter for r in records if not r.is_synthetic)

    rng = random.Random(seed)
    shuffled = list(non_synth_chapters)
    rng.shuffle(shuffled)

    target_rows = max(int(round(val_frac * len(records))), MIN_TOTAL_VAL)
    target_rows = min(target_rows, max(0, len(records) - len(shuffled)))

    picked_global: list[str] = []
    picked_rows = 0
    for ch in shuffled:
        if picked_rows >= target_rows or len(picked_global) >= max(1, len(shuffled) - 1):
            break
        picked_global.append(ch)
        picked_rows += chapter_counts[ch]

    if not picked_global and shuffled:
        picked_global.append(shuffled[0])

    global_val_chapters = set(picked_global)

    # 2. Assign rows based on global chapter membership
    train_rows, val_rows = [], []
    for r in records:
        if not r.is_synthetic and r.chapter in global_val_chapters:
            val_rows.append(r)
        else:
            train_rows.append(r)

    # 3. Record per-language chapter lists for manifest compatibility
    val_chapters: dict[str, list[str]] = {}
    train_chapters: dict[str, list[str]] = {}
    for lang in sorted(by_lang):
        val_chapters[lang] = sorted({r.chapter for r in val_rows if r.lang == lang})
        train_chapters[lang] = sorted({r.chapter for r in train_rows if r.lang == lang})

    # 4. Assert leakage-free ACROSS ALL LANGUAGES
    all_train_chs = {r.chapter for r in train_rows if not r.is_synthetic}
    all_val_chs = {r.chapter for r in val_rows if not r.is_synthetic}
    cross_overlap = all_train_chs & all_val_chs
    if cross_overlap:
        raise AssertionError(
            f"FATAL CROSS-LINGUAL LEAK: Bab berikut muncul di train & val lintas bahasa: {sorted(cross_overlap)[:5]}"
        )

    plan = {"train_chapters": train_chapters, "val_chapters": val_chapters}
    return train_rows, val_rows, plan


# ---------------------------------------------------------------------------
# stage: assemble
# ---------------------------------------------------------------------------

def assemble(
    train_rows: list[Record],
    val_rows: list[Record],
    short_prompt: str,
    long_prompt: str | None,
    long_prompt_frac: float,
    seed: int,
    short_prompts_by_lang: dict[str, str] | None = None,
    multi_turn_frac: float = 0.0,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    prompts_map = short_prompts_by_lang or {}

    def render(rows: list[Record], allow_long: bool, allow_multiturn: bool) -> tuple[list[dict], list[dict]]:
        rng = random.Random(f"{seed}:prompt")
        msgs, meta = [], []
        i = 0
        n = len(rows)
        while i < n:
            r = rows[i]
            base_short = prompts_map.get(r.lang, short_prompt)
            use_long = bool(allow_long and long_prompt and r.lang == "en" and rng.random() < long_prompt_frac)
            system = long_prompt if use_long else base_short

            # Optional multi-turn grouping of 2-3 consecutive dialogue rows from the same chapter & language
            if (
                allow_multiturn
                and multi_turn_frac > 0
                and not r.is_synthetic
                and r.kind == "dialogue"
                and i + 1 < n
                and rows[i + 1].chapter == r.chapter
                and rows[i + 1].lang == r.lang
                and rows[i + 1].kind == "dialogue"
                and rng.random() < multi_turn_frac
            ):
                turns_count = 3 if (
                    i + 2 < n
                    and rows[i + 2].chapter == r.chapter
                    and rows[i + 2].lang == r.lang
                    and rows[i + 2].kind == "dialogue"
                    and rng.random() < 0.4
                ) else 2
                chunk = rows[i : i + turns_count]
                conv = [{"role": "system", "content": system}]
                for cr in chunk:
                    u_txt = SCENE_PREFIX + cr.user if cr.kind == "scene" else cr.user
                    conv.append({"role": "user", "content": u_txt})
                    conv.append({"role": "assistant", "content": cr.response})
                msgs.append({"messages": conv})
                meta.append(
                    {
                        "lang": r.lang,
                        "chapter": r.chapter,
                        "kind": "multiturn",
                        "turns": turns_count,
                        "prompt": "long" if use_long else "short",
                        "source": r.source,
                    }
                )
                i += turns_count
                continue

            user = SCENE_PREFIX + r.user if r.kind == "scene" else r.user
            msgs.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": r.response},
                    ]
                }
            )
            meta.append(
                {
                    "lang": r.lang,
                    "chapter": r.chapter,
                    "kind": r.kind,
                    "prompt": "long" if use_long else "short",
                    "source": r.source,
                }
            )
            i += 1
        return msgs, meta

    train_msgs, train_meta = render(train_rows, allow_long=True, allow_multiturn=True)
    val_msgs, val_meta = render(val_rows, allow_long=False, allow_multiturn=False)
    return train_msgs, train_meta, val_msgs, val_meta


# ---------------------------------------------------------------------------
# stage: stats
# ---------------------------------------------------------------------------

def token_stats(records_msgs: list[dict], meta: list[dict], tokenizer=None) -> dict:
    user_lens = [
        sum(len(t["content"]) for t in m["messages"] if t.get("role") == "user")
        for m in records_msgs
    ]
    resp_lens = [
        sum(len(t["content"]) for t in m["messages"] if t.get("role") == "assistant")
        for m in records_msgs
    ]
    sys_lens = [len(m["messages"][0]["content"]) for m in records_msgs]

    stats = {
        "rows": len(records_msgs),
        "user_chars": {"mean": _mean(user_lens), "p50": pct(user_lens, 0.5),
                       "p95": pct(user_lens, 0.95), "max": max(user_lens, default=0)},
        "response_chars": {"mean": _mean(resp_lens), "p50": pct(resp_lens, 0.5),
                           "p95": pct(resp_lens, 0.95), "max": max(resp_lens, default=0)},
        "system_chars": {"short_rows": sum(1 for n in sys_lens if n < 2000),
                         "long_rows": sum(1 for n in sys_lens if n >= 2000)},
        "by_lang": dict(Counter(m["lang"] for m in meta)),
        "by_kind": dict(Counter(m["kind"] for m in meta)),
        "estimate_tokens_chars_over_3_6": int(
            sum(sum(len(t["content"]) for t in m["messages"]) for m in records_msgs) / 3.6
        ),
    }

    if tokenizer is not None:
        try:
            total = 0
            for m in records_msgs:
                rendered = tokenizer.apply_chat_template(
                    m["messages"], tokenize=True, add_generation_prompt=False
                )
                if isinstance(rendered, dict) and "input_ids" in rendered:
                    rendered = rendered["input_ids"]
                elif hasattr(rendered, "input_ids"):
                    rendered = rendered.input_ids
                total += len(rendered)
            stats["exact_tokens"] = total
        except Exception as exc:  # template quirks must never fail the build
            stats["exact_tokens_error"] = f"{type(exc).__name__}: {exc}"

    return stats


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


# ---------------------------------------------------------------------------
# stage: verification
# ---------------------------------------------------------------------------

def verify_outputs(
    train_msgs: list[dict],
    val_msgs: list[dict],
    train_meta: list[dict],
    val_meta: list[dict],
    short_prompt: str,
    long_prompt: str | None,
    long_prompt_frac: float,
) -> list[str]:
    """Return a list of failure strings. Empty list == everything passed."""
    failures: list[str] = []

    if len(train_msgs) != len(train_meta) or len(val_msgs) != len(val_meta):
        failures.append("meta sidecar length does not match messages length")

    for name, msgs in (("train", train_msgs), ("val", val_msgs)):
        for i, row in enumerate(msgs):
            m = row.get("messages")
            if not isinstance(m, list) or len(m) < 3 or (len(m) - 1) % 2 != 0:
                failures.append(f"{name}[{i}]: 'messages' must have system + N*(user, assistant) turns")
                break
            roles = [t.get("role") for t in m]
            expected_roles = ["system"] + ["user", "assistant"] * ((len(m) - 1) // 2)
            if roles != expected_roles:
                failures.append(f"{name}[{i}]: wrong role order {roles}")
                break
            if any(not t.get("content", "").strip() for t in m if t.get("role") == "assistant"):
                failures.append(f"{name}[{i}]: empty assistant turn")
                break

    if not (250 <= len(short_prompt) <= 800):
        failures.append(f"short prompt length {len(short_prompt)} outside 250..800")
    if "Elaina" not in short_prompt or "Ashen Witch" not in short_prompt:
        failures.append("short prompt must mention 'Elaina' and 'Ashen Witch'")

    scene_ok = all(
        row["messages"][1]["content"].startswith(SCENE_PREFIX)
        for row, meta in zip(train_msgs + val_msgs, train_meta + val_meta)
        if meta["kind"] == "scene"
    )
    if not scene_ok:
        failures.append("a 'scene' row is missing the [Scene] prefix")

    if val_meta and any(m["prompt"] == "long" for m in val_meta):
        failures.append("val must use the short prompt only")

    if long_prompt and long_prompt_frac > 0 and train_meta:
        long_share = sum(1 for m in train_meta if m["prompt"] == "long") / len(train_meta)
        if not (long_prompt_frac * 0.5 <= long_share <= long_prompt_frac * 1.5):
            failures.append(
                f"long-prompt share in train is {long_share:.3f}, expected ~{long_prompt_frac:.3f}"
            )

    if len(val_msgs) < MIN_TOTAL_VAL and (len(train_msgs) + len(val_msgs)) >= MIN_TOTAL_VAL * 2:
        failures.append(f"val has only {len(val_msgs)} rows (floor is {MIN_TOTAL_VAL})")

    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare the Elaina SFT dataset (train/val + metadata).")
    p.add_argument("--source", action="append", metavar="LANG=PATH[,PATH]",
                   help="language slot, repeatable, e.g. en=output/raw_dialogue_pairs.jsonl")
    p.add_argument("--sources-file", metavar="JSON",
                   help="JSON config listing language slots (see training/sources.json)")
    p.add_argument("--short-prompt", default="training/elaina_system_prompt_short.txt")
    p.add_argument("--long-prompt", default="output/elaina_system_prompt.txt")
    p.add_argument("--out", default="training/out")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-response-chars", type=int, default=MIN_RESPONSE_CHARS)
    p.add_argument("--max-response-chars", type=int, default=MAX_RESPONSE_CHARS)
    p.add_argument("--min-user-chars", type=int, default=MIN_USER_CHARS)
    p.add_argument("--max-user-chars", type=int, default=MAX_USER_CHARS)
    p.add_argument("--fuzzy-threshold", type=int, default=FUZZY_THRESHOLD,
                   help="SequenceMatcher ratio*100 above which responses count as duplicates (0 disables)")
    p.add_argument("--long-prompt-frac", type=float, default=LONG_PROMPT_FRAC,
                   help="share of TRAIN rows that use the full system prompt (0 disables)")
    p.add_argument("--multi-turn-frac", type=float, default=0.12,
                   help="probability of grouping consecutive same-chapter dialogue rows into 2-3 turns (0 disables)")
    p.add_argument("--drop-scene", action="store_true",
                   help="drop third-person narration rows instead of prefixing them with [Scene]")
    p.add_argument("--tokenizer", metavar="MODEL_ID",
                   help="optional: count exact tokens with this tokenizer (needs transformers)")
    p.add_argument("--check", action="store_true",
                   help="build in memory, run every assertion, write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    short_path = Path(args.short_prompt)
    if not short_path.exists():
        raise SystemExit(f"short system prompt not found: {short_path}")
    short_prompt = short_path.read_text(encoding="utf-8").strip()

    short_prompts_by_lang = {"en": short_prompt}
    for _lc in ("id", "jp"):
        _lp = short_path.with_name(f"elaina_system_prompt_short_{_lc}.txt")
        if _lp.exists():
            short_prompts_by_lang[_lc] = _lp.read_text(encoding="utf-8").strip()

    long_path = Path(args.long_prompt)
    long_prompt = long_path.read_text(encoding="utf-8").strip() if long_path.exists() else None
    if long_prompt is None and args.long_prompt_frac > 0:
        print(f"[WARN] long prompt not found ({long_path}); every row will use the short prompt")

    slots = source_specs(args)
    print(f"[INFO] slots: {', '.join(lang for lang, _ in slots)}")

    out_dir = Path(args.out)
    records, load_report = load_records(slots, fallback_out_dir=out_dir if args.check else None)
    if not records:
        raise SystemExit("No usable pairs found. Check your --source paths.")

    filtered, dropped_len = length_filter(records, args)
    print(f"[FILTER] kept {len(filtered)} / {len(records)} (dropped {dict(dropped_len) or '{}'})")

    exact, dropped_exact = dedup_exact(filtered)
    print(f"[DEDUP] exact: kept {len(exact)} (dropped {dict(dropped_exact) or '{}'})")

    if args.fuzzy_threshold > 0:
        deduped, dropped_fuzzy = dedup_fuzzy(exact, args.fuzzy_threshold)
        print(f"[DEDUP] fuzzy: kept {len(deduped)} (dropped {dict(dropped_fuzzy) or '{}'})")
    else:
        deduped, dropped_fuzzy = exact, Counter()

    train_rows, val_rows, plan = chapter_split(deduped, args.val_frac, args.seed)
    print(f"[SPLIT] train {len(train_rows)} rows / val {len(val_rows)} rows "
          f"(by chapter; val chapters: {sum(len(v) for v in plan['val_chapters'].values())})")

    train_msgs, train_meta, val_msgs, val_meta = assemble(
        train_rows, val_rows, short_prompt, long_prompt, args.long_prompt_frac, args.seed,
        short_prompts_by_lang=short_prompts_by_lang, multi_turn_frac=args.multi_turn_frac,
    )

    failures = verify_outputs(train_msgs, val_msgs, train_meta, val_meta,
                              short_prompt, long_prompt, args.long_prompt_frac)
    if failures:
        print("\n[FAIL] dataset checks failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[CHECK] all structural assertions passed")

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f"[TOKENS] counting with {args.tokenizer}")
        except Exception as exc:
            print(f"[WARN] could not load tokenizer ({type(exc).__name__}: {exc}); "
                  f"falling back to a character estimate")

    stats = {
        "train": token_stats(train_msgs, train_meta, tokenizer),
        "val": token_stats(val_msgs, val_meta, tokenizer),
    }

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "params": {
            "min_response_chars": args.min_response_chars,
            "max_response_chars": args.max_response_chars,
            "min_user_chars": args.min_user_chars,
            "max_user_chars": args.max_user_chars,
            "fuzzy_threshold": args.fuzzy_threshold,
            "long_prompt_frac": args.long_prompt_frac,
            "drop_scene": args.drop_scene,
            "scene_prefix": SCENE_PREFIX,
        },
        "prompts": {
            "short_chars": len(short_prompt),
            "long_chars": len(long_prompt) if long_prompt else 0,
        },
        "counts": {
            "loaded": len(records),
            "after_length_filter": len(filtered),
            "after_exact_dedup": len(exact),
            "after_fuzzy_dedup": len(deduped),
            "train": len(train_msgs),
            "val": len(val_msgs),
            "by_lang": stats["train"].get("by_lang", {}),
            "dropped": {
                "length": dict(dropped_len),
                "exact": dict(dropped_exact),
                "fuzzy": dict(dropped_fuzzy),
            },
        },
        "chapters": plan,
        "load_report": load_report,
        "files": {},
    }

    out_dir = Path(args.out)
    if args.check:
        print("\n[CHECK] dry run — nothing written")
        print(json.dumps({"train": stats["train"], "val": stats["val"]}, indent=2, ensure_ascii=False))
        return 0

    write_jsonl(out_dir / "train.jsonl", train_msgs)
    write_jsonl(out_dir / "val.jsonl", val_msgs)
    write_jsonl(out_dir / "train_meta.jsonl", train_meta)
    write_jsonl(out_dir / "val_meta.jsonl", val_meta)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    for name in ("train.jsonl", "val.jsonl", "train_meta.jsonl", "val_meta.jsonl"):
        path = out_dir / name
        manifest["files"][name] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[OK] wrote {out_dir}/")
    print(f"     train {len(train_msgs)} rows | val {len(val_msgs)} rows")
    print(f"     by language: {stats['train'].get('by_lang', {})}")
    print(f"     next: python training/push_dataset_to_hub.py --dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())