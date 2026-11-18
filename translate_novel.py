"""
translate_novel.py

Translates Japanese datasets (output-JP/ -> output-ID/) and extracted novel chapters
(text-JP/ -> text-ID/) into Indonesian using Tencent Hy-MT2-30B-A3B served via
Kaggle/Colab llama-server (kaggle_llm_server.py).

Strict 4K Token Budgeting:
  - Input limit : 4096 tokens max per request (--max-input-tokens 4096)
  - Output limit: 4096 tokens max per request (--max-output-tokens 4096)
  - Dynamic token-aware batch packing + automatic binary batch splitting +
    oversized text chunking guarantee 0 truncated/cut-off translations.

Paired with:
    kaggle_llm_server.py  (llama-server on Kaggle dual T4 + Cloudflare Tunnel)

Usage examples:
    # 1. Translate full JP dataset to ID (output-JP/ -> output-ID/):
    py -3.14 translate_novel.py --mode dataset --base-url https://xxxx.trycloudflare.com

    # 2. Translate dialogue pairs only (output-JP/raw_dialogue_pairs.jsonl -> output-ID/):
    py -3.14 translate_novel.py --mode dialogue --base-url https://xxxx.trycloudflare.com

    # 3. Translate extracted novel chapters (text-JP/ -> text-ID/):
    py -3.14 translate_novel.py --mode text --base-url https://xxxx.trycloudflare.com

    # 4. Test translate a single chapter first:
    py -3.14 translate_novel.py --mode text --base-url https://xxxx.trycloudflare.com --chapter chapter_0001.txt

    # 5. Check connection to the remote translator server:
    py -3.14 translate_novel.py --test --base-url https://xxxx.trycloudflare.com

    # 6. Run offline unit self-tests (token budgeting, multi-line parser, chunking):
    py -3.14 translate_novel.py --self-test
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from tqdm import tqdm
import requests

DEFAULT_API_KEY = "sk-kaggle-local"

# Strict 4K token limits for both input prompt and output completion
MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 4096

# Reserved token margin for system/instruction wrapper & chat template markers
PROMPT_OVERHEAD_TOKENS = 180
OUTPUT_SAFETY_MARGIN = 128

# Japanese -> Indonesian token expansion factor (~1.3x-1.45x in BPE tokenizers)
JP_TO_ID_EXPANSION_RATIO = 1.45

NUM_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):、．]\s?(.*)$")
BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
CJK_CHAR_RE = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+")


# =====================================================================
# TOKEN ESTIMATION & 4K BUDGET CHUNKING
# =====================================================================

def estimate_tokens(text: str) -> int:
    """
    Conservative token estimator for Japanese + Indonesian/Latin text.
    - CJK characters count as ~1.2 tokens/char (safe upper bound for BPE tokenizers).
    - Non-CJK characters count as ~1 token per 3 chars.
    """
    if not text:
        return 0
    cjk_count = len(CJK_CHAR_RE.findall(text))
    non_cjk_count = len(text) - cjk_count
    newline_count = text.count("\n")
    estimated = math.ceil(cjk_count * 1.2) + math.ceil(non_cjk_count / 3.0) + (newline_count * 2)
    return max(1, estimated)


def compute_safe_input_budget(
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> int:
    """
    Compute the maximum input item tokens per batch so that:
      1. Total prompt tokens (items + instruction header) <= max_input_tokens (4096)
      2. Expected translated output tokens (input * expansion) <= max_output_tokens (4096)
    """
    max_by_input = max(256, max_input_tokens - PROMPT_OVERHEAD_TOKENS)
    max_by_output = max(256, int((max_output_tokens - OUTPUT_SAFETY_MARGIN) / JP_TO_ID_EXPANSION_RATIO))
    return min(max_by_input, max_by_output)


def split_oversized_text(
    text: str,
    max_chunk_tokens: int,
) -> tuple[list[str], str]:
    """
    Split a single oversized text into chunks that each fit within `max_chunk_tokens`
    without losing or truncating any characters.
    Returns (chunks, join_separator).
    """
    if estimate_tokens(text) <= max_chunk_tokens:
        return [text], ""

    # 1. Try splitting by double-newline paragraphs first
    if "\n\n" in text:
        parts = text.split("\n\n")
        if all(estimate_tokens(p) <= max_chunk_tokens for p in parts if p.strip()):
            return _merge_small_pieces(parts, "\n\n", max_chunk_tokens), "\n\n"

    # 2. Try splitting by single newline
    if "\n" in text:
        parts = text.split("\n")
        if all(estimate_tokens(p) <= max_chunk_tokens for p in parts if p.strip()):
            return _merge_small_pieces(parts, "\n", max_chunk_tokens), "\n"

    # 3. Split by sentence boundaries (Japanese 。！？ or Latin . ! ?)
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s]
    if len(sentences) > 1 and all(estimate_tokens(s) <= max_chunk_tokens for s in sentences):
        return _merge_small_pieces(sentences, " ", max_chunk_tokens), " "

    # 4. Fallback: hard split by character window when a single sentence has no punctuation
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for ch in text:
        ch_tok = 2 if CJK_CHAR_RE.match(ch) else 1
        if current and current_tokens + ch_tok > max_chunk_tokens:
            chunks.append("".join(current))
            current = [ch]
            current_tokens = ch_tok
        else:
            current.append(ch)
            current_tokens += ch_tok
    if current:
        chunks.append("".join(current))
    return chunks, ""


def _merge_small_pieces(pieces: list[str], sep: str, max_chunk_tokens: int) -> list[str]:
    """Greedily merge adjacent pieces using `sep` without exceeding `max_chunk_tokens`."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        candidate = f"{current}{sep}{piece}"
        if estimate_tokens(candidate) <= max_chunk_tokens:
            current = candidate
        else:
            merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def pack_items_by_tokens(
    items: list[str],
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    max_items: int = 25,
) -> list[list[str]]:
    """
    Pack a list of strings into sub-batches that strictly obey both:
      - Prompt token limit <= max_input_tokens (4096)
      - Expected translated output token limit <= max_output_tokens (4096)
      - Item count <= max_items
    """
    if not items:
        return []

    safe_budget = compute_safe_input_budget(max_input_tokens, max_output_tokens)
    hard_input_cap = max(256, max_input_tokens - PROMPT_OVERHEAD_TOKENS)

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for item in items:
        # Account for numbering prefix "N. " and newline("<br>") encoding overhead
        encoded_item = item.replace("\r\n", "\n").replace("\n", " <br> ")
        item_tokens = estimate_tokens(encoded_item) + 6

        if current_batch and (
            len(current_batch) >= max_items
            or current_tokens + item_tokens > safe_budget
            or estimate_tokens(build_translation_prompt(current_batch + [item])) > hard_input_cap
        ):
            batches.append(current_batch)
            current_batch = [item]
            current_tokens = item_tokens
        else:
            current_batch.append(item)
            current_tokens += item_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# =====================================================================
# PROMPT BUILDING & MULTI-LINE RESPONSE PARSER
# =====================================================================

def _encode_item_for_prompt(text: str) -> str:
    """Encode internal newlines as <br> so each numbered item stays on one logical line."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("\n", " <br> ")


def _decode_item_from_response(text: str) -> str:
    """Restore <br> markers back to real newlines."""
    restored = BR_TAG_RE.sub("\n", text)
    # Clean up optional spaces around restored newlines
    restored = re.sub(r"[ \t]*\n[ \t]*", "\n", restored)
    return restored.strip()


def parse_numbered_response(text: str, expected_count: int) -> list[str] | None:
    """
    Parse a numbered reply into a list of `expected_count` strings.
    Preserves multi-line continuation lines within any numbered item instead of dropping them,
    and restores `<br>` newline markers so multi-line dialogues/paragraphs are never truncated.
    """
    if not text or expected_count <= 0:
        return None

    result: dict[int, list[str]] = {}
    current_n: int | None = None

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = NUM_LINE_RE.match(line)
        if m:
            n = int(m.group(1))
            if 1 <= n <= expected_count and n not in result:
                current_n = n
                result[current_n] = [m.group(2).strip()]
                continue
            elif n in result and expected_count > 1:
                # Duplicate item number in a multi-item batch indicates malformed numbering
                return None

        # Continuation line belonging to the current numbered item
        if current_n is not None:
            result[current_n].append(line)

    if set(result.keys()) != set(range(1, expected_count + 1)):
        return None

    parsed_items: list[str] = []
    for i in range(1, expected_count + 1):
        joined = "\n".join(part for part in result[i] if part)
        decoded = _decode_item_from_response(joined)
        if not decoded:
            return None
        parsed_items.append(decoded)

    return parsed_items


def build_translation_prompt(items: list[str], target_lang: str = "Indonesian") -> str:
    """Build Hy-MT2 numbered batch prompt with <br> newline preservation."""
    body = "\n".join(f"{i + 1}. {_encode_item_for_prompt(text)}" for i, text in enumerate(items))
    return (
        f"Translate the following numbered light novel text into {target_lang}. "
        "Note that you should only output the translated result without any additional explanation:\n"
        "1. Keep the original number and period at the start of every line/paragraph; output exactly as many items as the input has.\n"
        "2. Translate each item independently — never merge, split, skip or add items.\n"
        "3. Faithfully preserve the tone, conversational register, and character voice of the source (especially for the traveler witch Elaina).\n"
        "4. If an item contains <br>, keep <br> in the corresponding position in your translation without breaking the numbered line.\n\n"
        f"{body}"
    )


# =====================================================================
# LLM CLIENT WITH 4K INPUT/OUTPUT ENFORCEMENT & TRUNCATION RECOVERY
# =====================================================================

def call_llm_chat_with_meta(
    base_url: str,
    api_key: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    retries: int = 5,
) -> tuple[str, str]:
    """
    Call OpenAI-compatible /v1/chat/completions with retry logic.
    Returns (content, finish_reason).
    Strictly caps max_tokens at MAX_OUTPUT_TOKENS (4096).
    """
    clean_url = base_url.rstrip("/")
    if not clean_url.endswith("/v1"):
        endpoint = f"{clean_url}/v1/chat/completions"
    else:
        endpoint = f"{clean_url}/chat/completions"

    capped_output_tokens = min(int(max_tokens), MAX_OUTPUT_TOKENS)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "Hy-MT2-30B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": capped_output_tokens,
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason") or "stop"
            return content, finish_reason
        except Exception as e:
            last_err = e
            if attempt < retries:
                wait_sec = min(3 * attempt, 30)
                time.sleep(wait_sec)
    raise RuntimeError(f"Translation request failed after {retries} retries: {last_err}")


def call_llm_chat(
    base_url: str,
    api_key: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    retries: int = 5,
) -> str:
    """Backward-compatible wrapper returning the response text."""
    content, _ = call_llm_chat_with_meta(
        base_url=base_url,
        api_key=api_key,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )
    return content


def _translate_single_oversized_item(
    item: str,
    base_url: str,
    api_key: str,
    target_lang: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> str:
    """
    Split an oversized single item into safe sub-chunks, translate each sub-chunk,
    and rejoin them with the original separator so nothing is ever cut off.
    """
    safe_budget = compute_safe_input_budget(max_input_tokens, max_output_tokens)
    chunks, sep = split_oversized_text(item, max_chunk_tokens=safe_budget)
    if len(chunks) <= 1:
        # Force smaller split if called due to finish_reason == "length"
        chunks, sep = split_oversized_text(item, max_chunk_tokens=max(128, estimate_tokens(item) // 2))

    translated_chunks = translate_items_batch(
        chunks,
        base_url=base_url,
        api_key=api_key,
        target_lang=target_lang,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_items=len(chunks),
    )
    return sep.join(translated_chunks)


def _translate_packed_subbatch(
    sub_items: list[str],
    base_url: str,
    api_key: str,
    target_lang: str,
    max_input_tokens: int,
    max_output_tokens: int,
) -> list[str]:
    """
    Translate a single token-packed sub-batch.
    If the response is truncated (`finish_reason == "length"`) or numbering fails:
      - If len(sub_items) > 1: recursively split the batch in half and translate both halves.
      - If len(sub_items) == 1: split the single item into smaller sentence/paragraph chunks
        if truncated, or retry with lower temperature so it is never cut off.
    """
    if not sub_items:
        return []

    safe_budget = compute_safe_input_budget(max_input_tokens, max_output_tokens)
    hard_input_cap = max(256, max_input_tokens - PROMPT_OVERHEAD_TOKENS)

    # If a single item exceeds the safe token budget, chunk it before sending
    if len(sub_items) == 1 and estimate_tokens(sub_items[0]) > safe_budget:
        return [
            _translate_single_oversized_item(
                sub_items[0],
                base_url=base_url,
                api_key=api_key,
                target_lang=target_lang,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            )
        ]

    prompt = build_translation_prompt(sub_items, target_lang)

    # Guard: if total prompt tokens exceed MAX_INPUT_TOKENS (4096), split immediately
    if estimate_tokens(prompt) > max_input_tokens:
        if len(sub_items) > 1:
            mid = len(sub_items) // 2
            left = _translate_packed_subbatch(
                sub_items[:mid], base_url, api_key, target_lang, max_input_tokens, max_output_tokens
            )
            right = _translate_packed_subbatch(
                sub_items[mid:], base_url, api_key, target_lang, max_input_tokens, max_output_tokens
            )
            return left + right
        else:
            return [
                _translate_single_oversized_item(
                    sub_items[0],
                    base_url=base_url,
                    api_key=api_key,
                    target_lang=target_lang,
                    max_input_tokens=max(512, hard_input_cap // 2),
                    max_output_tokens=max_output_tokens,
                )
            ]

    try:
        reply, finish_reason = call_llm_chat_with_meta(
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            temperature=0.7,
            max_tokens=max_output_tokens,
        )
        # Never accept a truncated output ("length") — split and re-translate cleanly
        if finish_reason != "length":
            parsed = parse_numbered_response(reply, len(sub_items))
            if parsed is not None:
                return parsed
    except Exception:
        finish_reason = "error"
        reply = ""

    # Binary split recovery when batch has multiple items
    if len(sub_items) > 1:
        mid = len(sub_items) // 2
        left = _translate_packed_subbatch(
            sub_items[:mid], base_url, api_key, target_lang, max_input_tokens, max_output_tokens
        )
        right = _translate_packed_subbatch(
            sub_items[mid:], base_url, api_key, target_lang, max_input_tokens, max_output_tokens
        )
        return left + right

    # Single-item recovery (len(sub_items) == 1)
    single_item = sub_items[0]
    if finish_reason == "length" and estimate_tokens(single_item) > 64:
        return [
            _translate_single_oversized_item(
                single_item,
                base_url=base_url,
                api_key=api_key,
                target_lang=target_lang,
                max_input_tokens=max(512, max_input_tokens // 2),
                max_output_tokens=max(512, max_output_tokens // 2),
            )
        ]

    # Retry single item with lower temperature
    single_prompt = build_translation_prompt([single_item], target_lang)
    try:
        single_reply, single_finish = call_llm_chat_with_meta(
            base_url=base_url,
            api_key=api_key,
            prompt=single_prompt,
            temperature=0.3,
            max_tokens=max_output_tokens,
        )
        if single_finish == "length" and estimate_tokens(single_item) > 64:
            return [
                _translate_single_oversized_item(
                    single_item,
                    base_url=base_url,
                    api_key=api_key,
                    target_lang=target_lang,
                    max_input_tokens=max(512, max_input_tokens // 2),
                    max_output_tokens=max(512, max_output_tokens // 2),
                )
            ]
        parsed_single = parse_numbered_response(single_reply, 1)
        if parsed_single:
            return [parsed_single[0]]
        cleaned = _decode_item_from_response(NUM_LINE_RE.sub(r"\2", single_reply.strip(), count=1))
        return [cleaned if cleaned else single_item]
    except Exception:
        return [single_item]


def translate_items_batch(
    items: list[str],
    base_url: str,
    api_key: str,
    target_lang: str = "Indonesian",
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    max_items: int = 25,
) -> list[str]:
    """
    Translate a list of strings while respecting the 4K input and 4K output token limits.
    - Empty/whitespace-only strings are bypassed and preserved in-place so they never
      confuse numbered batch alignment.
    - Non-empty strings are dynamically packed into token-safe sub-batches (`<= 4096` input
      tokens and `<= 4096` expected output tokens).
    """
    if not items:
        return []

    results = [""] * len(items)
    non_empty_indices: list[int] = []
    non_empty_items: list[str] = []

    for idx, text in enumerate(items):
        if text and text.strip():
            non_empty_indices.append(idx)
            non_empty_items.append(text.strip())

    if not non_empty_items:
        return results

    packed_batches = pack_items_by_tokens(
        non_empty_items,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_items=max_items,
    )

    translated_non_empty: list[str] = []
    for sub_batch in packed_batches:
        translated_sub = _translate_packed_subbatch(
            sub_batch,
            base_url=base_url,
            api_key=api_key,
            target_lang=target_lang,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        translated_non_empty.extend(translated_sub)

    for idx, trans_text in zip(non_empty_indices, translated_non_empty):
        results[idx] = trans_text

    return results


# =====================================================================
# NOVEL CHAPTER TRANSLATION (text-JP/ -> text-ID/)
# =====================================================================

def translate_chapter_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    batch_size: int = 25,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
):
    """Translate a novel chapter text file paragraph by paragraph within 4K token limits."""
    content = src_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Preserve header metadata (# Volume / # Chapter / # Source)
    header_lines = []
    body_lines = []
    in_header = True

    for line in lines:
        if in_header and (line.startswith("#") or not line.strip()):
            header_lines.append(line)
        else:
            in_header = False
            body_lines.append(line)

    body_text = "\n".join(body_lines)
    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]

    translated_paras = translate_items_batch(
        paragraphs,
        base_url=base_url,
        api_key=api_key,
        target_lang="Indonesian",
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_items=batch_size,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    header_block = "\n".join(header_lines).strip()
    if header_block:
        final_output = header_block + "\n\n" + "\n\n".join(translated_paras) + "\n"
    else:
        final_output = "\n\n".join(translated_paras) + "\n"

    # Write atomically via .tmp file so interrupted runs never leave half-translated chapters
    tmp_path = dst_path.with_suffix(".txt.tmp")
    tmp_path.write_text(final_output, encoding="utf-8")
    tmp_path.replace(dst_path)


# =====================================================================
# DATASET TRANSLATION PIPELINE (output-JP/ -> output-ID/)
# =====================================================================

def _load_valid_jsonl_records(path: Path) -> list[dict]:
    """Load valid JSONL lines from `path`, ignoring any trailing corrupted line."""
    if not path.exists():
        return []
    valid: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return valid


def translate_dialogue_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    batch_size: int = 20,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
):
    """Translate dialogue pairs in raw_dialogue_pairs.jsonl from JP to ID with 4K token limits."""
    if not src_path.exists():
        raise FileNotFoundError(f"Source file {src_path} not found.")

    records = _load_valid_jsonl_records(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict] = []
    if dst_path.exists() and not force:
        existing_records = _load_valid_jsonl_records(dst_path)
        # Rewrite clean prefix in case an interrupted run left a partial line
        with open(dst_path, "w", encoding="utf-8") as f:
            for rec in existing_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done_count = len(existing_records)
    remaining = records[done_count:]
    if not remaining:
        print(f"[INFO] {dst_path.name} is already fully translated ({done_count} pairs).")
        return

    print(
        f"[INFO] Translating {len(remaining)} dialogue pairs in {src_path.name} "
        f"({done_count} already completed, token limit: {max_input_tokens} in / {max_output_tokens} out)..."
    )
    file_mode = "a" if done_count > 0 else "w"

    with open(dst_path, file_mode, encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(remaining), batch_size), desc=f"Translating {src_path.name}"):
            batch = remaining[i : i + batch_size]
            user_msgs = [b.get("user_message", "") for b in batch]
            elaina_resps = [b.get("elaina_response", "") for b in batch]
            scenarios = [b.get("scenario", "") for b in batch]

            trans_users = translate_items_batch(
                user_msgs,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )
            trans_resps = translate_items_batch(
                elaina_resps,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )
            trans_scenarios = translate_items_batch(
                scenarios,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )

            for b, u, r, s in zip(batch, trans_users, trans_resps, trans_scenarios):
                new_record = dict(b)
                new_record["chapter"] = b.get("chapter", "")
                new_record["scenario"] = s
                new_record["user_message"] = u
                new_record["elaina_response"] = r
                out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            out_f.flush()


def translate_synthetic_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    batch_size: int = 20,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
):
    """Translate synthetic_pairs.jsonl from JP to ID with 4K token limits."""
    if not src_path.exists():
        return

    records = _load_valid_jsonl_records(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict] = []
    if dst_path.exists() and not force:
        existing_records = _load_valid_jsonl_records(dst_path)
        with open(dst_path, "w", encoding="utf-8") as f:
            for rec in existing_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done_count = len(existing_records)
    remaining = records[done_count:]
    if not remaining:
        print(f"[INFO] {dst_path.name} is already fully translated ({done_count} pairs).")
        return

    print(f"[INFO] Translating {len(remaining)} synthetic pairs in {src_path.name} ({done_count} already completed)...")
    file_mode = "a" if done_count > 0 else "w"

    with open(dst_path, file_mode, encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(remaining), batch_size), desc=f"Translating {src_path.name}"):
            batch = remaining[i : i + batch_size]
            user_msgs = [b.get("user_message", "") for b in batch]
            elaina_resps = [b.get("elaina_response", "") for b in batch]

            trans_users = translate_items_batch(
                user_msgs,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )
            trans_resps = translate_items_batch(
                elaina_resps,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )

            for b, u, r in zip(batch, trans_users, trans_resps):
                new_record = dict(b)
                new_record["user_message"] = u
                new_record["elaina_response"] = r
                out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            out_f.flush()


def translate_system_prompt_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
):
    """Translate elaina_system_prompt.txt paragraph by paragraph within 4K token limits."""
    if not src_path.exists():
        return
    if dst_path.exists() and not force and dst_path.stat().st_size > 0:
        print(f"[INFO] {dst_path.name} already exists — skipping (use --force to overwrite).")
        return

    print(f"[INFO] Translating system prompt {src_path.name} -> {dst_path}...")
    raw_text = src_path.read_text(encoding="utf-8").strip()
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    translated_paras = translate_items_batch(
        paragraphs,
        base_url=base_url,
        api_key=api_key,
        target_lang="Indonesian",
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_items=15,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n\n".join(translated_paras) + "\n", encoding="utf-8")
    print(f"[OK] Saved translated system prompt to {dst_path}")


def translate_profile_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
):
    """Translate elaina_profile.json fields into Indonesian within 4K token limits."""
    if not src_path.exists():
        return
    if dst_path.exists() and not force and dst_path.stat().st_size > 0:
        print(f"[INFO] {dst_path.name} already exists — skipping (use --force to overwrite).")
        return

    print(f"[INFO] Translating character profile {src_path.name} -> {dst_path}...")
    profile = json.loads(src_path.read_text(encoding="utf-8"))
    translated_profile = dict(profile)

    for list_key in ("personality_traits", "speech_patterns", "values_and_worldview"):
        items = profile.get(list_key)
        if isinstance(items, list) and items:
            translated_profile[list_key] = translate_items_batch(
                [str(x) for x in items],
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            )

    memories = profile.get("key_memories")
    if isinstance(memories, list) and memories:
        events = [str(m.get("event", "")) if isinstance(m, dict) else str(m) for m in memories]
        impacts = [str(m.get("emotional_impact", "")) if isinstance(m, dict) else "" for m in memories]
        trans_events = translate_items_batch(
            events, base_url=base_url, api_key=api_key, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens
        )
        trans_impacts = translate_items_batch(
            impacts, base_url=base_url, api_key=api_key, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens
        )
        new_memories = []
        for m, ev, imp in zip(memories, trans_events, trans_impacts):
            if isinstance(m, dict):
                nm = dict(m)
                nm["event"] = ev
                if "emotional_impact" in nm:
                    nm["emotional_impact"] = imp
                new_memories.append(nm)
            else:
                new_memories.append(ev)
        translated_profile["key_memories"] = new_memories

    relationships = profile.get("relationships")
    if isinstance(relationships, list) and relationships:
        descs = [str(r.get("description", "")) if isinstance(r, dict) else str(r) for r in relationships]
        trans_descs = translate_items_batch(
            descs, base_url=base_url, api_key=api_key, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens
        )
        new_rels = []
        for r, d in zip(relationships, trans_descs):
            if isinstance(r, dict):
                nr = dict(r)
                nr["description"] = d
                new_rels.append(nr)
            else:
                new_rels.append(d)
        translated_profile["relationships"] = new_rels

    for str_key in ("physical_description", "notes"):
        val = profile.get(str_key)
        if isinstance(val, str) and val.strip():
            translated_profile[str_key] = translate_items_batch(
                [val],
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            )[0]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(translated_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Saved translated profile to {dst_path}")


def translate_finetune_dataset_file(
    src_path: Path,
    dst_path: Path,
    base_url: str,
    api_key: str,
    batch_size: int = 20,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
    translated_system_prompt: str | None = None,
):
    """
    Translate elaina_finetune_dataset.jsonl (messages format) from JP to ID
    with 4K input/output token limits and checkpoint/resume support.
    Deduplicates identical system prompts so the system prompt is only translated once.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source finetune dataset {src_path} not found.")

    records = _load_valid_jsonl_records(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict] = []
    if dst_path.exists() and not force:
        existing_records = _load_valid_jsonl_records(dst_path)
        with open(dst_path, "w", encoding="utf-8") as f:
            for rec in existing_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done_count = len(existing_records)
    remaining = records[done_count:]
    if not remaining:
        print(f"[INFO] {dst_path.name} is already fully translated ({done_count} conversations).")
        return

    print(
        f"[INFO] Translating {len(remaining)} conversations in {src_path.name} "
        f"({done_count} already completed, token limit: {max_input_tokens} in / {max_output_tokens} out)..."
    )

    # Cache translated system prompts to avoid re-translating the same system prompt thousands of times
    system_prompt_cache: dict[str, str] = {}

    file_mode = "a" if done_count > 0 else "w"
    with open(dst_path, file_mode, encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(remaining), batch_size), desc=f"Translating {src_path.name}"):
            batch = remaining[i : i + batch_size]

            # 1. Translate any unseen system prompts once
            for rec in batch:
                for msg in rec.get("messages", []):
                    if msg.get("role") == "system":
                        sys_text = msg.get("content", "")
                        if sys_text and sys_text not in system_prompt_cache:
                            if translated_system_prompt:
                                system_prompt_cache[sys_text] = translated_system_prompt
                            else:
                                sys_paras = [p.strip() for p in sys_text.split("\n\n") if p.strip()]
                                trans_sys_paras = translate_items_batch(
                                    sys_paras,
                                    base_url=base_url,
                                    api_key=api_key,
                                    max_input_tokens=max_input_tokens,
                                    max_output_tokens=max_output_tokens,
                                )
                                system_prompt_cache[sys_text] = "\n\n".join(trans_sys_paras)

            # 2. Collect all non-system messages and metadata scenarios across the batch
            non_sys_texts: list[str] = []
            non_sys_locations: list[tuple[int, int]] = []
            scenarios: list[str] = []

            for b_idx, rec in enumerate(batch):
                for m_idx, msg in enumerate(rec.get("messages", [])):
                    if msg.get("role") != "system":
                        non_sys_texts.append(msg.get("content", ""))
                        non_sys_locations.append((b_idx, m_idx))
                meta = rec.get("metadata") or {}
                scenarios.append(meta.get("scenario", "") if isinstance(meta, dict) else "")

            trans_non_sys = translate_items_batch(
                non_sys_texts,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size * 2,
            )
            trans_scenarios = translate_items_batch(
                scenarios,
                base_url=base_url,
                api_key=api_key,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_items=batch_size,
            )

            # Map translated messages back into each record
            new_batch = json.loads(json.dumps(batch))
            for (b_idx, m_idx), trans_text in zip(non_sys_locations, trans_non_sys):
                new_batch[b_idx]["messages"][m_idx]["content"] = trans_text

            for b_idx, rec in enumerate(new_batch):
                for msg in rec.get("messages", []):
                    if msg.get("role") == "system":
                        orig_sys = msg.get("content", "")
                        if orig_sys in system_prompt_cache:
                            msg["content"] = system_prompt_cache[orig_sys]
                if isinstance(rec.get("metadata"), dict) and "scenario" in rec["metadata"]:
                    rec["metadata"]["scenario"] = trans_scenarios[b_idx]
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()


def assemble_finetune_from_translated_parts(
    dst_dir: Path,
    force: bool = False,
) -> bool:
    """
    If output-ID/ already has translated raw_dialogue_pairs.jsonl (and/or synthetic_pairs.jsonl)
    plus elaina_system_prompt.txt, assemble output-ID/elaina_finetune_dataset.jsonl directly
    so we don't pay twice to translate the exact same pairs.
    """
    finetune_dst = dst_dir / "elaina_finetune_dataset.jsonl"
    raw_dst = dst_dir / "raw_dialogue_pairs.jsonl"
    synth_dst = dst_dir / "synthetic_pairs.jsonl"
    sys_prompt_dst = dst_dir / "elaina_system_prompt.txt"

    if not raw_dst.exists() and not synth_dst.exists():
        return False

    if finetune_dst.exists() and not force and finetune_dst.stat().st_size > 0:
        return True

    default_id_prompt = (
        "Kamu adalah Elaina, Penyihir Abu (Witch of Ashen), seorang penyihir pengembara muda "
        "yang cerdas, percaya diri, sedikit narsis, menyukai uang dan roti, serta selalu mengamati "
        "dunia dengan rasa ingin tahu namun tetap menjaga jarak sebagai seorang musafir."
    )
    system_prompt = (
        sys_prompt_dst.read_text(encoding="utf-8").strip()
        if sys_prompt_dst.exists()
        else default_id_prompt
    )

    records_out: list[dict] = []
    if raw_dst.exists():
        for item in _load_valid_jsonl_records(raw_dst):
            u = item.get("user_message", "").strip()
            r = item.get("elaina_response", "").strip()
            if u and r:
                records_out.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": u},
                            {"role": "assistant", "content": r},
                        ],
                        "metadata": {
                            "source": "novel",
                            "chapter": item.get("chapter", ""),
                            "scenario": item.get("scenario", ""),
                        },
                    }
                )

    if synth_dst.exists():
        for item in _load_valid_jsonl_records(synth_dst):
            u = item.get("user_message", "").strip()
            r = item.get("elaina_response", "").strip()
            if u and r:
                records_out.append(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": u},
                            {"role": "assistant", "content": r},
                        ],
                        "metadata": {
                            "source": "synthetic",
                            "category": item.get("category", "general"),
                            "scenario": item.get("scenario_type", ""),
                        },
                    }
                )

    if not records_out:
        return False

    with open(finetune_dst, "w", encoding="utf-8") as f:
        for rec in records_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Assembled {len(records_out)} translated conversations into {finetune_dst}")
    return True


def translate_dataset_directory(
    src_dir: Path,
    dst_dir: Path,
    base_url: str,
    api_key: str,
    batch_size: int = 20,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    force: bool = False,
):
    """Translate all JP dataset artifacts in `src_dir` (output-JP/) to `dst_dir` (output-ID/)."""
    if not src_dir.exists():
        print(f"[ERROR] Source dataset folder {src_dir}/ not found.")
        print("        Generate the JP dataset first (e.g., py -3.14 run_all.py --JP --skip-extract)")
        print("        or run chapter translation with: py -3.14 translate_novel.py --mode text --base-url <URL>")
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    found_any = False

    # 1. System prompt
    sys_src = src_dir / "elaina_system_prompt.txt"
    sys_dst = dst_dir / "elaina_system_prompt.txt"
    if sys_src.exists():
        found_any = True
        translate_system_prompt_file(
            sys_src,
            sys_dst,
            base_url=base_url,
            api_key=api_key,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=force,
        )

    # 2. Character profile JSON
    prof_src = src_dir / "elaina_profile.json"
    prof_dst = dst_dir / "elaina_profile.json"
    if prof_src.exists():
        found_any = True
        translate_profile_file(
            prof_src,
            prof_dst,
            base_url=base_url,
            api_key=api_key,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=force,
        )

    # 3. Raw dialogue pairs
    raw_src = src_dir / "raw_dialogue_pairs.jsonl"
    raw_dst = dst_dir / "raw_dialogue_pairs.jsonl"
    if raw_src.exists():
        found_any = True
        translate_dialogue_file(
            raw_src,
            raw_dst,
            base_url=base_url,
            api_key=api_key,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=force,
        )

    # 4. Synthetic dialogue pairs
    synth_src = src_dir / "synthetic_pairs.jsonl"
    synth_dst = dst_dir / "synthetic_pairs.jsonl"
    if synth_src.exists():
        found_any = True
        translate_synthetic_file(
            synth_src,
            synth_dst,
            base_url=base_url,
            api_key=api_key,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=force,
        )

    # 5. Final finetune dataset
    ft_src = src_dir / "elaina_finetune_dataset.jsonl"
    ft_dst = dst_dir / "elaina_finetune_dataset.jsonl"
    if raw_dst.exists() or synth_dst.exists():
        assembled = assemble_finetune_from_translated_parts(dst_dir, force=force)
        found_any = found_any or assembled
    elif ft_src.exists():
        found_any = True
        trans_sys = sys_dst.read_text(encoding="utf-8").strip() if sys_dst.exists() else None
        translate_finetune_dataset_file(
            ft_src,
            ft_dst,
            base_url=base_url,
            api_key=api_key,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=force,
            translated_system_prompt=trans_sys,
        )

    if not found_any:
        print(f"[ERROR] No dataset files found in {src_dir}/.")
        sys.exit(1)

    print(f"\n[DONE] Dataset translation complete! Translated dataset saved to {dst_dir}/")


# =====================================================================
# CONNECTION TEST & SELF-TEST SUITE
# =====================================================================

def test_connection(base_url: str, api_key: str) -> bool:
    """Test connection to Kaggle/Colab llama-server."""
    print(f"Testing server endpoint at: {base_url}")
    print(f"Token budget configured  : {MAX_INPUT_TOKENS} input tokens / {MAX_OUTPUT_TOKENS} output tokens")
    prompt = build_translation_prompt(["私、旅の魔女イレイナです。\nどうぞよしなに。"], "Indonesian")
    try:
        reply, finish_reason = call_llm_chat_with_meta(base_url, api_key, prompt, temperature=0.7)
        parsed = parse_numbered_response(reply, 1)
        res = parsed[0] if parsed else reply.strip()
        print("\n[OK] Connection successful!")
        print(f"  Finish reason : {finish_reason}")
        print("  Input         : 私、旅の魔女イレイナです。 / どうぞよしなに。")
        print(f"  Output        : {res}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        return False


def run_self_test() -> bool:
    """Verify 4K token budgeting, oversized splitting, and multi-line response parsing."""
    print("Running translate_novel.py self-tests...")

    # 1. Multi-line & <br> response parsing test
    sample_reply = (
        "1. Halo, namaku Elaina. <br> Aku adalah seorang penyihir pengembara.\n"
        "2. \"Benarkah itu?\"\n"
        "\"Tentu saja, aku tidak pernah berbohong.\"\n"
        "3. Perjalanan ini masih sangat panjang."
    )
    parsed = parse_numbered_response(sample_reply, 3)
    assert parsed is not None and len(parsed) == 3, f"Expected 3 parsed items, got {parsed}"
    assert "\n" in parsed[0], "Expected <br> to be decoded back to newline in item 1"
    assert "\"Tentu saja, aku tidak pernah berbohong.\"" in parsed[1], "Expected continuation line in item 2 to be preserved"

    # 2. 4K token packing test
    synthetic_jp_items = ["私、旅の魔女イレイナです。灰の魔女と呼ばれています。" * 8 for _ in range(40)]
    batches = pack_items_by_tokens(
        synthetic_jp_items,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_items=25,
    )
    assert len(batches) > 1, "Expected 40 long JP items to be split across multiple 4K-safe batches"
    for b in batches:
        prompt_tok = estimate_tokens(build_translation_prompt(b))
        assert prompt_tok <= MAX_INPUT_TOKENS, f"Batch prompt tokens {prompt_tok} exceeded {MAX_INPUT_TOKENS}"
        est_out = int(sum(estimate_tokens(x) for x in b) * JP_TO_ID_EXPANSION_RATIO)
        assert est_out <= MAX_OUTPUT_TOKENS, f"Estimated output tokens {est_out} exceeded {MAX_OUTPUT_TOKENS}"

    # 3. Oversized single-item splitting & lossless reassembly test
    long_item = "\n\n".join(["これは非常に長い段落です。旅の魔女イレイナは様々な国を訪れました。" * 15 for _ in range(12)])
    safe_budget = compute_safe_input_budget(MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)
    chunks, sep = split_oversized_text(long_item, max_chunk_tokens=safe_budget)
    assert len(chunks) > 1, "Expected oversized item to be split into multiple chunks"
    for c in chunks:
        assert estimate_tokens(c) <= safe_budget, f"Chunk token count {estimate_tokens(c)} exceeded safe budget {safe_budget}"
    assert sep.join(chunks) == long_item, "Rejoined chunks must match original text without any character loss"

    print("[OK] All translate_novel.py self-tests passed! (4K input / 4K output / 0 truncation verified)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Dataset and Novel Translator using Hy-MT2-30B-A3B (JP -> ID) with strict 4K token budgeting"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of Cloudflare tunnel (e.g. https://xxx.trycloudflare.com)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key (default: sk-kaggle-local or env LLM_API_KEY)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "dataset", "dialogue", "finetune", "text", "all"],
        default="auto",
        help=(
            "Translation mode: "
            "'dataset' (translate output-JP/ -> output-ID/), "
            "'dialogue' (translate raw_dialogue_pairs.jsonl only), "
            "'finetune' (translate elaina_finetune_dataset.jsonl only), "
            "'text' (translate text-JP/ -> text-ID/), "
            "'all' (translate both text-JP/ and output-JP/), "
            "or 'auto' (default: dataset if output-JP/ has files, else text)"
        ),
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Optional custom source directory (defaults to output-JP or text-JP based on mode)",
    )
    parser.add_argument(
        "--dst-dir",
        default=None,
        help="Optional custom destination directory (defaults to output-ID or text-ID based on mode)",
    )
    parser.add_argument(
        "--chapter",
        default=None,
        help="Optional: translate only a specific chapter file in text mode (e.g. chapter_0001.txt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Max number of items per batch before token cap (default: 25)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"Maximum input prompt tokens per request (default: {MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help=f"Maximum output completion tokens per request (default: {MAX_OUTPUT_TOKENS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing translated files instead of resuming",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke test remote translator server without doing full translation",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline unit tests for 4K token budgeting and multi-line parser",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    # Enforce hard ceiling of 4096 tokens so user/CLI never exceeds model's 4K limit
    max_input_tokens = min(max(512, args.max_input_tokens), MAX_INPUT_TOKENS)
    max_output_tokens = min(max(512, args.max_output_tokens), MAX_OUTPUT_TOKENS)

    base_url = args.base_url or os.environ.get("TRANSLATE_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    api_key = args.api_key or os.environ.get("LLM_API_KEY", DEFAULT_API_KEY)

    if not base_url:
        print("[ERROR] Missing base URL. Run kaggle_llm_server.py first, then pass --base-url <printed URL>")
        sys.exit(1)

    if args.test:
        test_connection(base_url, api_key)
        return

    # Resolve 'auto' mode intelligently
    mode = args.mode
    if mode == "auto":
        if args.chapter:
            mode = "text"
        else:
            out_jp = Path(args.src_dir) if args.src_dir else Path("output-JP")
            has_jp_dataset = out_jp.exists() and any(
                (out_jp / fname).exists()
                for fname in (
                    "raw_dialogue_pairs.jsonl",
                    "elaina_finetune_dataset.jsonl",
                    "synthetic_pairs.jsonl",
                    "elaina_system_prompt.txt",
                    "elaina_profile.json",
                )
            )
            mode = "dataset" if has_jp_dataset else "text"
            print(f"[AUTO MODE] Selected '--mode {mode}' (token limit: {max_input_tokens} in / {max_output_tokens} out)")

    if mode in ("text", "all"):
        src_dir = Path(args.src_dir) if (args.src_dir and mode == "text") else Path("text-JP")
        dst_dir = Path(args.dst_dir) if (args.dst_dir and mode == "text") else Path("text-ID")

        if not src_dir.exists():
            print(f"[ERROR] Source folder {src_dir} not found. Extract Japanese EPUB first: py -3.14 extract_text.py --JP")
            sys.exit(1)

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy manifest.json if present so downstream tools have chapter metadata
        src_manifest = src_dir / "manifest.json"
        dst_manifest = dst_dir / "manifest.json"
        if src_manifest.exists() and (not dst_manifest.exists() or args.force):
            shutil.copy2(src_manifest, dst_manifest)

        if args.chapter:
            chapter_files = [src_dir / args.chapter]
            if not chapter_files[0].exists():
                print(f"[ERROR] {chapter_files[0]} not found in {src_dir}/")
                sys.exit(1)
        else:
            chapter_files = sorted(src_dir.glob("chapter_*.txt"))

        if not chapter_files:
            print(f"[ERROR] No chapter files found in {src_dir}/")
            sys.exit(1)

        print(
            f"[TRANSLATE] Translating {len(chapter_files)} chapter(s) from {src_dir}/ to {dst_dir}/ "
            f"(max {max_input_tokens} input tokens, {max_output_tokens} output tokens)..."
        )

        for chap_path in tqdm(chapter_files, desc="Translating chapters"):
            out_path = dst_dir / chap_path.name
            if out_path.exists() and not args.force and out_path.stat().st_size > 0:
                continue
            translate_chapter_file(
                chap_path,
                out_path,
                base_url=base_url,
                api_key=api_key,
                batch_size=args.batch_size,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            )

        print(f"\n[DONE] Text translation complete! Translated novel chapters saved to {dst_dir}/")
        if mode == "text":
            print("Next step: You can build the Indonesian dataset with:")
            print("    py -3.14 run_all.py --ID --skip-extract")

    if mode == "dataset" or (mode == "all" and Path("output-JP").exists()):
        src_dir = Path(args.src_dir) if (args.src_dir and mode == "dataset") else Path("output-JP")
        dst_dir = Path(args.dst_dir) if (args.dst_dir and mode == "dataset") else Path("output-ID")
        translate_dataset_directory(
            src_dir=src_dir,
            dst_dir=dst_dir,
            base_url=base_url,
            api_key=api_key,
            batch_size=min(args.batch_size, 20),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=args.force,
        )

    elif mode == "dialogue":
        src_dir = Path(args.src_dir) if args.src_dir else Path("output-JP")
        dst_dir = Path(args.dst_dir) if args.dst_dir else Path("output-ID")
        src_file = src_dir / "raw_dialogue_pairs.jsonl"
        dst_file = dst_dir / "raw_dialogue_pairs.jsonl"

        if not src_file.exists():
            print(f"[ERROR] Source dialogue file {src_file} not found. Run dialogue_extractor.py --JP first.")
            sys.exit(1)

        translate_dialogue_file(
            src_file,
            dst_file,
            base_url=base_url,
            api_key=api_key,
            batch_size=min(args.batch_size, 20),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=args.force,
        )
        print(f"\n[DONE] Dialogue translation complete! Saved to {dst_file}")

    elif mode == "finetune":
        src_dir = Path(args.src_dir) if args.src_dir else Path("output-JP")
        dst_dir = Path(args.dst_dir) if args.dst_dir else Path("output-ID")
        src_file = src_dir / "elaina_finetune_dataset.jsonl"
        dst_file = dst_dir / "elaina_finetune_dataset.jsonl"

        if not src_file.exists():
            print(f"[ERROR] Source finetune dataset {src_file} not found. Run build_dataset.py --JP first.")
            sys.exit(1)

        translate_finetune_dataset_file(
            src_file,
            dst_file,
            base_url=base_url,
            api_key=api_key,
            batch_size=min(args.batch_size, 20),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            force=args.force,
        )
        print(f"\n[DONE] Finetune dataset translation complete! Saved to {dst_file}")


if __name__ == "__main__":
    main()
