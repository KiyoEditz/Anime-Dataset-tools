"""
llm_client.py

Thin wrapper around the `openai` SDK, pointed at ANY OpenAI-compatible
endpoint you choose -- your own LiteLLM (Ollama + Hermes-4-14B via
Cloudflare Tunnel), NVIDIA Build/NIM, Groq, OpenRouter, Gemini, etc. No
code needs to change to switch providers -- just change the environment
variables below.

Configure via environment variables (put them in a .env file or export manually):

    LLM_BASE_URL   Endpoint URL, must end with /v1
    LLM_API_KEY    API key for your chosen provider
    LLM_MODEL      Model name as expected by that provider

Rate limiting (optional, provider-dependent):

    LLM_MIN_SECONDS_BETWEEN_CALLS   Minimum seconds between calls (default 0).
                                     Useful for providers with a strict
                                     requests-per-minute (RPM) cap.
    LLM_TPM_LIMIT                   Tokens-per-minute budget (default 0 =
                                     disabled). Set this if your provider
                                     enforces a TPM cap (e.g. Groq free tier:
                                     12,000 TPM on some models). The client
                                     proactively paces requests to stay under
                                     this budget using a sliding 60-second
                                     window, instead of only reacting to 429s.
    LLM_MAX_OUTPUT_TOKENS           Cap on generated tokens per call (default
                                     2048). Also used to reserve TPM budget
                                     before the real usage is known. Raise
                                     this if you see JSON getting truncated
                                     on large profile-compression calls.
    LLM_MAX_AUTO_SLEEP_SECONDS      Longest wait the client will sleep
                                     through automatically for a 429 (default
                                     300s = 5 min). Longer waits (e.g. a daily
                                     quota reset) raise a clear error instead
                                     of blocking for hours.

Note: small/local models (e.g. 14B) can sometimes be inconsistent about
returning "pure JSON with no extra text". The `chat_json` function below
handles this with JSON block extraction + automatic retries.
"""

import json
import os
import re
import time
from collections import deque

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4000/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-local")
MODEL = os.environ.get("LLM_MODEL", "hermes-4-14b")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ── Per-call spacing (requests-per-minute style limits) ────────────────────
_MIN_INTERVAL = float(os.environ.get("LLM_MIN_SECONDS_BETWEEN_CALLS", "0"))
_last_call_time = [0.0]


def _respect_call_interval():
    if _MIN_INTERVAL <= 0:
        return
    elapsed = time.time() - _last_call_time[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time[0] = time.time()


# ── Tokens-per-minute (TPM) budget ──────────────────────────────────────────
# Some free providers (e.g. Groq) cap total TOKENS per minute, separately
# from requests-per-minute or tokens-per-day. This is a proactive sliding-
# window limiter: before each call, it reserves an estimate of the tokens
# this call will use (prompt + a reserved output allowance), sleeping just
# long enough to stay under LLM_TPM_LIMIT. After the call returns, the
# reservation is corrected with the provider's REAL reported usage, so the
# tracking stays accurate over time instead of drifting from rough estimates.
TPM_LIMIT = int(os.environ.get("LLM_TPM_LIMIT", "0"))
MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "2048"))
_token_window = deque()  # each entry: {"time": float, "tokens": int}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for English text (~4 characters per token)."""
    return max(1, len(text) // 4)


def _reserve_token_budget(estimated_tokens: int):
    """Reserve TPM budget before a call, sleeping if needed to stay under
    TPM_LIMIT. Returns the reservation entry so it can be corrected with
    the real usage once the response comes back (or None if TPM limiting
    is disabled)."""
    if TPM_LIMIT <= 0:
        return None

    now = time.time()
    while _token_window and now - _token_window[0]["time"] > 60:
        _token_window.popleft()

    used = sum(e["tokens"] for e in _token_window)
    if used + estimated_tokens > TPM_LIMIT:
        oldest_time = _token_window[0]["time"] if _token_window else now
        wait = 60 - (now - oldest_time) + 0.5
        if wait > 0:
            print(
                f"[TPM] Approaching the token-per-minute budget "
                f"({used}/{TPM_LIMIT}). Waiting {wait:.1f}s to stay under it..."
            )
            time.sleep(wait)
        now = time.time()
        while _token_window and now - _token_window[0]["time"] > 60:
            _token_window.popleft()

    entry = {"time": time.time(), "tokens": estimated_tokens}
    _token_window.append(entry)
    return entry


def _finalize_token_usage(entry, actual_tokens):
    """Correct a reservation with the real token usage once it's known."""
    if entry is not None and actual_tokens:
        entry["tokens"] = actual_tokens


# ── Rate limit (429) handling ───────────────────────────────────────────────
# Providers like Groq distinguish between per-minute limits (RPM/TPM) and
# per-day limits (TPD). If the error reports a SHORT wait time (e.g. an RPM
# or TPM limit), the script automatically sleeps and retries. If the wait
# time is LONG (e.g. a daily token limit that only resets in hours), the
# script stops with a clear message -- the checkpoint is already saved, so
# you just re-run the same script later to RESUME, not start over.
MAX_AUTO_SLEEP_SECONDS = float(os.environ.get("LLM_MAX_AUTO_SLEEP_SECONDS", "300"))
MAX_RATE_LIMIT_RETRIES = 5

_RETRY_AFTER_PATTERN = re.compile(
    r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE
)


def _parse_retry_after_seconds(message: str):
    """Extract the wait duration from a provider error message (Groq format: '1h5m24.288s')."""
    m = _RETRY_AFTER_PATTERN.search(message)
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2)) if m.group(2) else 0
    seconds = float(m.group(3)) if m.group(3) else 0.0
    return hours * 3600 + minutes * 60 + seconds


def _extract_json_block(raw: str):
    """Extract the first JSON block from the model's output, even if wrapped in text/markdown."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    # try parsing directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # fall back to finding the outermost { ... } or [ ... ] block
    for opener, closer in (("[", "]"), ("{", "}")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_not_exception_type(RateLimitError),
)
def _chat_once(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """A single raw API call. Automatically retried (short backoff) only for
    transient errors like timeouts/connection issues. RateLimitError is
    excluded from this retry -- it's handled separately by chat() below,
    which knows how to parse the provider's requested wait time."""
    estimated = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + MAX_OUTPUT_TOKENS
    reservation = _reserve_token_budget(estimated)
    _respect_call_interval()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    actual_total = getattr(usage, "total_tokens", None) if usage else None
    _finalize_token_usage(reservation, actual_total)

    return response.choices[0].message.content


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Chat call, returns the model's raw text output.

    Handles rate limits (429) intelligently:
    - If the provider's requested wait is SHORT (<= LLM_MAX_AUTO_SLEEP_SECONDS,
      default 5 minutes) -> sleep automatically then retry. This handles
      brief RPM-type limits.
    - If the wait is LONG (e.g. a daily token limit that only resets in
      hours) -> raise immediately with a clear message, WITHOUT blocking
      the terminal for hours. Progress is already safe in the checkpoint;
      just re-run the same script later to resume.
    """
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return _chat_once(system_prompt, user_prompt, temperature)
        except RateLimitError as e:
            wait_seconds = _parse_retry_after_seconds(str(e))
            if wait_seconds is None:
                wait_seconds = 30.0  # fallback if the message format isn't recognized

            if wait_seconds <= MAX_AUTO_SLEEP_SECONDS:
                print(
                    f"[RATE LIMIT] Provider asked us to wait ~{wait_seconds:.0f}s. "
                    f"Sleeping automatically, then retrying..."
                )
                time.sleep(wait_seconds + 2)
                continue

            minutes = wait_seconds / 60
            raise RuntimeError(
                f"Provider rate limit: need to wait ~{minutes:.0f} minutes "
                f"({minutes / 60:.1f} hours) for the quota to reset. This is "
                f"likely a DAILY limit (tokens per day), not just a per-minute cap.\n"
                f"Progress processed so far is SAVED (see the checkpoint in the "
                f"output/ folder). Wait for the quota to reset, then re-run the "
                f"same command -- the script will RESUME from where it left off, "
                f"not start over."
            ) from e

    raise RuntimeError(
        f"Still hitting rate limits after {MAX_RATE_LIMIT_RETRIES} attempts. "
        f"Try again in a bit."
    )


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.3):
    """
    Same as chat(), but forces & validates JSON output.
    Retries up to 3 times with additional instructions if parsing fails.
    """
    attempt_prompt = user_prompt
    for attempt in range(3):
        raw = chat(system_prompt, attempt_prompt, temperature=temperature)
        parsed = _extract_json_block(raw)
        if parsed is not None:
            return parsed
        attempt_prompt = (
            user_prompt
            + "\n\nIMPORTANT: your previous answer was not valid JSON. "
            "Reply with PURE JSON only, no explanation, no markdown."
        )
    raise ValueError(f"Failed to get valid JSON after 3 attempts. Last output:\n{raw}")
