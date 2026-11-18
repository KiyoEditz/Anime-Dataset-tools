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
import warnings
from collections import deque

from dotenv import load_dotenv
load_dotenv()

# Filter harmless warning from older openai/pydantic compatibility layer on Python 3.14+
warnings.filterwarnings("ignore", message=".*Core Pydantic V1 functionality.*")

from openai import (
    OpenAI,
    RateLimitError,
    NotFoundError,
    AuthenticationError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)

def _normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith("/openai") or "googleapis.com" in url or url.endswith("/v1"):
        return url
    return f"{url}/v1"

BASE_URL = _normalize_base_url(os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"))
API_KEY = os.environ.get("LLM_API_KEY", "sk-local")

def _parse_models(raw: str) -> list[str]:
    raw = raw.strip().strip('"').strip("'")
    models = [m.strip().strip('"').strip("'") for m in raw.split(",") if m.strip()]
    return models if models else ["default"]

MODELS = _parse_models(os.environ.get("LLM_MODEL", "default"))
MODEL = MODELS[0]
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))

try:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
except TypeError:
    import httpx
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=httpx.Client(timeout=TIMEOUT))

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


def _clean_model_output(raw: str) -> str:
    """Strip reasoning/think tags and clean extra whitespace."""
    if not raw:
        return ""
    # Strip <think>...</think> (DeepSeek-R1 / Qwen reasoning models)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _extract_json_block(raw: str):
    """Extract the first JSON block from the model's output, even if wrapped in text/markdown."""
    if not raw:
        return None
    raw = _clean_model_output(raw)

    # 1. Try parsing directly
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences ```json ... ``` or ``` ... ```
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
    for candidate in fence_matches:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3. Fall back to finding the outermost { ... } or [ ... ] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def _is_daily_quota_error(exc: Exception) -> bool:
    """Check if the error is specifically a daily quota / token limit error."""
    msg = str(exc).lower()

    wait_seconds = _parse_retry_after_seconds(str(exc))
    if wait_seconds is not None and wait_seconds > MAX_AUTO_SLEEP_SECONDS:
        return True

    daily_keywords = [
        "per day",
        "per-day",
        "daily",
        "requests_per_day",
        "tokens_per_day",
        "requests per day",
        "tokens per day",
        "tpd limit",
        "rpd limit",
        "day limit",
        "insufficient_quota",
        "free_tier_requests_per_day",
        "exceeded your current quota",
    ]
    return any(keyword in msg for keyword in daily_keywords)


def _chat_single_call(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Single raw API call for a specific model."""
    estimated = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + MAX_OUTPUT_TOKENS
    reservation = _reserve_token_budget(estimated)
    _respect_call_interval()

    response = client.chat.completions.create(
        model=model,
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

    content = response.choices[0].message.content or ""
    return _clean_model_output(content)


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Chat call, returns the model's raw text output.

    Iterates through MODELS starting from the first model on EVERY call.
    - If a model is busy/overloaded (e.g. Google API 503 high demand), it retries up to 3 times,
      then falls back to the next model in the list.
    - If a model returns 404 (not found / retired), it immediately falls back to the next model.
    - If a model is rate-limited (429 per-minute limit), it falls back to the next model immediately.
    - If the error is a DAILY token/quota limit, fallback is halted and the error is raised immediately.
    - Once any call succeeds, the fallback cycle resets to the first model for subsequent calls.
    """
    last_exception = None
    total_models = len(MODELS)

    for model_idx, model in enumerate(MODELS):
        is_last_model = (model_idx == total_models - 1)
        next_model = MODELS[model_idx + 1] if not is_last_model else None

        for attempt in range(1, 4):
            try:
                return _chat_single_call(model, system_prompt, user_prompt, temperature)
            except Exception as e:
                last_exception = e

                # 1. Daily quota limit check: Fallback does NOT apply to daily limit. Stop immediately.
                if _is_daily_quota_error(e):
                    wait_seconds = _parse_retry_after_seconds(str(e))
                    wait_msg = f" (wait ~{wait_seconds / 60:.0f}m)" if wait_seconds else ""
                    raise RuntimeError(
                        f"Daily token/quota limit reached on '{model}'{wait_msg}: {e}\n"
                        f"Progress processed so far is SAVED in the output/ folder. "
                        f"Please wait for your daily quota to reset before resuming."
                    ) from e

                # 2. Authentication error: invalid API key, no model will work
                if isinstance(e, AuthenticationError):
                    raise

                # 3. Model not found (404) or retired / unavailable
                if isinstance(e, NotFoundError) or "not found" in str(e).lower() or "not available" in str(e).lower() or "does not exist" in str(e).lower():
                    if not is_last_model:
                        print(f"[FALLBACK] Model '{model}' not found or unavailable. Falling back to '{next_model}'...")
                        break
                    else:
                        raise

                # 4. RateLimitError (429) - Short rate limit (RPM/TPM)
                if isinstance(e, RateLimitError):
                    wait_seconds = _parse_retry_after_seconds(str(e))
                    if wait_seconds is None:
                        wait_seconds = 10.0

                    if not is_last_model:
                        print(
                            f"[FALLBACK] Model '{model}' rate-limited / full (wait ~{wait_seconds:.0f}s). "
                            f"Falling back to '{next_model}'..."
                        )
                        break
                    else:
                        if wait_seconds <= MAX_AUTO_SLEEP_SECONDS and attempt < MAX_RATE_LIMIT_RETRIES:
                            print(
                                f"[RATE LIMIT] Provider asked us to wait ~{wait_seconds:.0f}s. "
                                f"Sleeping automatically, then retrying ({attempt}/{MAX_RATE_LIMIT_RETRIES})..."
                            )
                            time.sleep(wait_seconds + 2)
                            continue
                        else:
                            raise RuntimeError(
                                f"Still hitting rate limits on '{model}'. Quota or limit exceeded: {e}"
                            ) from e

                # 5. Overloaded / Busy / Transient server errors (503, 500, timeouts, connection)
                if attempt < 3:
                    backoff = 2 ** attempt
                    print(
                        f"[RETRY] Model '{model}' is busy or encountered error ({type(e).__name__}: {e}). "
                        f"Retrying in {backoff}s (attempt {attempt}/3)..."
                    )
                    time.sleep(backoff)
                    continue
                else:
                    if not is_last_model:
                        print(
                            f"[FALLBACK] Model '{model}' failed after 3 attempts ({type(e).__name__}: {e}). "
                            f"Falling back to '{next_model}'..."
                        )
                        break
                    else:
                        raise

    raise RuntimeError(
        f"All configured models ({', '.join(MODELS)}) failed. Last error: {last_exception}"
    ) from last_exception


_chat_once = chat


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
