"""
llm_client.py

Wrapper tipis di atas SDK `openai`, diarahkan ke endpoint OpenAI-compatible
APA PUN yang kamu pilih -- bisa LiteLLM (Ollama + Hermes-4-14B lewat
Cloudflare Tunnel), bisa juga langsung ke NVIDIA Build/NIM
(https://integrate.api.nvidia.com/v1), atau provider OpenAI-compatible
lain. Tidak ada kode yang perlu diubah untuk pindah provider -- cukup
ganti environment variable di bawah.

Konfigurasi lewat environment variable (taruh di file .env atau export manual):

    LLM_BASE_URL   URL endpoint, harus diakhiri /v1
                   - LiteLLM kamu sendiri: https://nama-tunnel-kamu.trycloudflare.com/v1
                   - NVIDIA Build/NIM:     https://integrate.api.nvidia.com/v1
    LLM_API_KEY    - LiteLLM: sesuai master_key/virtual key di config kamu
                   - NVIDIA Build: API key "nvapi-..." dari build.nvidia.com
    LLM_MODEL      nama model sesuai provider, mis. "hermes-4-14b" (alias LiteLLM)
                   atau "meta/llama-3.3-70b-instruct" (NVIDIA Build)

Catatan: model kecil/lokal (mis. 14B) kadang kurang konsisten soal "JSON
murni tanpa teks tambahan". Fungsi `chat_json` di bawah sudah menangani
ini dengan ekstraksi blok JSON + retry otomatis.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4000/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-local")
MODEL = os.environ.get("LLM_MODEL", "hermes-4-14b")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Beberapa provider gratis (mis. NVIDIA Build/NIM) membatasi request per menit
# (saat ini umumnya ~40 RPM). Atur LLM_MIN_SECONDS_BETWEEN_CALLS di .env kalau
# kamu mulai kena error 429 (rate limit). Default 0 = tanpa jeda tambahan.
_MIN_INTERVAL = float(os.environ.get("LLM_MIN_SECONDS_BETWEEN_CALLS", "0"))
_last_call_time = [0.0]

# ── Penanganan rate limit (429) ───────────────────────────────────────────
# Provider seperti Groq membedakan limit per-menit (RPM) dan limit per-hari
# (TPD/TPM). Kalau errornya menyebutkan waktu tunggu SINGKAT (mis. limit RPM),
# script akan otomatis tidur lalu retry. Kalau waktu tunggunya PANJANG (mis.
# limit token harian yang baru reset besok/beberapa jam lagi), script akan
# berhenti dengan pesan jelas -- checkpoint sudah tersimpan, jadi tinggal
# jalankan ulang script yang sama nanti untuk MELANJUTKAN, bukan mulai ulang.
MAX_AUTO_SLEEP_SECONDS = float(os.environ.get("LLM_MAX_AUTO_SLEEP_SECONDS", "300"))
MAX_RATE_LIMIT_RETRIES = 5

_RETRY_AFTER_PATTERN = re.compile(
    r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE
)


def _parse_retry_after_seconds(message: str):
    """Ambil durasi tunggu dari pesan error provider (format Groq: '1h5m24.288s')."""
    m = _RETRY_AFTER_PATTERN.search(message)
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2)) if m.group(2) else 0
    seconds = float(m.group(3)) if m.group(3) else 0.0
    return hours * 3600 + minutes * 60 + seconds


def _respect_rate_limit():
    if _MIN_INTERVAL <= 0:
        return

    elapsed = time.time() - _last_call_time[0]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time[0] = time.time()


def _extract_json_block(raw: str):
    """Ambil blok JSON pertama dari output model, walau dibungkus teks/markdown."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    # coba parse langsung dulu
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # cari blok { ... } atau [ ... ] terluar sebagai fallback
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _chat_once(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Satu kali panggilan API mentah. Retry otomatis (backoff pendek) hanya
    untuk error transient seperti timeout/koneksi -- BUKAN untuk rate limit,
    yang ditangani terpisah oleh chat() di bawah."""
    _respect_rate_limit()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def chat(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Panggilan chat, mengembalikan teks mentah dari model.

    Menangani rate limit (429) secara cerdas:
    - Kalau waktu tunggu yang diminta provider SINGKAT (<= LLM_MAX_AUTO_SLEEP_SECONDS,
      default 5 menit) -> tidur otomatis lalu retry. Ini menangani limit RPM sesaat.
    - Kalau waktu tunggunya PANJANG (mis. limit token harian yang baru reset
      berjam-jam lagi) -> langsung raise error dengan pesan jelas, TIDAK menahan
      terminal tidur berjam-jam. Progress sudah aman di checkpoint; tinggal
      jalankan ulang script yang sama nanti untuk melanjutkan.
    """
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return _chat_once(system_prompt, user_prompt, temperature)
        except RateLimitError as e:
            wait_seconds = _parse_retry_after_seconds(str(e))
            if wait_seconds is None:
                wait_seconds = 30.0  # fallback kalau format pesan tidak dikenali

            if wait_seconds <= MAX_AUTO_SLEEP_SECONDS:
                print(
                    f"[RATE LIMIT] Provider minta tunggu ~{wait_seconds:.0f} detik. "
                    f"Menunggu otomatis lalu mencoba lagi..."
                )
                time.sleep(wait_seconds + 2)
                continue

            minutes = wait_seconds / 60
            raise RuntimeError(
                f"Rate limit provider: perlu menunggu ~{minutes:.0f} menit "
                f"({minutes / 60:.1f} jam) sebelum kuota reset. Ini kemungkinan "
                f"batas HARIAN (token per day), bukan sekadar batas per-menit.\n"
                f"Progress yang sudah diproses TETAP TERSIMPAN (lihat checkpoint "
                f"di folder output/). Tunggu kuota reset, lalu jalankan ulang "
                f"perintah yang sama -- script akan MELANJUTKAN dari titik "
                f"terakhir, bukan mengulang dari awal."
            ) from e

    raise RuntimeError(
        f"Masih kena rate limit setelah {MAX_RATE_LIMIT_RETRIES}x percobaan. "
        f"Coba lagi beberapa saat lagi."
    )


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.3):
    """
    Sama seperti chat(), tapi memaksa & memvalidasi output berupa JSON.
    Akan retry sampai 3x dengan instruksi tambahan kalau parsing gagal.
    """
    attempt_prompt = user_prompt
    for attempt in range(3):
        raw = chat(system_prompt, attempt_prompt, temperature=temperature)
        parsed = _extract_json_block(raw)
        if parsed is not None:
            return parsed
        attempt_prompt = (
            user_prompt
            + "\n\nPENTING: jawaban kamu sebelumnya bukan JSON valid. "
            "Balas HANYA dengan JSON murni, tanpa penjelasan, tanpa markdown."
        )
    raise ValueError(f"Gagal mendapatkan JSON valid setelah 3 percobaan. Output terakhir:\n{raw}")
