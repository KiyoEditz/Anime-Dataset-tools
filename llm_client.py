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

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
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


def _respect_rate_limit():
    if _MIN_INTERVAL <= 0:
        return
    import time

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
def chat(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Panggilan chat biasa, mengembalikan teks mentah dari model."""
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
