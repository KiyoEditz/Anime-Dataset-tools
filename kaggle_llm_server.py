# -*- coding: utf-8 -*-
"""
kaggle_llm_server.py

Kaggle Translation Server — Hy-MT2-30B-A3B (llama.cpp edition)
Direct llama-server OpenAI-compatible API + Cloudflare Tunnel on Kaggle dual T4 GPU (32 GB VRAM).

Setup checklist in Kaggle Notebook:
  1. Notebook Settings -> Accelerator -> "GPU T4 x2"
  2. Notebook Settings -> Internet -> "On" (requires phone verification on Kaggle)
  3. Copy-paste this script into a Kaggle notebook cell or upload as a script, then run.
  4. Copy the printed Public Tunnel URL to use with:
       py -3.14 translate_novel.py --mode dataset --base-url <printed URL>
       py -3.14 translate_novel.py --mode text --base-url <printed URL>
       py -3.14 lrc_translate_v4.py *.lrc --to id --base-url <printed URL>
"""

import subprocess
import time
import os
import re
import sys
import gc
import atexit
import signal
import shutil
import glob
import tarfile
import requests
import socket

# =====================================================================
# CONFIGURATION
# =====================================================================

# Tencent Hunyuan MT2 30B MoE (3B active), calibrated GGUF by mradermacher (~18.3 GB)
# Fits Kaggle dual T4 (32 GB combined VRAM) comfortably without CPU offloading
MODEL_REPO = "mradermacher/Hy-MT2-30B-A3B-i1-GGUF:Q4_K_M"

LLAMA_PORT = 8080
API_KEY = "sk-kaggle-local"
LLAMA_CPP_TAG = "b12076"

# Multi-arch prebuilt binary (covers T4/sm75, A100, L4, RTX in one release)
GITHUB_RELEASE_URL = "https://github.com/KiyoEditz/AIChat-With-Notebooks/releases/download/v1.4.0/llama-cuda-multiarch-b12076.tar.gz"

MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 4096
SLOT_CTX_TOKENS = MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS  # 8192 per slot (4k input + 4k output)
PARALLEL_SLOTS = 2
NUM_CTX = SLOT_CTX_TOKENS * PARALLEL_SLOTS              # 16384 total ctx across 2 slots
CACHE_RAM = 8192
GPU_IDS = "0,1"

# Sampling parameters recommended in the Tencent Hy-MT2 model card for 30B-A3B:
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = 0
REPEAT_PENALTY = 1.0

# Scratch storage: /kaggle/tmp avoids filling /kaggle/working
SCRATCH_DIR = "/kaggle/tmp" if os.path.isdir("/kaggle/tmp") else "/tmp"
MODEL_DIR = f"{SCRATCH_DIR}/models"
XET_CACHE_DIR = f"{SCRATCH_DIR}/hf-xet"
LLAMA_SRC_DIR = f"{SCRATCH_DIR}/llama.cpp"
PREBUILT_DIR = f"{SCRATCH_DIR}/llama-prebuilt"
MIN_FREE_DISK_MARGIN_GB = 4


# =====================================================================
# UTILITIES
# =====================================================================

def sh(cmd, check=True, quiet=False, capture=False):
    kw = dict(shell=True, check=check)
    if capture:
        kw.update(capture_output=True, text=True)
    elif quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kw)


def bg(cmd, log=None, env=None, cwd=None):
    out = open(log, "w") if log else subprocess.DEVNULL
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT, env=full_env, cwd=cwd)
    finally:
        if log and out != subprocess.DEVNULL:
            out.close()


def section(n, total, title):
    bar = "─" * 64
    print(f"\n{bar}\n [{n}/{total}] {title}\n{bar}")


def run_with_retry(fn, max_attempts=3, label="operation"):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_s = attempt * 5
            print(f"  ↻ retry {attempt}/{max_attempts} in {wait_s}s ({e})...")
            time.sleep(wait_s)


def acquire_free_port(preferred_port=8080, fallbacks=(8081, 8082, 8088)):
    sh("pkill -9 -f '[l]lama-server'", check=False, quiet=True)
    sh("pkill -9 -f '[c]loudflared'", check=False, quiet=True)
    sh(f"fuser -k -9 {preferred_port}/tcp 2>/dev/null", check=False, quiet=True)
    time.sleep(1)
    for p in [preferred_port] + list(fallbacks):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred_port


def detect_gpu_arch():
    r = sh("nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader", check=False, capture=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    archs = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            archs.append(parts[2].replace(".", ""))
    return archs


def check_dual_gpu_usage():
    r = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader", check=False, capture=True)
    if r.returncode != 0:
        return
    print("  GPU memory usage:")
    for line in [l for l in r.stdout.strip().splitlines() if l.strip()]:
        print(f"    {line}")


# =====================================================================
# CLEANUP HANDLERS
# =====================================================================

_procs = {"llama": None, "cloudflared": None}
_cleaned_up = False


def cleanup(*_args):
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    print("\n  🧹 Cleaning up processes and freeing RAM/VRAM...")
    for name, proc in _procs.items():
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    sh("pkill -9 -f '[l]lama-server'", check=False, quiet=True)
    sh("pkill -9 -f '[c]loudflared'", check=False, quiet=True)
    gc.collect()
    print("  ✅ Cleanup complete")


atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)
sh("pkill -9 -f '[l]lama-server'", check=False, quiet=True)
sh("pkill -9 -f '[c]loudflared'", check=False, quiet=True)


# =====================================================================
# 1. LLAMA-SERVER BINARY
# =====================================================================

def verify_prebuilt(pkg_dir):
    run_sh = f"{pkg_dir}/run.sh"
    if not os.path.isfile(run_sh):
        raise RuntimeError("prebuilt package has no run.sh")
    os.chmod(run_sh, 0o755)
    os.chmod(f"{pkg_dir}/llama-server", 0o755)
    r = sh(f"{run_sh} --version", check=False, capture=True)
    if r.returncode != 0:
        raise RuntimeError("prebuilt binary failed --version test")
    print("  ✅ Prebuilt llama-server verified successfully.")
    return run_sh


def try_prebuilt():
    if not GITHUB_RELEASE_URL:
        raise RuntimeError("GITHUB_RELEASE_URL not set")
    os.makedirs(PREBUILT_DIR, exist_ok=True)
    tarball_path = f"{PREBUILT_DIR}/prebuilt.tar.gz"
    print("  Downloading prebuilt llama-server binary...")

    def download_step():
        r = requests.get(GITHUB_RELEASE_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(tarball_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    run_with_retry(download_step, label="prebuilt download")
    extract_dir = f"{PREBUILT_DIR}/extracted"
    sh(f"rm -rf {extract_dir}", check=False, quiet=True)
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path) as tar:
        tar.extractall(extract_dir)
    entries = [d for d in glob.glob(f"{extract_dir}/*") if os.path.isdir(d)]
    if not entries:
        raise RuntimeError("empty prebuilt tarball")
    return verify_prebuilt(entries[0])


def build_from_source():
    print("  Compiling llama-server from source (fallback)...\n")
    sh(f"rm -rf {LLAMA_SRC_DIR} && git clone --depth 1 --branch {LLAMA_CPP_TAG} https://github.com/ggerganov/llama.cpp {LLAMA_SRC_DIR}")
    sh(f"cmake -B {LLAMA_SRC_DIR}/build -S {LLAMA_SRC_DIR} -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NO_VMM=ON")
    sh(f"cmake --build {LLAMA_SRC_DIR}/build --config Release -j$(nproc) --target llama-server")
    server_bin = f"{LLAMA_SRC_DIR}/build/bin/llama-server"
    if not os.path.isfile(server_bin):
        raise RuntimeError("llama-server build failed")
    return server_bin


def ensure_llama_server():
    try:
        return try_prebuilt()
    except Exception as e:
        print(f"  Prebuilt unavailable ({e}) -- building from source...")
        return build_from_source()


# =====================================================================
# 2. MODEL DOWNLOAD
# =====================================================================

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    sh("pip install -q -U huggingface_hub", check=False, quiet=True)
    from huggingface_hub import snapshot_download

    repo_id, _, quant = MODEL_REPO.partition(":")
    patterns = [f"*{quant}*.gguf"] if quant else ["*.gguf"]
    print(f"  Repo   : {repo_id}")
    print(f"  Quant  : {quant or '(any)'}")

    existing = sorted(glob.glob(f"{MODEL_DIR}/**/*{quant}*.gguf", recursive=True))
    if existing:
        print(f"  ✅ Model already present: {existing[0]}")
        return existing[0]

    print("  Downloading Hy-MT2 GGUF (~18.3 GB)...\n")
    run_with_retry(lambda: snapshot_download(repo_id=repo_id, local_dir=MODEL_DIR, allow_patterns=patterns),
                   max_attempts=3, label="model download")

    ggufs = sorted(glob.glob(f"{MODEL_DIR}/**/*.gguf", recursive=True))
    if not ggufs:
        raise RuntimeError(f"No .gguf files found in {MODEL_DIR} after download")
    print(f"  Model file: {ggufs[0]}")
    return ggufs[0]


# =====================================================================
# 3. SERVER READINESS STATE MACHINE
# =====================================================================

def wait_for_ready(proc, log_path, base_url, api_key, timeout=2400):
    start = time.time()
    print("  Waiting for model to load into dual T4 VRAM...")
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = open(log_path, errors="ignore").read()[-1200:] if os.path.isfile(log_path) else ""
            raise RuntimeError(f"llama-server exited early ({proc.returncode}):\n{tail}")
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("  ✅ llama-server health OK: model fully loaded")
                return True
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Timeout waiting for llama-server readiness")


# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  Kaggle Translation Server · Hy-MT2-30B-A3B (llama.cpp)         ║")
    print("╚" + "═" * 64 + "╝")

    TOTAL_STEPS = 6

    # 1. Check GPUs
    section(1, TOTAL_STEPS, "Check GPUs")
    archs = detect_gpu_arch()
    if not archs:
        raise RuntimeError("No GPU visible. Set Accelerator to 'GPU T4 x2' in the right-hand panel.")
    print(f"  Detected {len(archs)} GPU(s): {', '.join(archs)}")

    # 2. Ensure binary
    section(2, TOTAL_STEPS, "llama-server binary")
    server_cmd = ensure_llama_server()

    # 3. Download model
    section(3, TOTAL_STEPS, "Download Hy-MT2-30B-A3B model")
    model_path = ensure_model()

    # 4. Start llama-server
    section(4, TOTAL_STEPS, "Start llama-server (dual T4)")
    port = acquire_free_port(LLAMA_PORT)
    llama_log = "/tmp/llama_server.log"
    llama_env = {"CUDA_VISIBLE_DEVICES": GPU_IDS}

    llama_cmd = (
        f"{server_cmd} "
        f"-m {model_path} "
        f"--host 0.0.0.0 --port {port} "
        f"--api-key {API_KEY} "
        f"--ctx-size {NUM_CTX} "
        f"--parallel {PARALLEL_SLOTS} "
        f"-n {MAX_OUTPUT_TOKENS} "
        f"--no-mmap "
        f"--flash-attn on -ctk q8_0 -ctv q8_0 "
        f"-ngl 99 --split-mode layer "
        f"--temp {TEMPERATURE} --top-p {TOP_P} --top-k {TOP_K} --repeat-penalty {REPEAT_PENALTY} "
        f"--jinja "
        f"--reasoning off --chat-template-kwargs '{{\"enable_thinking\": false}}'"
    )

    _procs["llama"] = bg(llama_cmd, llama_log, env=llama_env)
    wait_for_ready(_procs["llama"], llama_log, f"http://localhost:{port}", API_KEY)
    check_dual_gpu_usage()

    # 5. Start Cloudflare Tunnel
    section(5, TOTAL_STEPS, "Cloudflare Tunnel -> Public HTTPS URL")
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")
    _procs["cloudflared"] = bg(f"cloudflared tunnel --url http://localhost:{port}", "/tmp/cloudflared.log")

    print("  Waiting for public tunnel URL...")
    tunnel_url = None
    for _ in range(90):
        time.sleep(1)
        try:
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", open("/tmp/cloudflared.log").read())
            if m:
                tunnel_url = m.group(0)
                break
        except FileNotFoundError:
            pass

    # 6. Smoke test
    section(6, TOTAL_STEPS, "Smoke Test: Japanese -> Indonesian Translation")
    smoke_test_prompt = (
        "Translate the following numbered line into Indonesian. "
        "Note that you should only output the translated result without any additional explanation:\n"
        "1. Keep the original number and period at the start of every line.\n\n"
        "1. 私、旅の魔女イレイナです。"
    )
    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "Hy-MT2-30B-A3B",
                "messages": [{"role": "user", "content": smoke_test_prompt}],
                "max_tokens": 128,
            },
            timeout=120,
        )
        resp.raise_for_status()
        out_text = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  Input : 1. 私、旅の魔女イレイナです。")
        print(f"  Output: {out_text}")
        print("  ✅ Smoke test passed!")
    except Exception as e:
        print(f"  ⚠️ Smoke test warning ({e})")

    print("\n" + "=" * 66)
    if tunnel_url:
        print("SERVER IS READY")
        print("=" * 66)
        print(f"  Public URL : {tunnel_url}")
        print(f"  OpenAI URL : {tunnel_url}/v1")
        print(f"  API Key    : {API_KEY}")
        print(f"  Model      : Hy-MT2-30B-A3B (Q4_K_M)")
        print(f"  Token Limit: {MAX_INPUT_TOKENS} input / {MAX_OUTPUT_TOKENS} output per slot ({SLOT_CTX_TOKENS} ctx/slot)")
        print("\n  Run local dataset translation (output-JP -> output-ID):")
        print(f"    py -3.14 translate_novel.py --mode dataset --base-url {tunnel_url}")
        print("\n  Run local novel chapter translation (text-JP -> text-ID):")
        print(f"    py -3.14 translate_novel.py --mode text --base-url {tunnel_url}")
        print("\n  Or translate lyrics/subtitles:")
        print(f"    py -3.14 lrc_translate_v4.py *.lrc --to id --base-url {tunnel_url}")
    else:
        print(f"Local server ready on http://localhost:{port} (tunnel URL not found yet)")
    print("=" * 66 + "\n")

    # Keep-alive loop
    tick = 0
    try:
        while True:
            time.sleep(60)
            tick += 1
            try:
                requests.get(f"http://localhost:{port}/health", timeout=5)
                status = "healthy"
            except Exception:
                status = "unreachable"
            print(f"  [{time.strftime('%H:%M:%S')}] heartbeat #{tick:04d} | server: {status} | tunnel: {tunnel_url or 'offline'}")
    except KeyboardInterrupt:
        print("\nStopping server...")
        cleanup()


if __name__ == "__main__":
    main()

