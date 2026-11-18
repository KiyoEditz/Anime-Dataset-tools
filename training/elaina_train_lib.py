#!/usr/bin/env python3
"""
elaina_train_lib.py

Framework-agnostic helpers shared by the Molab notebooks and by the local
self-test. Nothing in here needs a GPU, and nothing imports torch at module
import time -- so this file imports cleanly on the Windows dev machine and
the self-test proves the risky logic without renting an A100.

What lives here:

  * disk / RAM / GPU measurement and the disk tier gate
  * chat-template introspection (derive the model-turn opener instead of
    hard-coding "<start_of_turn>model\\n", which differs between Gemma 3/4)
  * response-only loss masking done at token level, deterministically
  * bounded-disk "push then delete" shard streaming to the HuggingFace Hub

Run the built-in checks with:
    python training/elaina_train_lib.py --self-test
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2

USER_MARK = "ZXQ_USER_MARKER_7F3A"
ASSIST_MARK = "ZXQ_ASSISTANT_MARKER_7F3A"


# ---------------------------------------------------------------------------
# measuring the machine
# ---------------------------------------------------------------------------

def to_gib(n: float) -> float:
    return round(n / GIB, 2)


def disk_report(paths) -> dict:
    """disk_usage() for each path. Never raises -- an unusable path is reported."""
    report = {}
    for p in paths:
        try:
            u = shutil.disk_usage(str(p))
            report[str(p)] = {
                "total_gib": to_gib(u.total),
                "free_gib": to_gib(u.free),
                "used_gib": to_gib(u.used),
            }
        except Exception as exc:  # path missing, permission, etc.
            report[str(p)] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def min_free_gib(report: dict) -> float | None:
    """The tightest free space across the measured paths -- that is the real budget."""
    values = [v["free_gib"] for v in report.values() if "free_gib" in v]
    return min(values) if values else None


def host_ram_gib() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        return to_gib(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return to_gib(int(line.split()[1]) * 1024)
    except Exception:
        pass
    return None


def gpu_info() -> dict:
    """Best-effort GPU description. Works with no torch and no GPU present."""
    info = {"torch": None, "cuda": None, "capability": None, "name": None,
            "vram_gib": None, "bf16": None, "error": None, "nvidia_smi": None}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        info["bf16"] = bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["capability"] = list(torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["vram_gib"] = to_gib(props.total_memory)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["nvidia_smi"] = out.stdout.strip().splitlines()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# the disk gate
# ---------------------------------------------------------------------------

# (tier, min_free_gib, min_vram_gib)
TIER_RULES = [
    ("full31b", 150.0, 80.0),
    ("small31b", 90.0, 80.0),
    ("12b", 45.0, 24.0),
]

GATE_TEXT = """
**ENV GATE: tidak bisa lanjut di mesin ini.**

| Ukuran | Nilai |
|---|---|
| Disk bebas tersempit | **{free} GB** |
| RAM host tersedia | {ram} GB |
| GPU | {gpu} |
| compute capability | {cap} |
| VRAM | {vram} GB |
| torch / CUDA | {torch} / {cuda} |

Butuh minimal **{need} GB** disk bebas untuk jalur **{tier_path}**.

Ukuran acuan supaya tidak menipu diri: base `KiyoEditz/Gemma-4-Dark-Gemistry-31B` bf16 = **62,5 GB**,
merged bf16 = 62,5 GB, f16/bf16 GGUF = 62 GB, Q8_0 = 33 GB, mmproj vision = ~1,2 GB.

Pilihan:
1. Gunakan instance GPU NVIDIA Blackwell 96 GB VRAM dengan volume disk ≥150 GB (Tier A) atau ≥90 GB (Tier B).
2. Untuk verifikasi pipeline cepat tanpa 31B, jalankan mode smoke (`ELAINA_MODE=smoke` dengan `google/gemma-4-E2B-it`).
3. Jalankan notebook ini di **HF Jobs** dengan dataset & repo yang sama.

Tidak ada unduhan yang dimulai sebelum gate lolos.
"""


def check_torch_cuda_compat(torch_v: str, cuda_v: str) -> bool:
    """Return True if torch >= 2.7.0 and CUDA >= 12.8 (required for Blackwell sm_120)."""
    t_clean = str(torch_v or "0").split("+")[0].strip()
    c_clean = str(cuda_v or "0").split("+")[0].strip()
    try:
        from packaging import version  # noqa: PLC0415

        t_ver = version.parse(t_clean)
        c_ver = version.parse(c_clean)
        t_base = version.parse(f"{t_ver.major}.{t_ver.minor}.{t_ver.micro}")
        c_base = version.parse(f"{c_ver.major}.{c_ver.minor}")
        return t_base >= version.parse("2.7.0") and c_base >= version.parse("12.8")
    except Exception:
        t_nums = [int(m) for m in re.findall(r"\d+", t_clean)[:2]]
        c_nums = [int(m) for m in re.findall(r"\d+", c_clean)[:2]]
        torch_ok = len(t_nums) >= 2 and ((t_nums[0] > 2) or (t_nums[0] == 2 and t_nums[1] >= 7))
        cuda_ok = len(c_nums) >= 2 and ((c_nums[0] > 12) or (c_nums[0] == 12 and c_nums[1] >= 8))
        return torch_ok and cuda_ok


def env_gate(disk_paths, min_vram_override: float | None = None) -> dict:
    """Measure, classify, and explain. Returns a dict; caller decides to halt.

    No downloads happen before this returns, so a wrong assumption about the
    container is cheap.
    """
    report = disk_report(disk_paths)
    free = min_free_gib(report)
    ram = host_ram_gib()
    gpu = gpu_info()
    vram = gpu.get("vram_gib")

    effective_vram = min_vram_override if min_vram_override is not None else vram

    tier = "blocked"
    if free is not None:
        for name, need_disk, need_vram in TIER_RULES:
            if free >= need_disk and (effective_vram is None or effective_vram >= need_vram):
                tier = name
                break

    warnings = []
    if gpu.get("error"):
        warnings.append(
            "torch belum terpasang atau gagal dimuat "
            f"({gpu['error']}). Pasang torch cu128 sebelum memuat model."
        )
    elif gpu.get("capability") == [12, 0]:
        torch_v = gpu.get("torch") or "0"
        cuda_v = gpu.get("cuda") or "0"
        if not check_torch_cuda_compat(str(torch_v), str(cuda_v)):
            warnings.append(
                f"Blackwell sm_120 terdeteksi tapi torch {torch_v} / CUDA {cuda_v} "
                "mungkin tidak punya kernel. Pasang: pip install --index-url "
                "https://download.pytorch.org/whl/cu128 torch torchvision"
            )

    # Xet keeps a second copy of every shard unless this is off -- already used
    # in Collab-Llama.py for the same reason.
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    need = {"full31b": 150, "small31b": 90, "12b": 45}.get(tier, 90)
    result = {
        "tier": tier,
        "free_gib": free,
        "ram_gib": ram,
        "vram_gib": vram,
        "gpu": gpu,
        "disk": report,
        "warnings": warnings,
        "blocked": tier == "blocked",
        "message": GATE_TEXT.format(
            free=f"{free:.1f}" if free is not None else "tidak terbaca",
            ram=f"{ram:.1f}" if ram is not None else "tidak terbaca",
            gpu=gpu.get("name") or "tidak ada",
            cap=gpu.get("capability") or "-",
            vram=f"{vram:.0f}" if vram is not None else "-",
            torch=gpu.get("torch") or "-",
            cuda=gpu.get("cuda") or "-",
            need=need,
            tier_path={"full31b": "31B penuh", "small31b": "31B hemat disk",
                       "12b": "smoke / 24GB"}.get(tier, "31B"),
        ) if tier == "blocked" else "",
    }
    return result


def require_free(path, need_gib: float, what: str) -> None:
    """Raise with a readable message instead of failing halfway through a 60 GB write."""
    try:
        free = to_gib(shutil.disk_usage(str(path)).free)
    except Exception as exc:
        raise RuntimeError(f"cannot measure free space at {path}: {exc}") from exc
    if free < need_gib:
        raise RuntimeError(
            f"{what}: butuh ~{need_gib:.1f} GB bebas di {path}, tersedia {free:.1f} GB. "
            f"Hapus artefak sebelumnya (cache base / merged / GGUF) lalu ulangi."
        )


# ---------------------------------------------------------------------------
# chat template introspection
# ---------------------------------------------------------------------------

def derive_turn_parts(apply_template) -> dict:
    """Derive the user-turn opener, model-turn opener, and EOT closer from the template.

    Gemma 3 uses `<start_of_turn>model\\n` and `<end_of_turn>\\n`;
    Gemma 4 uses `<|turn>model\\n` (`<|turn>model\\n<|channel>thought\\n<channel|>` when
    `add_generation_prompt=True`) and `<turn|>\\n`.
    Reading these out of the template avoids hard-coding version-specific tags.

    `apply_template` is called like tokenizer.apply_chat_template.
    """
    probe = [
        {"role": "user", "content": USER_MARK},
        {"role": "assistant", "content": ASSIST_MARK},
    ]
    rendered = apply_template(probe, tokenize=False, add_generation_prompt=False)
    if USER_MARK not in rendered or ASSIST_MARK not in rendered:
        raise RuntimeError(
            "chat template dropped the probe content; cannot locate the turns. "
            f"Render was: {rendered[:400]!r}"
        )

    instruction_part = rendered[: rendered.index(USER_MARK)]

    user_only = [probe[0]]
    plain = apply_template(user_only, tokenize=False, add_generation_prompt=False)
    with_gen = apply_template(user_only, tokenize=False, add_generation_prompt=True)
    if not with_gen.startswith(plain):
        raise RuntimeError(
            "add_generation_prompt=True is not an extension of the plain render; "
            "the template is not prefix-stable and response masking would be wrong."
        )
    response_part = with_gen[len(plain):]
    if not response_part.strip():
        raise RuntimeError("generation prompt is empty; cannot derive the model-turn opener")

    rendered_response_part = (
        rendered[len(plain) : rendered.index(ASSIST_MARK)]
        if rendered.startswith(plain)
        else response_part
    )
    eot_part = rendered[rendered.index(ASSIST_MARK) + len(ASSIST_MARK) :]

    return {
        "instruction_part": instruction_part,
        "response_part": response_part,
        "rendered_response_part": rendered_response_part,
        "eot_part": eot_part,
        "rendered_probe": rendered,
    }


def _extract_turn_texts(
    tokenizer,
    prefix_msgs: list[dict],
    upto_msgs: list[dict],
    prev_full: str,
    response_part: str | None = None,
) -> tuple[str, str, str, bool]:
    """Return (seg_prompt_text, seg_response_text, full_upto, recovered) for one assistant turn."""
    prompt_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True
    )
    full_upto = tokenizer.apply_chat_template(
        upto_msgs, tokenize=False, add_generation_prompt=False
    )

    if not prompt_text.startswith(prev_full):
        raise RuntimeError("multi-turn prefix render is not monotonic across turns")

    recovered = False
    if full_upto.startswith(prompt_text):
        seg_prompt = prompt_text[len(prev_full) :]
        seg_response = full_upto[len(prompt_text) :]
    else:
        plain_before = tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=False
        )
        reply_raw = upto_msgs[-1].get("content", "")
        reply_trimmed = reply_raw.strip() or reply_raw
        if (
            full_upto.startswith(plain_before)
            and prompt_text.startswith(plain_before)
            and reply_trimmed
            and reply_trimmed in full_upto[len(plain_before) :]
        ):
            gen_opener = prompt_text[len(plain_before) :]
            rest = full_upto[len(plain_before) :]
            op_len = rest.index(reply_trimmed)
            rendered_opener = rest[:op_len]
            if rendered_opener and gen_opener.startswith(rendered_opener):
                # Gemma 4 style: add_generation_prompt=True injects empty thought channel
                # (<|turn>model\n<|channel>thought\n<channel|>) whereas add_generation_prompt=False
                # renders <|turn>model\n{reply}<turn|>\n. Keep prompt_text (with thought channel)
                # masked as -100 so training matches inference, and supervise {reply}<turn|>\n.
                seg_prompt = prompt_text[len(prev_full) :]
                seg_response = rest[op_len:]
                recovered = True
            else:
                raise RuntimeError(
                    "the generation prompt opener does not match the rendered model opener"
                )
        elif response_part and full_upto[len(prev_full) :].count(response_part) == 1:
            tail = full_upto[len(prev_full) :]
            cut = tail.index(response_part) + len(response_part)
            seg_prompt = tail[:cut]
            seg_response = tail[cut:]
            recovered = True
        else:
            raise RuntimeError(
                "the generation prompt is not a prefix of the full render; "
                "response masking would be incorrect for this template"
            )

    return seg_prompt, seg_response, full_upto, recovered


def tokenize_with_mask(tokenizer, messages, max_len: int, response_part: str | None = None):
    """Tokenize one conversation; labels are -100 on prompts, real ids on assistant replies.

    Supports:
      * single-turn with or without a system message ([user, assistant] or [system, user, assistant])
      * multi-turn conversations ([system, u1, a1, u2, a2, ...]), supervising every assistant turn
      * Gemma 4 templates where add_generation_prompt=True injects `<|channel>thought\\n<channel|>`
      * EOT-preserving truncation so the final supervised token is always the turn-closer token
    """
    if not isinstance(messages, list) or len(messages) < 2:
        raise RuntimeError("messages must contain at least one user turn and one assistant turn")
    if messages[-1].get("role") not in ("assistant", "model"):
        raise RuntimeError("conversation must end with an assistant turn")

    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") in ("assistant", "model")
    ]
    if not assistant_indices or assistant_indices[0] == 0:
        raise RuntimeError("conversation must have a prompt turn before the first assistant turn")

    input_ids: list[int] = []
    labels: list[int] = []
    prompt_tokens = 0
    response_tokens = 0
    first_prompt_tokens = 0
    truncated = 0
    recovered_any = False
    prev_full = ""

    for turn_idx, a_idx in enumerate(assistant_indices):
        prefix_msgs = messages[:a_idx]
        upto_msgs = messages[: a_idx + 1]
        seg_prompt, seg_response, prev_full, rec = _extract_turn_texts(
            tokenizer, prefix_msgs, upto_msgs, prev_full, response_part
        )
        recovered_any = recovered_any or rec

        p_ids = list(tokenizer(seg_prompt, add_special_tokens=False)["input_ids"])
        r_ids = list(tokenizer(seg_response, add_special_tokens=False)["input_ids"])
        if not r_ids:
            raise RuntimeError("assistant turn tokenized to nothing")

        budget = max_len - len(input_ids) - len(p_ids)
        if budget < 8:
            if turn_idx == 0:
                raise RuntimeError(
                    f"prompt alone is {len(p_ids)} tokens, leaving {budget} for the reply "
                    f"(max_len={max_len})"
                )
            truncated = 1
            break

        if len(r_ids) > budget:
            # Preserve the EOT closer token(s) at the end of the response so the model
            # always learns to emit the stop token (<turn|> / <end_of_turn> / <eos>).
            eot_tail = r_ids[-2:] if seg_response.endswith("\n") and len(r_ids) >= 2 else r_ids[-1:]
            keep = max(1, budget - len(eot_tail))
            r_ids = r_ids[:keep] + list(eot_tail)
            truncated = 1

        if turn_idx == 0:
            first_prompt_tokens = len(p_ids)
        prompt_tokens += len(p_ids)
        response_tokens += len(r_ids)
        input_ids.extend(p_ids)
        input_ids.extend(r_ids)
        labels.extend([-100] * len(p_ids))
        labels.extend(r_ids)

        if truncated:
            break

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_tokens": prompt_tokens,
        "first_prompt_tokens": first_prompt_tokens,
        "response_tokens": response_tokens,
        "num_turns": len(assistant_indices),
        "truncated": truncated,
        "recovered": recovered_any,
    }


class ResponseOnlyCollator:
    """Right-pads input_ids / attention_mask (and optional token_type_ids / mm_token_type_ids), pads labels with -100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch  # noqa: PLC0415

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        has_tti = all("token_type_ids" in f for f in features)
        has_mm_tti = all("mm_token_type_ids" in f for f in features)
        tti_list, mm_tti_list = [], []

        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
            if has_tti:
                tti_list.append(list(f["token_type_ids"]) + [0] * pad)
            if has_mm_tti:
                mm_tti_list.append(list(f["mm_token_type_ids"]) + [0] * pad)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
        if has_tti:
            batch["token_type_ids"] = torch.tensor(tti_list, dtype=torch.long)
        if has_mm_tti:
            batch["mm_token_type_ids"] = torch.tensor(mm_tti_list, dtype=torch.long)
        return batch


def masking_report(tokenizer, row, max_len: int, response_part: str | None = None) -> dict:
    """Human-checkable evidence that only the assistant span(s) are supervised."""
    result = tokenize_with_mask(tokenizer, row["messages"], max_len, response_part)
    ids = result["input_ids"]
    labels = result["labels"]
    supervised = [i for i, lab in enumerate(labels) if lab != -100]
    contiguous = supervised == list(range(supervised[0], supervised[-1] + 1)) if supervised else False
    first_prompt = result.get("first_prompt_tokens", result["prompt_tokens"])
    num_turns = result.get("num_turns", 1)
    ok = (
        bool(supervised)
        and supervised[0] == first_prompt
        and (contiguous if num_turns == 1 else labels[-1] != -100)
    )
    return {
        **result,
        "contiguous_span": contiguous,
        "first_supervised": supervised[0] if supervised else None,
        "supervised_tokens": len(supervised),
        "prompt_preview": tokenizer.decode(ids[: min(24, len(ids))]),
        "reply_preview": tokenizer.decode(ids[supervised[0]: supervised[0] + 40]) if supervised else "",
        "expected_prompt_tokens": first_prompt,
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# hub helpers
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def redact(token: str | None) -> str:
    if not token:
        return "(none)"
    return f"{token[:6]}...{token[-4:]} ({len(token)} chars)" if len(token) > 12 else "hf_***"


def hub_file_ok(api, repo_id: str, filename: str, min_bytes: int = 1, repo_type: str = "model") -> bool:
    """Confirm a file really landed before the local copy is deleted."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
        for sibling in info.siblings or []:
            if sibling.rfilename == filename:
                size = getattr(sibling, "size", None)
                if size is None:
                    return True
                return size >= min_bytes
    except Exception:
        return False
    return False


def stream_push_shards(model, tokenizer, repo_id: str, api, tmp_dir,
                       max_shard: str = "4GB", logger=print,
                       extra_files: dict | None = None) -> list[str]:
    """Save -> upload -> delete, one shard at a time.

    Peak local disk is one shard instead of the whole 62.5 GB merged model.
    Every delete is preceded by a Hub existence check, so a failed upload
    leaves the local file in place.
    """
    tmp = Path(tmp_dir)
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    logger(f"[MERGE] writing to {tmp} (max_shard={max_shard})")
    model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size=max_shard)
    tokenizer.save_pretrained(str(tmp))

    for name, content in (extra_files or {}).items():
        (tmp / name).write_text(content, encoding="utf-8")

    api.create_repo(repo_id, private=True, exist_ok=True)
    uploaded = []

    shards = sorted(p for p in tmp.iterdir() if p.name.endswith(".safetensors"))
    for shard in shards:
        size = shard.stat().st_size
        logger(f"[PUSH] {shard.name} ({to_gib(size):.2f} GB) ...")
        api.upload_file(path_or_fileobj=str(shard), path_in_repo=shard.name,
                        repo_id=repo_id)
        if not hub_file_ok(api, repo_id, shard.name, min_bytes=size):
            raise RuntimeError(f"{shard.name} did not appear on the Hub; keeping it locally")
        shard.unlink()
        uploaded.append(shard.name)
        logger(f"       ok, local copy deleted")

    for small in sorted(tmp.iterdir()):
        if small.is_file():
            api.upload_file(path_or_fileobj=str(small), path_in_repo=small.name,
                            repo_id=repo_id)
            uploaded.append(small.name)
            small.unlink()

    shutil.rmtree(tmp, ignore_errors=True)
    logger(f"[MERGE] done: {len(uploaded)} file(s) in {repo_id}")
    return uploaded


def run_logged(cmd, logger=print, cwd=None, env=None, timeout=None) -> int:
    """Run a command, streaming its output. Used for pip / llama.cpp builds."""
    logger(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logger(line.rstrip())
    return proc.wait(timeout=timeout)


def write_run_manifest(path, **fields) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# self-test -- runs on Windows, no GPU, no transformers
# ---------------------------------------------------------------------------

def _self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("disk_report / env_gate")
    rep = disk_report(["."])
    check("disk_report returns a number", "free_gib" in rep["."] or "error" in rep["."])
    gate = env_gate(["."])
    check("gate classifies without raising", gate["tier"] in {"full31b", "small31b", "12b", "blocked"})
    check("gate reports numbers, not raw bytes", isinstance(gate["free_gib"], float) or gate["free_gib"] is None)

    print("check_torch_cuda_compat")
    check("torch 2.7.0 + cuda 12.8 ok", check_torch_cuda_compat("2.7.0+cu128", "12.8"))
    check("torch 2.8.0.dev + cuda 12.9 ok", check_torch_cuda_compat("2.8.0.dev20260501+cu129", "12.9"))
    check("torch 3.0.0 + cuda 13.0 ok", check_torch_cuda_compat("3.0.0", "13.0"))
    check("torch 2.6.0 + cuda 12.8 rejected", not check_torch_cuda_compat("2.6.0", "12.8"))
    check("torch 2.7.0 + cuda 12.4 rejected", not check_torch_cuda_compat("2.7.0", "12.4"))

    print("derive_turn_parts")

    def fake_template(msgs, tokenize=False, add_generation_prompt=False):
        s = "<bos>"
        for m in msgs:
            s += f"<{m['role']}>\n{m['content']}<end>"
        if add_generation_prompt:
            s += "<assistant>\n"
        return s

    parts = derive_turn_parts(fake_template)
    check("instruction_part is the user opener",
          parts["instruction_part"].endswith("<user>\n"), parts["instruction_part"])
    check("response_part is the model opener",
          parts["response_part"] == "<assistant>\n", parts["response_part"])
    check("eot_part is <end>", parts["eot_part"] == "<end>", parts["eot_part"])

    class FakeTok:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [(ord(c) % 90) + 10 for c in text]}

        def decode(self, ids):
            return "".join(chr((i - 10) + 32) for i in ids)

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return fake_template(msgs, tokenize, add_generation_prompt)

    tok = FakeTok()
    row = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "Shoot. Fine, I suppose."},
    ]}
    out = tokenize_with_mask(tok, row["messages"], 512, parts["response_part"])
    check("labels length matches input_ids", len(out["labels"]) == len(out["input_ids"]))
    check("prompt is masked", all(l == -100 for l in out["labels"][: out["prompt_tokens"]]))
    check("reply is supervised", all(l != -100 for l in out["labels"][out["prompt_tokens"]:]))
    check("no truncation at 512", out["truncated"] == 0)

    # 12 tokens of headroom clears the `budget < 8` guard and still forces truncation.
    tiny = tokenize_with_mask(tok, row["messages"], out["prompt_tokens"] + 12, parts["response_part"])
    check("truncation keeps the prompt whole",
          tiny["prompt_tokens"] == out["prompt_tokens"] and tiny["truncated"] == 1)
    check("truncation preserves the EOT token at the end of labels (H2)",
          tiny["labels"][-1] == out["labels"][-1] and tiny["labels"][-1] != -100)

    # 2-message single-turn without system prompt (H3)
    no_sys = tokenize_with_mask(tok, row["messages"][1:], 512, parts["response_part"])
    check("2-message [user, assistant] masks user and supervises assistant (H3)",
          no_sys["response_tokens"] > 0 and no_sys["labels"][-1] != -100)

    # Multi-turn conversation (H3)
    mt_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "reply two"},
    ]
    mt_out = tokenize_with_mask(tok, mt_msgs, 512, parts["response_part"])
    check("multi-turn has 2 turns and supervises both assistant spans (H3)",
          mt_out["num_turns"] == 2 and mt_out["response_tokens"] > len("reply one<end>"))

    # Gemma 4 template with empty thought channel on add_generation_prompt=True (H7)
    def gemma4_template(msgs, tokenize=False, add_generation_prompt=False):
        s = "<bos>"
        for m in msgs:
            role = "model" if m["role"] == "assistant" else m["role"]
            s += f"<|turn>{role}\n{m['content'].strip()}<turn|>\n"
        if add_generation_prompt:
            s += "<|turn>model\n<|channel>thought\n<channel|>"
        return s

    class Gemma4FakeTok(FakeTok):
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return gemma4_template(msgs, tokenize, add_generation_prompt)

    g4_tok = Gemma4FakeTok()
    g4_parts = derive_turn_parts(g4_tok.apply_chat_template)
    check("Gemma 4 derive_turn_parts captures thought channel opener",
          g4_parts["response_part"] == "<|turn>model\n<|channel>thought\n<channel|>"
          and g4_parts["rendered_response_part"] == "<|turn>model\n"
          and g4_parts["eot_part"] == "<turn|>\n")
    g4_mr = masking_report(g4_tok, row, 512, g4_parts["response_part"])
    check("Gemma 4 masking_report ok (H7)", g4_mr["ok"] and g4_mr["recovered"])

    print("masking_report")
    mr = masking_report(tok, row, 512, parts["response_part"])
    check("span is contiguous", mr["contiguous_span"])
    check("span starts at the prompt boundary", mr["first_supervised"] == mr["prompt_tokens"])

    print("ResponseOnlyCollator (torch optional)")
    try:
        import torch  # noqa: F401, PLC0415

        coll = ResponseOnlyCollator(0)
        batch = coll([out, tiny])
        check("batch shape", tuple(batch["input_ids"].shape)[0] == 2)
        check("padding labels are -100", int(batch["labels"][0, -1]) in (-100,) or
              len(out["input_ids"]) == batch["input_ids"].shape[1])
        check("attention mask has zeros where padded",
              int(batch["attention_mask"][1, -1]) == 0 or batch["input_ids"].shape[1] == len(tiny["input_ids"]))
    except ImportError:
        print("  [skip] torch not installed on this machine")

    print("redact")
    check("token is not echoed", "abcdefghijklmnop" not in redact("hfabcd1234efghijklmnop"))

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        return 1
    print("\nall self-tests passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
    print("Run with --self-test to verify the masking and gate logic.")