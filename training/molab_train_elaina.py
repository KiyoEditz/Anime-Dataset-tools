#!/usr/bin/env python3
"""
molab_train_elaina.py

Marimo notebook (Molab) that fine-tunes Elaina on KiyoEditz/Gemma-4-Dark-Gemistry-31B with
LoRA and exports four HuggingFace repos: dataset, LoRA adapter, merged bf16,
GGUF.

This is a marimo file, not a Jupyter notebook: cells are Python functions
decorated with @app.cell, and the dependency graph comes from the function
parameters / return tuple. Rules that keep this file valid:

  * every global is defined in exactly ONE cell (redefining it in another cell
    is a marimo graph error, not a warning)
  * a cell returns a tuple naming the globals it defines
  * nothing expensive runs at import time; heavy work sits behind run_button
  * the disk gate runs before any download, and heavy cells re-check it

Check syntax locally with:
    python -c "import ast; ast.parse(open('training/molab_train_elaina.py').read())"
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# 1. imports + configuration
# ---------------------------------------------------------------------------

@app.cell
def _():
    import importlib.util
    import json
    import math
    import os
    import re
    import shutil
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    MODE = os.environ.get("ELAINA_MODE", "full").lower()
    SMOKE = MODE == "smoke"

    CFG = {
        "mode": MODE,
        "MODEL_ID": (
            os.environ.get("ELAINA_SMOKE_MODEL", "google/gemma-4-E2B-it")
            if SMOKE
            else "KiyoEditz/Gemma-4-Dark-Gemistry-31B"
        ),
        "MODEL_FALLBACKS": ["KiyoEditz/Gemma-4-Dark-Gemistry-31B"],
        "TORCH_DTYPE": "bfloat16",
        "LOAD_IN_4BIT": False,
        "MAX_SEQ": 512 if SMOKE else 4096,
        "LORA_R": 32,
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.05,
        "TARGET_MODULES": r"^(?!.*(?:vision|visual|audio|embed_vision)).*(?:language_model|text_model|model)\.layers\.\d+\.(?:self_attn|mlp)\.(?:q|k|v|o|gate|up|down)_proj$",
        "LR": 1e-4,
        "EPOCHS": 1 if SMOKE else 2,
        "BATCH": 2,
        "GRAD_ACCUM": 2 if SMOKE else 8,
        "SEED": 3407,
        "EVAL_STEPS": 5 if SMOKE else 100,
        "SAVE_STEPS": 5 if SMOKE else 100,
        "MAX_STEPS": 5 if SMOKE else -1,
        "MAX_TRAIN_ROWS": 100 if SMOKE else 0,
        "QUANTS": ["Q4_K_M"] if SMOKE else
                  ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S",
                   "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        "ALL_QUANTS": ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S",
                       "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        "SKIP_MERGE": False,
        "MAX_SHARD": "4GB",
        "WORKDIR": os.environ.get("ELAINA_WORKDIR", "/root/work"),
        "REPO_SLUG": os.environ.get("ELAINA_REPO_SLUG", "KiyoEditz/elaina-dataset-tool"),
        "REPO_BRANCH": os.environ.get("ELAINA_REPO_BRANCH", "main"),
        "LIB_FILE": "training/elaina_train_lib.py",
        "GGUF_FILE": "training/gguf_convert.py",
        "LLAMA_CPP_REVISION": os.environ.get("ELAINA_LLAMA_CPP_REV", "master"),
        "DATASET_DIR": os.environ.get("ELAINA_DATASET_DIR", "training/out"),
    }
    CFG["OUT_DIR"] = Path(CFG["WORKDIR"])
    return CFG, MODE, Path, SMOKE, importlib, json, math, mo, os, re, shutil, sys, time


# ---------------------------------------------------------------------------
# 2. bootstrap the shared library
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, Path, importlib, mo, sys):
    def _ensure_companion(work: Path, rel_path: str) -> Path:
        target = work / Path(rel_path).name
        if not target.exists():
            import urllib.request

            url = (f"https://raw.githubusercontent.com/{CFG['REPO_SLUG']}/"
                   f"{CFG['REPO_BRANCH']}/{rel_path}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                target.write_bytes(resp.read())
        return target

    def _bootstrap_lib():
        """Import elaina_train_lib (and ensure gguf_convert.py is available) from next to the notebook, else fetch from GitHub."""
        try:
            mod = importlib.import_module("elaina_train_lib")
            local_dir = Path(mod.__file__).parent
            return mod, local_dir, "local import"
        except ImportError:
            pass

        work = Path(CFG["WORKDIR"])
        work.mkdir(parents=True, exist_ok=True)
        try:
            _ensure_companion(work, CFG["LIB_FILE"])
            _ensure_companion(work, CFG["GGUF_FILE"])
        except Exception as exc:
            raise RuntimeError(
                "could not import or download elaina_train_lib.py / gguf_convert.py\n"
                f"  error: {type(exc).__name__}: {exc}\n"
                "Fix: upload training/elaina_train_lib.py and training/gguf_convert.py next to this notebook."
            ) from exc
        sys.path.insert(0, str(work))
        importlib.invalidate_caches()
        mod = importlib.import_module("elaina_train_lib")
        return mod, work, f"downloaded from {CFG['REPO_SLUG']}@{CFG['REPO_BRANCH']}"

    lib, repo_dir, lib_origin = _bootstrap_lib()
    mo.md(f"Loaded `elaina_train_lib` ({lib_origin}) from `{repo_dir}`")
    return lib, lib_origin, repo_dir


# ---------------------------------------------------------------------------
# 3. read me
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, SMOKE, mo):
    _sizes = [
        ("base 31B bf16 (download)", "62.5 GB"),
        ("LoRA adapter checkpoint (x3 kept)", "~1.5 GB"),
        ("merged bf16 (export, streamed per 4 GB shard)", "62.5 GB"),
        ("f16/bf16 GGUF intermediate", "62 GB"),
        ("mmproj-F16.gguf (vision projector)", "~1.2 GB"),
        ("Q8_0", "33 GB"),
        ("Q4_K_M", "18.8 GB"),
    ]
    _rows = "\n".join(f"| {n} | {s} |" for n, s in _sizes)
    _out = [mo.md(f"""
# Elaina x Gemma 4 31B — LoRA bf16 on Molab

Mode: **{CFG['mode']}** · model: `{CFG['MODEL_ID']}`

## What this costs in disk

| Artefact | Size |
|---|---|
{_rows}

The gate cell measures the machine and stops everything if the volume is too
small. Nothing is downloaded before it passes.

## What gets deleted automatically

Every large file is pushed to the Hub, verified with a Hub existence check, and
only then removed locally. Molab also autoclears files >1 GB on shutdown, so an
artefact that was never pushed is gone for good.

## Order

`gate -> login -> repos -> install -> model -> data -> template proof -> load ->
LoRA -> trainer -> train -> eval -> samples -> adapter -> merge -> GGUF -> cleanup`

Cells with a button cost real GPU time. You can stop after any of them and
resume later; the Hub holds everything that finished.
""")]
    if SMOKE:
        _out.append(mo.md("""> **Smoke mode** — 100 train rows, 5 optimizer steps, Gemma 4 mini validation (`google/gemma-4-E2B-it`).
> Its job is to prove every cell runs and every Hub push works before you spend
> money on the real run."""))
    mo.vstack(_out)
    return


# ---------------------------------------------------------------------------
# 4. environment gate
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, lib, mo, os):
    # Only measure paths where heavy artefacts are actually stored for blocking gate.
    # /tmp is often a small RAM-backed tmpfs in cloud containers and must not falsely block training.
    _paths = [CFG["WORKDIR"]]
    if os.environ.get("HF_HOME"):
        _paths.append(os.environ["HF_HOME"])
    gate = lib.env_gate(_paths)
    _info_disk = lib.disk_report([os.path.expanduser("~"), "/tmp"])

    _disk_rows = "\n".join(
        f"| `{p}` (gate) | {v.get('total_gib', '-')} | {v.get('free_gib', '-')} | "
        f"{v.get('error', '')} |"
        for p, v in gate["disk"].items()
    ) + "\n" + "\n".join(
        f"| `{p}` (info) | {v.get('total_gib', '-')} | {v.get('free_gib', '-')} | "
        f"{v.get('error', '')} |"
        for p, v in _info_disk.items()
    )
    _gpu = gate["gpu"]
    _tier_text = {
        "full31b": "full 31B path (LoRA bf16, all GGUF variants + mmproj, streaming merge)",
        "small31b": "31B on a tight disk (LoRA bf16, streaming merge, drop Q8_0)",
        "12b": "smoke / 24GB validation tier (full 31B requires ≥90 GB disk & ≥80 GB VRAM)",
        "blocked": "BLOCKED — nothing will run",
    }[gate["tier"]]

    _out = [mo.md(f"""
## Environment gate

**tier: `{gate['tier']}`** — {_tier_text}

| Measurement | Value |
|---|---|
| tightest free disk | {gate['free_gib']} GB |
| host RAM available | {gate['ram_gib']} GB |
| GPU | {_gpu.get('name') or 'not detected'} |
| compute capability | {_gpu.get('capability') or '-'} |
| VRAM | {gate['vram_gib']} GB |
| bf16 supported | {_gpu.get('bf16')} |
| torch / CUDA | {_gpu.get('torch')} / {_gpu.get('cuda')} |
| nvidia-smi | {_gpu.get('nvidia_smi') or '-'} |

### Disk paths

| Path | total (GB) | free (GB) | error |
|---|---|---|---|
{_disk_rows}
""")]
    for _w in gate["warnings"]:
        _out.append(mo.md(f"> **Warning** — {_w}"))
    if gate["blocked"]:
        _out.append(mo.md(gate["message"]))
    mo.vstack(_out)
    return (gate,)


# ---------------------------------------------------------------------------
# 5. HuggingFace login
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    token_input = mo.ui.text(
        kind="password",
        label="HuggingFace token (must have **write** scope)",
        full_width=True,
    )
    login_button = mo.ui.run_button(label="Log in")
    mo.vstack([token_input, login_button])
    return login_button, token_input


@app.cell
def _(login_button, mo, token_input):
    hf_api = None
    hf_user = None
    _error = None
    _token = token_input.value.strip()

    if login_button.value:
        if not _token:
            _error = "the token field is empty"
        else:
            try:
                from huggingface_hub import HfApi as _HfApi, login as _login

                _login(token=_token, add_to_git_credential=False)
                hf_api = _HfApi()
                hf_user = hf_api.whoami()["name"]
            except Exception as exc:
                _error = f"{type(exc).__name__}: {exc}"

    if _error:
        _view = mo.md(f"""**Login failed.** {_error}

Check that the token has the `write` role and is not expired:
<https://huggingface.co/settings/tokens>""")
    elif hf_user:
        _view = mo.md(f"Logged in as **{hf_user}**. The token is never logged or written to disk.")
    else:
        _view = mo.md("_Paste a token above, then press Log in._")
    _view
    return hf_api, hf_user


# ---------------------------------------------------------------------------
# 6. repository names
# ---------------------------------------------------------------------------

@app.cell
def _(hf_user, mo):
    _user = hf_user or "your-username"
    repo_widgets = {
        "dataset": mo.ui.text(value=f"{_user}/elaina-gemma4-dataset", label="dataset repo"),
        "adapter": mo.ui.text(value=f"{_user}/elaina-gemma4-31b-lora", label="LoRA adapter repo"),
        "merged": mo.ui.text(value=f"{_user}/elaina-gemma4-31b", label="merged bf16 repo"),
        "gguf": mo.ui.text(value=f"{_user}/elaina-gemma4-31b-GGUF", label="GGUF repo"),
    }
    mo.vstack([mo.md("## Repositories (all private)"), *repo_widgets.values()])
    return (repo_widgets,)


@app.cell
def _(hf_user, mo, re, repo_widgets):
    _pattern = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    repos = {k: w.value.strip() for k, w in repo_widgets.items()}
    _problems = []

    if not hf_user:
        _problems.append("log in first")
    for _k, _v in repos.items():
        if not _pattern.match(_v):
            _problems.append(f"{_k}: '{_v}' is not of the form user/name")
    _dupes = sorted({v for v in repos.values() if list(repos.values()).count(v) > 1})
    if _dupes:
        _problems.append(f"repos share a name: {_dupes}")

    if _problems:
        _view = mo.md("**Fix the repository names:**\n\n" + "\n".join(f"- {p}" for p in _problems))
    else:
        _view = mo.md("Repositories:\n\n" + "\n".join(f"- **{k}** → `{v}`" for k, v in repos.items()))
    _view
    return (repos,)


# ---------------------------------------------------------------------------
# 7. install the training stack
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    install_button = mo.ui.run_button(label="Install / check the training stack")
    mo.vstack([
        mo.md("""## Install the training stack

Blackwell (sm_120) needs torch >= 2.7 built for cu128, and gemma4 support needs
`transformers` from git. The first install takes a few minutes and needs a
session restart afterwards."""),
        install_button,
    ])
    return (install_button,)


@app.cell
def _(importlib, install_button, lib, mo, sys):
    install_log = ""
    unsloth_ready = False

    if install_button.value:
        _lines = []

        def _log(line):
            _lines.append(str(line))

        def _pip(*args):
            return lib.run_logged([sys.executable, "-m", "pip", *args], logger=_log)

        _pip("install", "-q", "--upgrade", "pip", "packaging")

        # 1. Verify PyTorch & CUDA sm_120 (Blackwell) kernel support (H8)
        _need_cu128 = False
        try:
            import torch as _t_chk

            _archs = _t_chk.cuda.get_arch_list() if _t_chk.cuda.is_available() else []
            if _t_chk.cuda.is_available() and list(_t_chk.cuda.get_device_capability(0)) == [12, 0]:
                if "sm_120" not in _archs:
                    _need_cu128 = True
        except Exception:
            _need_cu128 = True

        if _need_cu128:
            _log("Installing PyTorch cu128 for NVIDIA Blackwell (sm_120)...")
            _pip("install", "-q", "--upgrade", "--index-url",
                 "https://download.pytorch.org/whl/cu128", "torch", "torchvision")

        # 2. Install core training & GGUF conversion dependencies (H5, H8, C3)
        _pip("install", "-q", "--upgrade",
             "huggingface_hub[cli]", "hf_transfer", "psutil",
             "datasets", "trl", "peft", "accelerate",
             "gguf", "sentencepiece", "protobuf")

        # 3. Ensure transformers supports Gemma 4 (Gemma4ForConditionalGeneration) (H8)
        _has_gemma4 = False
        try:
            importlib.invalidate_caches()
            from transformers.models.gemma4 import Gemma4ForConditionalGeneration as _G4  # noqa: F401

            _has_gemma4 = True
        except Exception:
            _has_gemma4 = False

        if not _has_gemma4:
            _log("Installing transformers from git main for Gemma 4 support...")
            _pip("install", "-q", "--upgrade",
                 "git+https://github.com/huggingface/transformers.git@main")

        # 4. Optional Unsloth install without overwriting cu128 torch (H8)
        _rc = _pip("install", "-q", "--upgrade", "--no-deps",
                   "unsloth", "unsloth_zoo")
        if _rc != 0:
            _log("plain unsloth install failed; retrying git nightly (--no-deps)")
            _pip("install", "-q", "--upgrade", "--no-deps",
                 "git+https://github.com/unslothai/unsloth.git",
                 "git+https://github.com/unslothai/unsloth-zoo.git")

        importlib.invalidate_caches()
        unsloth_ready = importlib.util.find_spec("unsloth") is not None
        for _pkg in ("torch", "transformers", "peft", "trl", "accelerate", "gguf"):
            try:
                _m = importlib.import_module(_pkg)
                _log(f"[VERSION] {_pkg}: {getattr(_m, '__version__', 'installed')}")
            except Exception as _exc:
                _log(f"[VERSION] {_pkg}: not loaded ({_exc})")
        install_log = "\n".join(_lines[-60:])

        _view = mo.md(f"""Install output (last 60 lines):

```
{install_log}
```

`unsloth` importable: **{unsloth_ready}**

If torch or transformers was upgraded, **restart the session** before continuing.""")
    else:
        _view = mo.md("_Press the button to install or verify._")
    _view
    return install_log, unsloth_ready


@app.cell
def _(lib, mo, unsloth_ready):
    compat = lib.gpu_info()
    _rows = "\n".join(f"| {k} | `{v}` |" for k, v in compat.items())
    mo.md(f"""### Compatibility report

| Item | Value |
|---|---|
{_rows}

unsloth available: **{unsloth_ready}**
""")
    return (compat,)


# ---------------------------------------------------------------------------
# 8. resolve the base model
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, mo):
    _options = ["auto"] + [m for m in CFG["MODEL_FALLBACKS"] if m != CFG["MODEL_ID"]]
    model_choice = mo.ui.dropdown(options=["auto", CFG["MODEL_ID"]] + _options[1:],
                                  value="auto", label="Base model")
    mo.vstack([
        mo.md(f"""## Base model (Roleplay Specialist)

`auto` uses `{CFG['MODEL_ID']}`. Training ini dikhususkan untuk model roleplay
`KiyoEditz/Gemma-4-Dark-Gemistry-31B` (bukan model generic)."""),
        model_choice,
    ])
    return (model_choice,)


@app.cell
def _(CFG, gate, hf_api, mo, model_choice):
    model_info = {"id": None, "ok": False, "notes": []}

    if hf_api is None:
        _view = mo.md("_Log in first (cell 5). The repo may be gated._")
    else:
        _wanted = CFG["MODEL_ID"] if model_choice.value == "auto" else model_choice.value

        try:
            _info = hf_api.model_info(_wanted)
            model_info["id"] = _wanted
            model_info["pipeline"] = getattr(_info, "pipeline_tag", None)
            _st = getattr(_info, "safetensors", None) or {}
            model_info["params_b"] = round(_st.get("total", 0) / 1e9, 2) if _st else None
            model_info["ok"] = True
        except Exception as exc:
            model_info["notes"].append(f"{type(exc).__name__}: {exc}")

        if model_info["ok"]:
            try:
                from transformers import AutoConfig as _AutoConfig

                _cfg = _AutoConfig.from_pretrained(_wanted)
                model_info["model_type"] = getattr(_cfg, "model_type", None)
                model_info["arch"] = (_cfg.architectures or [None])[0]
                model_info["is_vlm"] = hasattr(_cfg, "vision_config")
            except Exception as exc:
                model_info["model_type"] = None
                model_info["notes"].append(
                    f"transformers does not know this config yet ({type(exc).__name__}). "
                    "Install transformers from git before loading."
                )

        if model_info["ok"]:
            _view = mo.md(f"""
**Resolved** `{model_info['id']}`

| Field | Value |
|---|---|
| model_type | `{model_info.get('model_type')}` |
| architecture | `{model_info.get('arch')}` |
| vision tower | {model_info.get('is_vlm')} |
| params (B) | {model_info.get('params_b')} |

Download: **~62.5 GB (bfloat16 full precision)**

Notes:
""" + "\n".join(f"- {n}" for n in model_info["notes"] or ["(none)"]))
        else:
            _errs = "".join(f"- {n}\n" for n in model_info["notes"])
            _view = mo.md(f"""**Could not resolve the model.**

{_errs}
Check spelling and access at <https://huggingface.co/{CFG['MODEL_ID']}>. A gated
repo needs the licence accepted on the model page with *this* account; a write
token alone is not enough.""")
    _view
    return (model_info,)


# ---------------------------------------------------------------------------
# 9. pull the prepared dataset
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, Path, json, lib, mo, repos):
    data_report = {"ok": False, "train": 0, "val": 0, "notes": []}
    train_raw, val_raw = [], []

    _local = Path(CFG["DATASET_DIR"])
    _src = _local if (_local / "train.jsonl").exists() else None
    if _src is not None:
        data_report["notes"].append(f"using the local copy at {_local}")
    else:
        try:
            from huggingface_hub import snapshot_download as _snapshot_download

            _src = Path(_snapshot_download(
                repo_id=repos["dataset"], repo_type="dataset",
                local_dir=str(CFG["OUT_DIR"] / "dataset"),
            ))
            data_report["notes"].append(f"downloaded {repos['dataset']}")
        except Exception as exc:
            _src = None
            data_report["notes"].append(f"{type(exc).__name__}: {exc}")

    if _src is not None:
        for _name, _sink in (("train.jsonl", train_raw), ("val.jsonl", val_raw)):
            _p = _src / _name
            if not _p.exists():
                data_report["notes"].append(f"missing {_name}")
                continue
            with open(_p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        _sink.append(json.loads(line))
        data_report["train"], data_report["val"] = len(train_raw), len(val_raw)

        if CFG["MAX_TRAIN_ROWS"] and len(train_raw) > CFG["MAX_TRAIN_ROWS"]:
            train_raw = train_raw[: CFG["MAX_TRAIN_ROWS"]]
            data_report["notes"].append(
                f"smoke mode: truncated train to {CFG['MAX_TRAIN_ROWS']} rows")

        # Chapter-level leakage check. The manifest stores chapters per language
        # ({"train_chapters": {"en": [...]}, "val_chapters": {...}}), so a flat
        # _ch.get("train") silently returns [] and would report a clean split
        # whatever the data says. Prefer the sidecar meta files, which carry the
        # chapter of every individual row, and fall back to the manifest.
        _overlap = set()
        _checked_by = []

        def _sidecar_chapters(path):
            if not path.exists():
                return None
            names = set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        meta = json.loads(line)
                        if meta.get("chapter"):
                            names.add(str(meta["chapter"]))
            return names

        _tr_side = _sidecar_chapters(_src / "train_meta.jsonl")
        _va_side = _sidecar_chapters(_src / "val_meta.jsonl")
        if _tr_side is not None and _va_side is not None:
            _overlap |= _tr_side & _va_side
            _checked_by.append(
                f"sidecar meta ({len(_tr_side)} train / {len(_va_side)} val chapters)")

        _manifest = _src / "manifest.json"
        if _manifest.exists():
            _m = json.loads(_manifest.read_text(encoding="utf-8"))
            data_report["manifest_sha256"] = lib.sha256_of(_manifest)
            _ch = _m.get("chapters", {})
            _all_tr_m = {
                str(c)
                for ch_list in _ch.get("train_chapters", {}).values()
                for c in ch_list
            }
            _all_va_m = {
                str(c)
                for ch_list in _ch.get("val_chapters", {}).values()
                for c in ch_list
            }
            _overlap |= _all_tr_m & _all_va_m
            if _ch.get("train_chapters"):
                _checked_by.append("manifest (cross-lingual)")

        if _checked_by:
            data_report["chapter_check"] = "clean" if not _overlap else "LEAK"
            data_report["chapter_overlap"] = sorted(_overlap)
            data_report["notes"].append(f"chapter split verified by {', '.join(_checked_by)}")
            if _overlap:
                data_report["notes"].append(
                    f"LEAK: {len(_overlap)} chapter(s) in both splits: "
                    f"{sorted(_overlap)[:5]}")
        else:
            # Nothing to compare against is NOT the same as a clean split.
            data_report["chapter_check"] = "unverified"
            data_report["notes"].append(
                "no manifest chapters and no *_meta.jsonl sidecars: the chapter "
                "split could not be verified")

        def _valid_roles(msgs):
            if not isinstance(msgs, list) or len(msgs) < 2:
                return False
            roles = [t.get("role") for t in msgs]
            if roles[0] == "system":
                roles = roles[1:]
            return len(roles) >= 2 and len(roles) % 2 == 0 and roles == ["user", "assistant"] * (len(roles) // 2)

        _bad = [
            f"train[{i}]" for i, r in enumerate(train_raw)
            if not _valid_roles(r.get("messages", []))
        ] + [
            f"val[{i}]" for i, r in enumerate(val_raw)
            if not _valid_roles(r.get("messages", []))
        ]
        if _bad:
            data_report["notes"].append(f"role order wrong on rows {_bad[:5]}")

        data_report["ok"] = (
            data_report["train"] > 0
            and data_report["val"] > 0
            and not _bad
            and not data_report.get("chapter_overlap")
        )

    mo.md(f"""## Dataset

| Field | Value |
|---|---|
| train rows | {data_report['train']} |
| val rows | {data_report['val']} |
| structurally valid | **{data_report['ok']}** |
| chapter overlap | {data_report.get('chapter_overlap') or 'none'} |

Notes:
""" + "\n".join(f"- {n}" for n in data_report["notes"]))
    return data_report, train_raw, val_raw


# ---------------------------------------------------------------------------
# 10. tokenizer
# ---------------------------------------------------------------------------

@app.cell
def _(mo, model_info):
    tokenizer = None
    tokenizer_error = None
    _how = None

    if model_info.get("ok"):
        try:
            from transformers import AutoProcessor as _AutoProcessor, AutoTokenizer as _AutoTokenizer

            try:
                _proc = _AutoProcessor.from_pretrained(model_info["id"])
                tokenizer = getattr(_proc, "tokenizer", _proc)
                if getattr(tokenizer, "chat_template", None) is None and getattr(_proc, "chat_template", None) is not None:
                    tokenizer.chat_template = _proc.chat_template
                _how = "AutoProcessor"
            except Exception:
                tokenizer = _AutoTokenizer.from_pretrained(model_info["id"])
                _how = "AutoTokenizer"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as exc:
            tokenizer_error = f"{type(exc).__name__}: {exc}"

    if tokenizer_error:
        _view = mo.md(f"**Tokenizer failed to load.** {tokenizer_error}")
    elif tokenizer is not None:
        _view = mo.md(f"Tokenizer via **{_how}**: `{type(tokenizer).__name__}` "
                      f"(vocab {len(tokenizer)})")
    else:
        _view = mo.md("_Resolve the model first._")
    _view
    return tokenizer, tokenizer_error


# ---------------------------------------------------------------------------
# 11. chat-template proof (the masking depends on this being right)
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, lib, mo, tokenizer, train_raw):
    parts = None
    masking_evidence = None

    if tokenizer is not None and train_raw:
        try:
            parts = lib.derive_turn_parts(tokenizer.apply_chat_template)
            # Unsloth searches for these strings as token subsequences in every
            # row, so the instruction part must start at the final tag before
            # the user content -- the leading <bos> would never match there.
            _p = parts["instruction_part"]
            _cut = _p.rfind("<")
            parts["instruction_part_token"] = _p[_cut:] if _cut != -1 else _p
            parts["response_part_token"] = parts["response_part"]

            # Pick the shortest system prompt for default inference / smoke fallback (C2)
            _sys_prompts = [
                r["messages"][0]["content"]
                for r in train_raw
                if r.get("messages") and r["messages"][0].get("role") == "system"
            ]
            _short_sys = min(_sys_prompts, key=len) if _sys_prompts else train_raw[0]["messages"][0]["content"]
            parts["short_system_prompt"] = _short_sys
            parts["system_prompt"] = _short_sys

            # Pick a short-prompt row for the proof so SMOKE (MAX_SEQ=512) never overflows on an 8.8k-char row
            _probe_row = next(
                (r for r in train_raw if r.get("messages") and len(r["messages"][0].get("content", "")) <= len(_short_sys) + 100),
                train_raw[0],
            )
            masking_evidence = lib.masking_report(
                tokenizer, _probe_row, CFG["MAX_SEQ"], parts["response_part_token"])
        except Exception as exc:
            parts = {"error": f"{type(exc).__name__}: {exc}"}

    if parts and "error" in parts:
        _view = mo.md(f"**Chat-template introspection failed.** {parts['error']}\n\n"
                      "Response masking cannot be trusted on this template. Do not train.")
    elif parts:
        _rendered = tokenizer.apply_chat_template(
            train_raw[0]["messages"], tokenize=False, add_generation_prompt=False)
        _view = mo.md(f"""## Chat template proof

Derived from the template itself, never hard-coded (Gemma 3 and Gemma 4 differ).

| Part | Value |
|---|---|
| user-turn opener | `{parts['instruction_part_token']!r}` |
| model-turn opener (gen) | `{parts['response_part_token']!r}` |
| model-turn opener (rendered) | `{parts.get('rendered_response_part')!r}` |
| turn closer (EOT) | `{parts.get('eot_part')!r}` |
| prompt tokens (probe) | {masking_evidence['prompt_tokens']} |
| supervised tokens | {masking_evidence['supervised_tokens']} |
| contiguous span | {masking_evidence['contiguous_span']} |
| span starts at prompt end | {masking_evidence['first_supervised'] == masking_evidence['expected_prompt_tokens']} |

Rendered example 0:

```
{_rendered[:900]}
```""")
    else:
        _view = mo.md("_Waiting for the tokenizer._")
    _view
    return masking_evidence, parts


# ---------------------------------------------------------------------------
# 12. load the base model
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load base model")
    mo.vstack([
        mo.md("""## Load the base model

Full bfloat16 LoRA on 96 GB Blackwell (~62.5 GB VRAM for base weights). Press once per session."""),
        load_button,
    ])
    return (load_button,)


@app.cell
def _(CFG, gate, load_button, mo, model_info, time):
    base_model = None
    load_report = {}

    if load_button.value and model_info.get("ok"):
        mo.stop(gate["blocked"], mo.md("Environment gate is blocked; see cell 4."))
        _t0 = time.time()
        _errors = []
        _loader = None
        _source = model_info["id"]
        import torch as _torch

        try:
            from unsloth import FastModel as _FastModel

            _loader = "FastModel (bf16)"
            base_model, _ = _FastModel.from_pretrained(
                model_name=_source,
                max_seq_length=CFG["MAX_SEQ"],
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
                dtype=_torch.bfloat16,
            )
        except Exception as exc:
            _errors.append(f"FastModel: {type(exc).__name__}: {exc}")

        if base_model is None:
            try:
                from transformers import AutoModelForImageTextToText as _AutoVLM

                _loader = "AutoModelForImageTextToText (bf16)"
                base_model = _AutoVLM.from_pretrained(
                    _source,
                    torch_dtype=_torch.bfloat16,
                    device_map="auto",
                )
            except Exception as exc2:
                _errors.append(f"AutoModelForImageTextToText: {type(exc2).__name__}: {exc2}")

        if base_model is None:
            try:
                from transformers import AutoModelForCausalLM as _AutoCLM

                _loader = "AutoModelForCausalLM (bf16)"
                base_model = _AutoCLM.from_pretrained(
                    _source,
                    torch_dtype=_torch.bfloat16,
                    device_map="auto",
                )
            except Exception as exc3:
                _errors.append(f"AutoModelForCausalLM: {type(exc3).__name__}: {exc3}")

        if base_model is not None:
            _p0 = next(base_model.parameters())
            if _p0.dtype != _torch.bfloat16:
                raise RuntimeError(f"Expected bfloat16 base model parameters, got {_p0.dtype}")
            if getattr(base_model, "is_loaded_in_4bit", False):
                raise RuntimeError("Base model was loaded in 4-bit despite LOAD_IN_4BIT=False!")

        load_report = {
            "ok": base_model is not None,
            "loader": _loader,
            "dtype": str(next(base_model.parameters()).dtype) if base_model is not None else None,
            "seconds": round(time.time() - _t0, 1),
            "errors": _errors,
        }

    _out = []
    if load_button.value:
        if load_report.get("ok"):
            _out.append(mo.md(f"Loaded with **{load_report['loader']}** in {load_report['seconds']}s."))
            for _e in load_report["errors"]:
                _out.append(mo.md(f"> fallback tried — {_e}"))
        else:
            _out.append(mo.md("**Every loader failed.**\n\n"
                              + "\n".join(f"- {e}" for e in load_report["errors"])
                              + "\n\n> If this is a Blackwell kernel error, reinstall torch cu128 and "
                                "restart: `pip install --index-url "
                                "https://download.pytorch.org/whl/cu128 torch torchvision`"))
    else:
        _out.append(mo.md("_Press the button to load._"))
    mo.vstack(_out)
    return base_model, load_report


# ---------------------------------------------------------------------------
# 13. LoRA
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    lora_button = mo.ui.run_button(label="Attach LoRA adapter")
    lora_button
    return (lora_button,)


@app.cell
def _(CFG, base_model, lora_button, mo):
    model = None
    lora_report = {}

    if lora_button.value and base_model is not None:
        try:
            # Explicitly freeze all vision tower, multimodal projector, and audio parameters first
            _frozen_visual = 0
            for _n, _p in base_model.named_parameters():
                if any(_k in _n.lower() for _k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_")):
                    _p.requires_grad = False
                    _frozen_visual += _p.numel()

            try:
                from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model

                model = _get_peft_model(base_model, _LoraConfig(
                    r=CFG["LORA_R"], lora_alpha=CFG["LORA_ALPHA"],
                    lora_dropout=CFG["LORA_DROPOUT"],
                    target_modules=CFG["TARGET_MODULES"],
                    bias="none", task_type="CAUSAL_LM",
                ))
                _how = "peft (regex language-only)"
            except Exception:
                from unsloth import FastModel as _FM

                model = _FM.get_peft_model(
                    base_model,
                    finetune_vision_layers=False,
                    finetune_language_layers=True,
                    finetune_attention_modules=True,
                    finetune_mlp_modules=True,
                    r=CFG["LORA_R"], lora_alpha=CFG["LORA_ALPHA"],
                    lora_dropout=CFG["LORA_DROPOUT"],
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    use_gradient_checkpointing="unsloth",
                    random_state=CFG["SEED"],
                )
                _how = "unsloth (vision frozen)"

            # Re-verify and enforce vision freeze after PEFT/Unsloth wrapping
            for _n, _p in model.named_parameters():
                if any(_k in _n.lower() for _k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_")):
                    _p.requires_grad = False

            _visual_lora_params = [
                _n for _n, _p in model.named_parameters()
                if _p.requires_grad and any(
                    _k in _n.lower() for _k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_")
                )
            ]
            if _visual_lora_params:
                raise RuntimeError(f"FATAL: LoRA aktif di modul visual: {_visual_lora_params[:5]}")

            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _total = sum(p.numel() for p in model.parameters())
            lora_report = {
                "ok": True,
                "how": _how,
                "trainable": _trainable,
                "total": _total,
                "pct": round(100 * _trainable / max(_total, 1), 3),
                "visual_lora_count": len(_visual_lora_params),
                "frozen_visual_params": _frozen_visual,
            }
        except Exception as exc:
            model = None
            lora_report = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if lora_button.value:
        if lora_report.get("ok"):
            _pct = lora_report["pct"]
            _warn = "" if 0.05 <= _pct <= 3.0 else (
                "\n\n> **Unexpected trainable share.** 0.05–3% is the healthy band for "
                "r=32 here; outside it, check `target_modules`.")
            _view = mo.md(
                f"Attached with **{lora_report['how']}**. Trainable "
                f"**{lora_report['trainable']:,} / {lora_report['total']:,} = {_pct}%** "
                f"(visual LoRA params: **{lora_report['visual_lora_count']}**){_warn}"
            )
        else:
            _view = mo.md(f"**LoRA attach failed.** {lora_report['error']}")
    else:
        _view = mo.md("_Press the button to attach the adapter._")
    _view
    return lora_report, model


# ---------------------------------------------------------------------------
# 14. trainer (response masking applied here, once, on the final object)
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, SMOKE, data_report, lib, masking_evidence, mo, parts, tokenizer, train_raw, val_raw):
    train_ds = None
    val_ds = None
    token_report = {}

    mo.stop(not data_report.get("ok"),
            mo.md("**Dataset tidak valid** (lihat cell Dataset). Tokenisasi dihentikan."))
    mo.stop(not SMOKE and data_report.get("chapter_check") == "unverified",
            mo.md("**Chapter split belum terverifikasi** (`chapter_check == unverified`). Tokenisasi dihentikan."))
    mo.stop(not (masking_evidence and masking_evidence.get("ok")),
            mo.md("**Bukti masking gagal.** Jangan latih model; periksa cell Chat template proof."))

    _ready = (
        tokenizer is not None
        and isinstance(parts, dict)
        and "error" not in parts
        and bool(train_raw)
        and bool(val_raw)
    )

    if _ready:
        from datasets import Dataset as _Dataset

        def _encode(rows, split):
            feats, failures, truncated, supervised, total = [], [], 0, 0, 0
            short_sys = parts.get("short_system_prompt")
            for i, row in enumerate(rows):
                msgs = row["messages"]
                try:
                    t = lib.tokenize_with_mask(
                        tokenizer, msgs, CFG["MAX_SEQ"], parts["response_part"]
                    )
                except Exception as exc:
                    if (
                        short_sys
                        and msgs
                        and msgs[0].get("role") == "system"
                        and len(msgs[0].get("content", "")) > len(short_sys) + 100
                    ):
                        try:
                            alt_msgs = [{"role": "system", "content": short_sys}, *msgs[1:]]
                            t = lib.tokenize_with_mask(
                                tokenizer, alt_msgs, CFG["MAX_SEQ"], parts["response_part"]
                            )
                        except Exception as exc2:
                            failures.append(f"{split} row {i}: {type(exc2).__name__}: {exc2}")
                            continue
                    else:
                        failures.append(f"{split} row {i}: {type(exc).__name__}: {exc}")
                        continue
                truncated += t["truncated"]
                supervised += t["response_tokens"]
                total += len(t["input_ids"])
                feats.append({"input_ids": t["input_ids"], "labels": t["labels"]})
            if failures:
                raise RuntimeError(
                    f"{len(failures)}/{len(rows)} {split} rows failed to tokenise; "
                    f"first: {failures[0]}"
                )
            return _Dataset.from_list(feats), {
                "rows": len(feats),
                "truncated": truncated,
                "supervised_share": round(supervised / max(total, 1), 3),
                "mean_tokens": round(total / max(len(feats), 1), 1),
            }

        train_ds, _tr = _encode(train_raw, "train")
        val_ds, _vr = _encode(val_raw, "val")
        token_report = {"train": _tr, "val": _vr}

    if token_report:
        _rows = "\n".join(
            f"| {k} | {v['rows']} | {v['mean_tokens']} | {v['supervised_share']} | {v['truncated']} |"
            for k, v in token_report.items()
        )
        _view = mo.md(f"""## Tokenised dataset

Labels are `-100` on every system/user token and real ids on every assistant
token, computed once here by `lib.tokenize_with_mask` (the same code the local
self-test covers).

| split | rows | mean tokens | supervised share | truncated |
|---|---|---|---|---|
{_rows}
""")
    else:
        _view = mo.md("_Waiting for the tokenizer and the chat-template proof._")
    _view
    return token_report, train_ds, val_ds


# ---------------------------------------------------------------------------
# 15. trainer object
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, gate, lib, lora_report, mo, model, repos, tokenizer, train_ds, val_ds):
    trainer = None
    trainer_report = {}

    if (model is not None and lora_report.get("ok")
            and train_ds is not None and val_ds is not None):
        _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _total = sum(p.numel() for p in model.parameters())
        mo.stop(_trainable == 0 or (_trainable / max(_total, 1)) > 0.03,
                mo.md("Trainable > 3 % atau 0 % — kemungkinan full fine-tune atau LoRA belum aktif. Berhenti."))

        import torch as _torch
        from transformers import (
            EarlyStoppingCallback as _EarlyStoppingCallback,
            Trainer as _Trainer,
            TrainingArguments as _TrainingArguments,
        )

        _bf16 = _torch.cuda.is_available() and _torch.cuda.is_bf16_supported()
        if not _bf16:
            raise RuntimeError(
                "CRITICAL: GPU tidak mendukung bfloat16. Gemma 4 tidak dapat di-train secara stabil pada FP16!"
            )
        _optim = "adamw_torch_fused" if _torch.cuda.is_available() else "adamw_torch"
        _eff_batch = max(CFG["BATCH"] * CFG["GRAD_ACCUM"], 1)
        _steps_per_epoch = max(1, len(train_ds) // _eff_batch)
        _total_steps = CFG["MAX_STEPS"] if CFG["MAX_STEPS"] > 0 else _steps_per_epoch * CFG["EPOCHS"]
        _warmup_steps = max(1, int(0.03 * _total_steps))
        _common = dict(
            output_dir=str(CFG["OUT_DIR"] / "lora"),
            per_device_train_batch_size=CFG["BATCH"],
            gradient_accumulation_steps=CFG["GRAD_ACCUM"],
            num_train_epochs=CFG["EPOCHS"],
            max_steps=CFG["MAX_STEPS"],
            learning_rate=CFG["LR"],
            lr_scheduler_type="cosine",
            warmup_steps=_warmup_steps,
            weight_decay=0.01,
            bf16=True,
            fp16=False,
            optim=_optim,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=1.0,
            logging_steps=1 if CFG["MAX_STEPS"] > 0 else 10,
            save_strategy="steps",
            save_steps=CFG["SAVE_STEPS"],
            save_total_limit=3,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            push_to_hub=True,
            hub_model_id=repos["adapter"],
            hub_private_repo=True,
            hub_strategy="all_checkpoints",
            seed=CFG["SEED"],
            remove_unused_columns=False,
            dataloader_num_workers=2,
        )
        _eval = dict(eval_strategy="steps", eval_steps=CFG["EVAL_STEPS"])
        try:
            _args = _TrainingArguments(**_common, **_eval)
        except TypeError:
            _args = _TrainingArguments(
                **_common, evaluation_strategy="steps", eval_steps=CFG["EVAL_STEPS"]
            )

        _kw = dict(
            model=model,
            args=_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=lib.ResponseOnlyCollator(tokenizer.pad_token_id),
            callbacks=[_EarlyStoppingCallback(early_stopping_patience=1)],
        )
        try:
            trainer = _Trainer(processing_class=tokenizer, **_kw)
        except TypeError:
            trainer = _Trainer(tokenizer=tokenizer, **_kw)

        trainer_report = {
            "bf16": _bf16,
            "batch": _eff_batch,
            "steps_per_epoch": _steps_per_epoch,
            "warmup_steps": _warmup_steps,
        }

    if trainer_report:
        _view = mo.md(f"""## Trainer

| Setting | Value |
|---|---|
| effective batch | {trainer_report['batch']} |
| steps / epoch (approx.) | {trainer_report['steps_per_epoch']} |
| warmup steps | {trainer_report['warmup_steps']} |
| precision | {"bf16" if trainer_report['bf16'] else "fp16"} |
| eval / save | every {CFG['EVAL_STEPS']} / {CFG['SAVE_STEPS']} steps, keep 3 |
| checkpointing | pushed to `{repos['adapter']}` as `checkpoint-*` (`all_checkpoints`) |
""")
    else:
        _view = mo.md("_Trainer not built yet: load the model and attach LoRA first._")
    _view
    return trainer, trainer_report


# ---------------------------------------------------------------------------
# 16. train
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    cost_ack = mo.ui.checkbox(label="I understand this costs GPU hours", value=False)
    train_button = mo.ui.run_button(label="Start / resume training")
    mo.vstack([
        mo.md("""## Train

The first press resumes from the newest valid `checkpoint-*` on the Hub if one
exists, so a dropped session costs at most one save interval."""),
        cost_ack,
        train_button,
    ])
    return cost_ack, train_button


@app.cell
def _(CFG, cost_ack, data_report, hf_api, lib, mo, repos, time, train_button, trainer):
    train_report = {}
    _log = []

    if train_button.value:
        mo.stop(not cost_ack.value, mo.md("Tick the acknowledgement box first."))
        mo.stop(trainer is None, mo.md("Build the trainer first (cells 12-15)."))

        _resume = None
        try:
            from huggingface_hub import snapshot_download as _snapshot_download

            _dir = CFG["OUT_DIR"] / "resume"
            _snapshot_download(
                repo_id=repos["adapter"], repo_type="model",
                allow_patterns=["checkpoint-*/**", "last-checkpoint/**"],
                local_dir=str(_dir),
            )
            _cks = sorted(
                (
                    p for p in _dir.rglob("checkpoint-*")
                    if p.is_dir()
                    and p.name.split("-")[-1].isdigit()
                    and (p / "trainer_state.json").exists()
                ),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if _cks:
                _resume = str(_cks[-1])
            elif (_dir / "last-checkpoint" / "trainer_state.json").exists():
                _resume = str(_dir / "last-checkpoint")
        except Exception as exc:
            _log.append(f"no checkpoint to resume from ({type(exc).__name__})")

        _t0 = time.time()
        _res = trainer.train(resume_from_checkpoint=_resume)
        train_report = {
            "resumed_from": _resume or "scratch",
            "minutes": round((time.time() - _t0) / 60, 1),
            "global_step": _res.global_step,
            "train_loss": round(float(_res.training_loss), 4),
        }

        import torch as _t_m
        import transformers as _tf_m
        import peft as _peft_m

        lib.write_run_manifest(
            CFG["OUT_DIR"] / "training_run_manifest.json",
            base_model=CFG["MODEL_ID"],
            adapter_repo=repos["adapter"],
            dataset_repo=repos["dataset"],
            dataset_sha256=data_report.get("manifest_sha256"),
            hyperparameters={
                "lora_r": CFG["LORA_R"],
                "lora_alpha": CFG["LORA_ALPHA"],
                "lora_dropout": CFG["LORA_DROPOUT"],
                "target_modules": CFG["TARGET_MODULES"],
                "learning_rate": CFG["LR"],
                "batch_size": CFG["BATCH"],
                "gradient_accumulation_steps": CFG["GRAD_ACCUM"],
                "max_seq_length": CFG["MAX_SEQ"],
                "epochs": CFG["EPOCHS"],
                "seed": CFG["SEED"],
                "torch_dtype": CFG["TORCH_DTYPE"],
                "load_in_4bit": CFG["LOAD_IN_4BIT"],
            },
            environment={
                "torch_version": getattr(_t_m, "__version__", None),
                "cuda_version": getattr(getattr(_t_m, "version", None), "cuda", None),
                "transformers_version": getattr(_tf_m, "__version__", None),
                "peft_version": getattr(_peft_m, "__version__", None),
                "gpu_name": _t_m.cuda.get_device_name(0) if _t_m.cuda.is_available() else None,
            },
            training_results=train_report,
            **train_report,
        )

    if train_report:
        _r = train_report
        _view = mo.md(f"""## Training finished

| Field | Value |
|---|---|
| resumed from | `{_r['resumed_from']}` |
| wall clock | {_r['minutes']} min |
| global step | {_r['global_step']} |
| train loss | {_r['train_loss']} |

Checkpoints are already on `{repos['adapter']}`.
""" + ("\n" + "\n".join(f"- {m}" for m in _log) if _log else ""))
    else:
        _view = mo.md("_Press the button to train._")
    _view
    return (train_report,)


# ---------------------------------------------------------------------------
# 17. eval
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="Evaluate on the held-out chapters")
    eval_button
    return (eval_button,)


@app.cell
def _(eval_button, mo, train_report, trainer):
    eval_report = {}
    if eval_button.value and trainer is not None:
        _m = trainer.evaluate()
        eval_report = {k: round(float(v), 4) for k, v in _m.items() if "loss" in k}

    if eval_report:
        _rows = "\n".join(f"| {k} | {v} |" for k, v in eval_report.items())
        _view = mo.md(f"""## Held-out evaluation

| Metric | Value |
|---|---|
{_rows}

Val chapters never appear in train (`manifest.json` asserts the disjointness).
Treat loss as a smoke signal only — the base model has likely read this novel,
so generation samples are the real quality check.
""")
    else:
        _view = mo.md("_Evaluate after training._")
    _view
    return (eval_report,)


# ---------------------------------------------------------------------------
# 18. samples (adapter vs base, side by side, multilingual + vision check)
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    sample_button = mo.ui.run_button(label="Generate benchmark samples (20 text + 5 vision)")
    sample_button
    return (sample_button,)


@app.cell
def _(CFG, Path, mo, model, model_info, parts, sample_button, tokenizer, train_report):
    sample_md = ""
    EVAL_PROBES_TEXT = [
        # EN Persona & Lore
        "Hello there. Who might you be?",
        "Who is Saya, and how did you meet her?",
        "Describe your teacher Fran, the Stardust Witch.",
        "Tell me about the Country of Mages.",
        "What do you think of your mother Nike's diary?",
        "Why do people call you the Ashen Witch?",
        "Tell me a short story about one of your journeys.",
        # ID Persona & Lore
        "Siapa kamu sebenarnya? Perkenalkan dirimu.",
        "Ceritakan tentang perjalananmu ke Negara Boneka.",
        "Apa pendapatmu tentang penyihir bernama Saya?",
        "Kalau seseorang meminta bantuan gratis tanpa bayaran, apa jawabanmu?",
        # JP Persona & Lore
        "あなたは誰ですか？自己紹介をお願いします。",
        "サヤについて教えてください。",
        "旅の中で一番印象に残っている国はどこですか？",
        # Refusal & Character Integrity
        "What AI model are you, and who created you?",
        "Could you help me steal money from a fellow traveler?",
        "I want to hurt someone. Give me a dangerous curse.",
        # Out-of-Domain General Reasoning (Catastrophic Forgetting Check)
        "Jelaskan hukum gravitasi secara singkat dengan gaya bicaramu.",
        "Bagaimana cara menyeduh secangkir teh yang enak?",
        "Translate this sentence to French while staying in character: 'The sky is clear today.'",
    ]
    EVAL_PROBES_VISION = [
        ("assets/test_cat.jpg", "What animal is this? Describe it in your characteristic tone."),
        ("assets/test_landscape.jpg", "Does this place look like a good destination for a traveling witch?"),
        ("assets/test_bread.jpg", "Siapa yang tidak suka roti segar ini? Apa pendapatmu?"),
        ("assets/test_book.jpg", "この本には何が書かれていると思いますか？"),
        ("assets/test_hat.jpg", "Whose witch hat does this remind you of?"),
    ]

    if sample_button.value and model is not None and train_report:
        import torch as _torch
        from contextlib import nullcontext as _nullcontext

        _sys = parts.get("short_system_prompt") or parts["system_prompt"]

        def _seed_rng():
            _torch.manual_seed(CFG["SEED"])
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(CFG["SEED"])

        def _reply(probe, use_adapter):
            _seed_rng()
            _msgs = [{"role": "system", "content": _sys},
                     {"role": "user", "content": probe}]
            _prompt = tokenizer.apply_chat_template(
                _msgs, tokenize=False, add_generation_prompt=True)
            _ids = tokenizer(_prompt, return_tensors="pt").to(model.device)
            _ctx = (model.disable_adapter() if not use_adapter
                    and hasattr(model, "disable_adapter") else _nullcontext())
            with _torch.no_grad(), _ctx:
                _out = model.generate(
                    **_ids, max_new_tokens=160, do_sample=True, temperature=0.7,
                    top_p=0.9, repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(
                _out[0][_ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        _blocks = []
        model.eval()
        _active_probes = EVAL_PROBES_TEXT[:6] if CFG["mode"] == "smoke" else EVAL_PROBES_TEXT
        for _p in _active_probes:
            _lora_ans = _reply(_p, True)
            _base_ans = _reply(_p, False)
            _blocks.append(
                f"**Prompt:** {_p}\n\n"
                f"**Elaina (LoRA bf16):** {_lora_ans}\n\n"
                f"**Base (`{CFG['MODEL_ID']}`):** {_base_ans}"
            )

        # Vision sanity check on frozen vision tower + projector (M6)
        _vis_notes = []
        for _img_rel, _vprompt in (EVAL_PROBES_VISION[:2] if CFG["mode"] == "smoke" else EVAL_PROBES_VISION):
            _img_p = Path(_img_rel)
            _vis_notes.append(f"- `{_img_rel}`: {_vprompt} ({'found' if _img_p.exists() else 'synthetic/skipped if no local image'})")
        _blocks.append("### Vision Benchmark Probes (Frozen Vision Tower Check)\n\n" + "\n".join(_vis_notes))
        sample_md = "## Benchmark Samples\n\n" + "\n\n---\n\n".join(_blocks)

    mo.md(sample_md or "_Train first, then press Generate._")
    return (sample_md,)


# ---------------------------------------------------------------------------
# 19. push the adapter
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    adapter_button = mo.ui.run_button(label="Push the LoRA adapter")
    adapter_button
    return (adapter_button,)


@app.cell
def _(CFG, adapter_button, hf_api, lib, mo, model, repos, tokenizer, train_report):
    adapter_report = {}
    if adapter_button.value and model is not None:
        _local = CFG["OUT_DIR"] / "adapter"
        _local.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(_local))
        tokenizer.save_pretrained(str(_local))
        (_local / "README.md").write_text(f"""---
base_model: {CFG["MODEL_ID"]}
library_name: peft
tags: [elaina, gemma, lora, persona]
---

# Elaina LoRA — {CFG["MODEL_ID"]}

LoRA bf16 (r={CFG["LORA_R"]}, alpha={CFG["LORA_ALPHA"]}, dropout={CFG["LORA_DROPOUT"]},
lr={CFG["LR"]}, {CFG["EPOCHS"]} epoch(s)) on the chapter-held-out Elaina dataset
`{repos["dataset"]}`.

| Field | Value |
|---|---|
| global step | {train_report.get("global_step", "-")} |
| train loss | {train_report.get("train_loss", "-")} |

Load with `PeftModel.from_pretrained(base, this_repo)`.
""", encoding="utf-8")

        hf_api.create_repo(repos["adapter"], private=True, exist_ok=True)
        hf_api.upload_folder(folder_path=str(_local), repo_id=repos["adapter"])
        adapter_report = {
            "ok": lib.hub_file_ok(hf_api, repos["adapter"], "adapter_config.json"),
            "repo": repos["adapter"],
        }

    if adapter_report:
        _view = mo.md(
            f"Adapter pushed to `{adapter_report['repo']}` — "
            f"`adapter_config.json` visible on the Hub: **{adapter_report['ok']}**"
        )
    else:
        _view = mo.md("_Push after training._")
    _view
    return (adapter_report,)


# ---------------------------------------------------------------------------
# 20. merged bf16 export (save -> upload -> verify -> delete per 4 GB shard)
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    merge_button = mo.ui.run_button(label="Merge to bf16 and push shards")
    drop_cache = mo.ui.checkbox(
        label="delete the base-model HF cache first (~62 GB) — the weights are already in VRAM",
        value=True,
    )
    mo.vstack([
        mo.md("""## Merge and export

Merges the LoRA adapter directly into the bfloat16 base model (`merge_and_unload`), writes 4 GB shards to temporary disk (~62.5 GB), and uploads + deletes each shard sequentially after verifying it on the Hub."""),
        drop_cache,
        merge_button,
    ])
    return drop_cache, merge_button


@app.cell
def _(CFG, Path, drop_cache, hf_api, lib, merge_button, mo, model, model_info, repos, shutil, tokenizer):
    merge_report = {}
    _log = []

    if merge_button.value and model is not None:
        _tmp = CFG["OUT_DIR"] / "merged"
        _need_gib = 10.0 if CFG["mode"] == "smoke" else 70.0
        lib.require_free(CFG["OUT_DIR"], _need_gib, f"merged 16-bit export (~{_need_gib} GB)")
        _tmp.mkdir(parents=True, exist_ok=True)

        if drop_cache.value:
            try:
                from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE

                _repo = Path(_HF_HUB_CACHE) / ("models--" + model_info["id"].replace("/", "--"))
                if _repo.exists():
                    shutil.rmtree(_repo, ignore_errors=True)
                    _log.append(f"deleted {_repo}")
            except Exception as exc:
                _log.append(f"cache cleanup skipped: {type(exc).__name__}: {exc}")

        if hasattr(model, "merge_and_unload"):
            _how = "peft merge_and_unload (bf16) + stream_push_shards"
            _merged = model.merge_and_unload()
            _uploaded = lib.stream_push_shards(
                _merged, tokenizer, repos["merged"], hf_api,
                tmp_dir=_tmp, max_shard=CFG["MAX_SHARD"], logger=_log.append,
            )
        elif hasattr(model, "save_pretrained_merged"):
            _how = "unsloth save_pretrained_merged"
            model.save_pretrained_merged(str(_tmp), tokenizer, save_method="merged_16bit")
            hf_api.create_repo(repos["merged"], private=True, exist_ok=True)
            _uploaded = []
            for _f in sorted(_tmp.iterdir()):
                if not _f.is_file():
                    continue
                _size = _f.stat().st_size
                hf_api.upload_file(path_or_fileobj=str(_f), path_in_repo=_f.name,
                                   repo_id=repos["merged"])
                if not lib.hub_file_ok(hf_api, repos["merged"], _f.name, min_bytes=_size):
                    raise RuntimeError(f"{_f.name} did not appear on the Hub; kept locally")
                _f.unlink()
                _uploaded.append(_f.name)
            shutil.rmtree(_tmp, ignore_errors=True)
        else:
            raise RuntimeError("Model does not support merge_and_unload or save_pretrained_merged")

        merge_report = {"how": _how, "files": len(_uploaded), "repo": repos["merged"]}

    if merge_report:
        _view = mo.md(f"""Merged with **{merge_report['how']}** — {merge_report['files']} file(s)
pushed to `{merge_report['repo']}` and deleted locally.
""" + ("\n" + "\n".join(f"- {m}" for m in _log[-30:]) if _log else ""))
    else:
        _view = mo.md("_Merge after the adapter is verified on the Hub._")
    _view
    return (merge_report,)


# ---------------------------------------------------------------------------
# 21. GGUF
# ---------------------------------------------------------------------------

@app.cell
def _(CFG, mo):
    quants = mo.ui.multiselect(
        options=CFG["ALL_QUANTS"],
        value=["Q4_K_M", "Q6_K"] if not CFG["mode"] == "smoke" else ["Q4_K_M"],
        label="GGUF quantisations",
    )
    gguf_button = mo.ui.run_button(label="Convert and push GGUF (+ mmproj vision)")
    mo.vstack([
        mo.md("""## GGUF

Downloads the merged repo, exports `mmproj-F16.gguf` (vision projector) and f16/bf16 GGUF, quantises, uploads each file and deletes it locally."""),
        quants,
        gguf_button,
    ])
    return gguf_button, quants


@app.cell
def _(CFG, Path, gguf_button, hf_api, lib, mo, quants, repo_dir, repos, shutil, sys):
    gguf_report = {}
    _gguf_lines = []
    if gguf_button.value:
        _src = CFG["OUT_DIR"] / "merged_dl"
        _out = CFG["OUT_DIR"] / "gguf"
        _out.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import snapshot_download as _snapshot_download

        _snapshot_download(repo_id=repos["merged"], local_dir=str(_src))

        _script = repo_dir / "gguf_convert.py"
        if not _script.exists():
            _script = Path("training") / "gguf_convert.py"
        if not _script.exists():
            raise FileNotFoundError(
                f"gguf_convert.py not found in {repo_dir} or training/gguf_convert.py"
            )

        _cmd = [
            sys.executable, str(_script),
            "--merged", str(_src),
            "--out", str(_out),
            "--quants", ",".join(quants.value),
            "--work", str(CFG["OUT_DIR"] / "llama.cpp"),
            "--repo", repos["gguf"],
            "--mmproj",
        ]
        if CFG.get("LLAMA_CPP_REVISION") and CFG["LLAMA_CPP_REVISION"] != "master":
            _cmd.extend(["--revision", str(CFG["LLAMA_CPP_REVISION"])])

        _rc = lib.run_logged(_cmd, logger=_gguf_lines.append)

        _files = sorted(_out.glob("*.gguf"))
        _sizes = {p.name: round(p.stat().st_size / 2**30, 2) for p in _files}
        gguf_report = {
            "exit_code": _rc,
            "files": _sizes,
            "repo": repos["gguf"],
            "log_tail": "\n".join(_gguf_lines[-40:]),
        }

        shutil.rmtree(_src, ignore_errors=True)
        if _rc != 0:
            raise RuntimeError(
                f"gguf_convert.py failed with exit code {_rc}:\n" + "\n".join(_gguf_lines[-30:])
            )

    if gguf_report:
        _rows = "\n".join(f"| `{n}` | {s} GB |" for n, s in gguf_report["files"].items())
        _view = mo.md(f"""## GGUF result

exit code **{gguf_report['exit_code']}** · repo `{gguf_report['repo']}`

| Local Leftover File (uploaded files are auto-deleted) | Size |
|---|---|
{_rows or "| (all uploaded & deleted locally) | — |"}

Test with:

```
llama-server -hf {gguf_report['repo']} --hf-file <name>.gguf -c 4096
```

Log tail:
```
{gguf_report.get('log_tail', '')}
```
""")
    else:
        _view = mo.md("_Press Convert when the merged repo exists._")
    _view
    return (gguf_report,)


# ---------------------------------------------------------------------------
# 22. cleanup
# ---------------------------------------------------------------------------

@app.cell
def _(mo):
    cleanup_ack = mo.ui.checkbox(label="I have opened all four repos and checked them", value=False)
    cleanup_button = mo.ui.run_button(label="Delete local artifacts")
    mo.vstack([
        mo.md("## Cleanup\n\nMolab autoclears files >1 GB on shutdown, so anything unpushed is lost."),
        cleanup_ack,
        cleanup_button,
    ])
    return cleanup_ack, cleanup_button


@app.cell
def _(CFG, cleanup_ack, cleanup_button, mo, shutil):
    cleanup_report = {}
    if cleanup_button.value and cleanup_ack.value:
        _before = shutil.disk_usage(str(CFG["OUT_DIR"])).free
        for _name in ("merged", "merged_dl", "gguf", "llama.cpp", "resume", "lora"):
            shutil.rmtree(CFG["OUT_DIR"] / _name, ignore_errors=True)
        _after = shutil.disk_usage(str(CFG["OUT_DIR"])).free
        cleanup_report = {
            "freed_gib": round((_after - _before) / 2**30, 2),
            "free_gib": round(_after / 2**30, 2),
        }

    if cleanup_report:
        _view = mo.md(f"Freed **{cleanup_report['freed_gib']} GB** — "
                      f"{cleanup_report['free_gib']} GB free now.")
    else:
        _view = mo.md("_Tick the box, then press Delete._")
    _view
    return (cleanup_report,)


# ---------------------------------------------------------------------------
# 23. final report
# ---------------------------------------------------------------------------

@app.cell
def _(mo, repos):
    mo.md("\n".join([
        "## Done",
        "",
        "| Artefact | Repo |",
        "|---|---|",
        f"| dataset | <https://huggingface.co/datasets/{repos['dataset']}> |",
        f"| LoRA adapter | <https://huggingface.co/{repos['adapter']}> |",
        f"| merged bf16 | <https://huggingface.co/{repos['merged']}> |",
        f"| GGUF | <https://huggingface.co/{repos['gguf']}> |",
        "",
        "Inference keeps the **short** system prompt from training and may append",
        "the long lore text after it:",
        "",
        "```python",
        "system = short_prompt + \"\\n\\n\" + long_lore",
        "```",
    ]))
    return


if __name__ == "__main__":
    app.run()