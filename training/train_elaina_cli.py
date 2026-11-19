#!/usr/bin/env python3
"""
train_elaina_cli.py

Terminal-first CLI pipeline for fine-tuning Elaina on Gemma 4 (bf16 LoRA, frozen
vision tower) without the Marimo UI. Uses the exact same safety gates, masking
proof, Hub auto-resume, streamed bf16 merge, and GGUF (+mmproj) export as
molab_train_elaina.py.

Usage:
    # 1. Configure .env (or export env vars)
    #    HF_TOKEN=hf_...
    #    ELAINA_MODE=full          # or 'smoke'
    #    ELAINA_DATASET_REPO=you/elaina-gemma4-dataset
    #
    # 2. Run end-to-end
    python training/train_elaina_cli.py --mode smoke
    python training/train_elaina_cli.py --mode full

    # Optional: run only specific stages
    python training/train_elaina_cli.py --mode full --skip-gguf
    python training/train_elaina_cli.py --mode full --stage export
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure training/ is on sys.path so `import elaina_train_lib` works from repo root or training/
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import elaina_train_lib as lib  # noqa: E402


def load_dotenv_file(env_path: Path) -> None:
    """Minimal .env loader (does not overwrite already-exported environment variables)."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def build_cfg(args) -> dict:
    mode = (args.mode or os.environ.get("ELAINA_MODE", "full")).lower()
    smoke = mode == "smoke"
    workdir = Path(args.workdir or os.environ.get("ELAINA_WORKDIR", "./work")).resolve()
    quants_env = os.environ.get("ELAINA_QUANTS", "")
    default_quants = ["Q4_K_M"] if smoke else ["Q4_K_M", "Q6_K"]
    quants = [q.strip() for q in quants_env.split(",") if q.strip()] if quants_env else default_quants
    if args.quants:
        quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    return {
        "mode": mode,
        "smoke": smoke,
        "MODEL_ID": (
            os.environ.get("ELAINA_SMOKE_MODEL", "google/gemma-4-E2B-it")
            if smoke
            else os.environ.get("ELAINA_MODEL_ID", "KiyoEditz/Gemma-4-Dark-Gemistry-31B")
        ),
        "MAX_SEQ": 512 if smoke else int(os.environ.get("ELAINA_MAX_SEQ", "4096")),
        "LORA_R": 32,
        "LORA_ALPHA": 64,
        "LORA_DROPOUT": 0.05,
        "TARGET_MODULES": (
            r"^(?!.*(?:vision|visual|audio|embed_vision)).*"
            r"(?:language_model|text_model|model)\.layers\.\d+\."
            r"(?:self_attn|mlp)\.(?:q|k|v|o|gate|up|down)_proj$"
        ),
        "LR": float(os.environ.get("ELAINA_LR", "1e-4")),
        "EPOCHS": 1 if smoke else int(os.environ.get("ELAINA_EPOCHS", "2")),
        "BATCH": int(os.environ.get("ELAINA_BATCH", "2")),
        "GRAD_ACCUM": 2 if smoke else int(os.environ.get("ELAINA_GRAD_ACCUM", "8")),
        "SEED": 3407,
        "EVAL_STEPS": 5 if smoke else 100,
        "SAVE_STEPS": 5 if smoke else 100,
        "MAX_STEPS": 5 if smoke else -1,
        "MAX_TRAIN_ROWS": 100 if smoke else 0,
        "QUANTS": quants,
        "MAX_SHARD": "4GB",
        "OUT_DIR": workdir,
        "DATASET_DIR": os.environ.get("ELAINA_DATASET_DIR", "training/out"),
        "LLAMA_CPP_REVISION": os.environ.get("ELAINA_LLAMA_CPP_REV", "master"),
    }


def step_env_gate(cfg: dict) -> dict:
    print("\n=== [1/8] Environment Gate ===")
    cfg["OUT_DIR"].mkdir(parents=True, exist_ok=True)
    paths = [str(cfg["OUT_DIR"])]
    if os.environ.get("HF_HOME"):
        paths.append(os.environ["HF_HOME"])
    gate = lib.env_gate(paths)
    gpu = gate["gpu"]
    print(f"  tier: {gate['tier']} | free disk: {gate['free_gib']} GB | RAM: {gate['ram_gib']} GB")
    print(f"  GPU: {gpu.get('name')} | VRAM: {gate['vram_gib']} GB | bf16: {gpu.get('bf16')}")
    print(f"  torch/CUDA: {gpu.get('torch')} / {gpu.get('cuda')}")
    for w in gate["warnings"]:
        print(f"  [WARN] {w}")
    if gpu.get("name") is None:
        print("  [WARN] No CUDA GPU detected in this shell! Check `nvidia-smi` or ensure Molab GPU runtime is active.")
        if not cfg["smoke"]:
            raise SystemExit("Aborting full run: no CUDA GPU detected. Run `nvidia-smi` in terminal to check.")
    if gate["blocked"]:
        raise SystemExit(f"Environment gate blocked:\n{gate['message']}")
    return gate


def step_hf_login_and_repos(cfg: dict) -> tuple:
    print("\n=== [2/8] HuggingFace Auth & Repos ===")
    from huggingface_hub import HfApi, login

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token.strip(), add_to_git_credential=False)
    hf_api = HfApi()
    try:
        user = hf_api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(
            f"Not logged in to HuggingFace ({type(exc).__name__}: {exc}). "
            "Set HF_TOKEN in .env or run `huggingface-cli login`."
        )

    suffix = "-smoke" if cfg["smoke"] else ""
    base_name = "elaina-gemma4-e2b" if cfg["smoke"] else "elaina-gemma4-31b"
    repos = {
        "dataset": os.environ.get("ELAINA_DATASET_REPO", f"{user}/elaina-gemma4-dataset"),
        "adapter": os.environ.get("ELAINA_ADAPTER_REPO", f"{user}/{base_name}{suffix}-lora"),
        "merged": os.environ.get("ELAINA_MERGED_REPO", f"{user}/{base_name}{suffix}"),
        "gguf": os.environ.get("ELAINA_GGUF_REPO", f"{user}/{base_name}{suffix}-GGUF"),
    }
    print(f"  user: {user} | mode: {cfg['mode']}")
    for k, v in repos.items():
        print(f"  {k:<8}: {v}")
    return hf_api, user, repos


def step_load_dataset(cfg: dict, repos: dict) -> tuple[list, list, dict]:
    print("\n=== [3/8] Dataset & Leakage Check ===")
    local_dir = Path(cfg["DATASET_DIR"])
    if (local_dir / "train.jsonl").exists() and (local_dir / "val.jsonl").exists():
        src = local_dir
        print(f"  using local dataset at {src}")
    else:
        from huggingface_hub import snapshot_download

        dl_dir = cfg["OUT_DIR"] / "dataset"
        print(f"  downloading dataset from {repos['dataset']} -> {dl_dir}")
        src = Path(snapshot_download(repo_id=repos["dataset"], repo_type="dataset", local_dir=str(dl_dir)))

    train_raw, val_raw = [], []
    for fname, sink in (("train.jsonl", train_raw), ("val.jsonl", val_raw)):
        p = src / fname
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sink.append(json.loads(line))

    manifest_sha = None
    m_path = src / "manifest.json"
    if m_path.exists():
        import hashlib

        manifest_sha = hashlib.sha256(m_path.read_bytes()).hexdigest()
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
        chapters = manifest.get("chapters", {})
        tr_ch = {c for chs in chapters.get("train_chapters", {}).values() for c in chs if not str(c).startswith(("synthetic_", "unknown_"))}
        va_ch = {c for chs in chapters.get("val_chapters", {}).values() for c in chs if not str(c).startswith(("synthetic_", "unknown_"))}
        overlap = tr_ch & va_ch
        if overlap:
            raise SystemExit(f"FATAL: cross-lingual chapter leak detected: {sorted(overlap)[:5]}")
        print(f"  chapter split verified (train={len(tr_ch)}, val={len(va_ch)}, overlap=0)")
    elif not cfg["smoke"]:
        raise SystemExit("manifest.json missing; refusing full run with unverified chapter split.")

    if cfg["MAX_TRAIN_ROWS"] and len(train_raw) > cfg["MAX_TRAIN_ROWS"]:
        train_raw = train_raw[: cfg["MAX_TRAIN_ROWS"]]
        print(f"  smoke mode: truncated train to {len(train_raw)} rows")

    for i, row in enumerate(train_raw + val_raw):
        msgs = row.get("messages", [])
        roles = [m.get("role") for m in msgs]
        expected = ["system"] + ["user", "assistant"] * ((len(msgs) - 1) // 2)
        if len(msgs) < 3 or (len(msgs) - 1) % 2 != 0 or roles != expected:
            raise SystemExit(f"Invalid message structure at row {i}: {roles}")

    print(f"  loaded train={len(train_raw)} | val={len(val_raw)}")
    return train_raw, val_raw, {"manifest_sha256": manifest_sha}


def step_tokenizer_and_proof(cfg: dict, train_raw: list) -> tuple:
    print("\n=== [4/8] Tokenizer & Chat-Template Proof ===")
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_ID"])
    if getattr(tokenizer, "chat_template", None) is None:
        proc = AutoProcessor.from_pretrained(cfg["MODEL_ID"])
        if getattr(proc, "chat_template", None):
            tokenizer.chat_template = proc.chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    parts = lib.derive_turn_parts(tokenizer.apply_chat_template)
    evidence = lib.masking_report(tokenizer, train_raw[0], max_len=cfg["MAX_SEQ"], response_part=parts["response_part"])
    print(f"  response_part: {parts['response_part']!r} | eot_part: {parts['eot_part']!r}")
    print(f"  masking ok: {evidence['ok']} | supervised={evidence['supervised_tokens']}/{len(evidence['input_ids'])}")
    if not evidence["ok"]:
        raise SystemExit(f"Masking proof failed: {evidence}")
    return tokenizer, parts


def step_load_and_lora(cfg: dict):
    print(f"\n=== [5/8] Load Base Model (bf16, cuda:0) & Attach LoRA ===")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    t0 = time.time()
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    base_model = None
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            cfg["MODEL_ID"], torch_dtype=torch.bfloat16, device_map=device_map
        )
        loader = "AutoModelForImageTextToText"
    except Exception as exc:
        print(f"  [WARN] AutoModelForImageTextToText failed ({exc}); falling back to AutoModelForCausalLM")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["MODEL_ID"], torch_dtype=torch.bfloat16, device_map=device_map
        )
        loader = "AutoModelForCausalLM"

    p0 = next(base_model.parameters())
    if p0.dtype != torch.bfloat16:
        raise SystemExit(f"Expected bfloat16 base model, got {p0.dtype}")
    meta_params = [n for n, p in base_model.named_parameters() if p.device.type == "meta"]
    if meta_params:
        raise SystemExit(f"Model has {len(meta_params)} parameters on 'meta' device: {meta_params[:3]}")
    print(f"  loaded {cfg['MODEL_ID']} via {loader} in {time.time() - t0:.1f}s")

    # Freeze all vision/audio/projector parameters
    frozen = 0
    for n, p in base_model.named_parameters():
        if any(k in n.lower() for k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_")):
            p.requires_grad = False
            frozen += p.numel()

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    model = get_peft_model(
        base_model,
        LoraConfig(
            r=cfg["LORA_R"],
            lora_alpha=cfg["LORA_ALPHA"],
            lora_dropout=cfg["LORA_DROPOUT"],
            target_modules=cfg["TARGET_MODULES"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_")):
            p.requires_grad = False

    visual_lora = [
        n for n, p in model.named_parameters()
        if p.requires_grad and any(k in n.lower() for k in ("vision", "visual", "img", "image", "projector", "embed_vision", "audio", "mm_"))
    ]
    if visual_lora:
        raise SystemExit(f"FATAL: visual parameters remain trainable: {visual_lora[:5]}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  frozen visual params: {frozen:,} | trainable LoRA params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def step_tokenize_and_train(cfg: dict, model, tokenizer, parts: dict, train_raw: list, val_raw: list, repos: dict, data_info: dict):
    print("\n=== [6/8] Tokenize & Train ===")
    import peft
    import torch
    import transformers
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    def encode(rows, split_name):
        feats, failures, truncated = [], [], 0
        for i, row in enumerate(rows):
            try:
                t = lib.tokenize_with_mask(tokenizer, row["messages"], cfg["MAX_SEQ"], parts["response_part"])
            except Exception as exc:
                failures.append(f"{split_name}[{i}]: {type(exc).__name__}: {exc}")
                continue
            if t["truncated"]:
                truncated += 1
            feat = {"input_ids": t["input_ids"], "labels": t["labels"]}
            if "attention_mask" in t:
                feat["attention_mask"] = t["attention_mask"]
            if "token_type_ids" in t:
                feat["token_type_ids"] = t["token_type_ids"]
            feats.append(feat)
        if failures:
            raise SystemExit(f"Tokenization failures in {split_name}:\n" + "\n".join(failures[:10]))
        print(f"  {split_name}: {len(feats)} rows encoded (truncated={truncated})")
        return Dataset.from_list(feats)

    train_ds = encode(train_raw, "train")
    val_ds = encode(val_raw, "val")

    eff_batch = max(cfg["BATCH"] * cfg["GRAD_ACCUM"], 1)
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps = cfg["MAX_STEPS"] if cfg["MAX_STEPS"] > 0 else steps_per_epoch * cfg["EPOCHS"]
    warmup_steps = max(1, int(0.03 * total_steps))

    args_kw = dict(
        output_dir=str(cfg["OUT_DIR"] / "lora"),
        per_device_train_batch_size=cfg["BATCH"],
        gradient_accumulation_steps=cfg["GRAD_ACCUM"],
        num_train_epochs=cfg["EPOCHS"],
        max_steps=cfg["MAX_STEPS"],
        learning_rate=cfg["LR"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=1 if cfg["MAX_STEPS"] > 0 else 10,
        eval_strategy="steps",
        eval_steps=cfg["EVAL_STEPS"],
        save_strategy="steps",
        save_steps=cfg["SAVE_STEPS"],
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        push_to_hub=True,
        hub_model_id=repos["adapter"],
        hub_private_repo=True,
        hub_strategy="all_checkpoints",
        seed=cfg["SEED"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kw),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lib.ResponseOnlyCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    resume = None
    try:
        rdir = cfg["OUT_DIR"] / "resume"
        snapshot_download(repo_id=repos["adapter"], repo_type="model", allow_patterns=["checkpoint-*/**", "last-checkpoint/**"], local_dir=str(rdir))
        cks = sorted(
            (p for p in rdir.rglob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit() and (p / "trainer_state.json").exists()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if cks:
            resume = str(cks[-1])
        elif (rdir / "last-checkpoint" / "trainer_state.json").exists():
            resume = str(rdir / "last-checkpoint")
    except Exception:
        pass

    print(f"  starting training (resume={resume or 'scratch'}, total_steps={total_steps})")
    t0 = time.time()
    res = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = trainer.evaluate()
    print(f"  done in {(time.time() - t0)/60:.1f}m | step={res.global_step} | train_loss={res.training_loss:.4f} | eval_loss={eval_metrics.get('eval_loss')}")

    lib.write_run_manifest(
        cfg["OUT_DIR"] / "training_run_manifest.json",
        base_model=cfg["MODEL_ID"],
        adapter_repo=repos["adapter"],
        dataset_repo=repos["dataset"],
        dataset_sha256=data_info.get("manifest_sha256"),
        hyperparameters={"lora_r": cfg["LORA_R"], "lora_alpha": cfg["LORA_ALPHA"], "lr": cfg["LR"], "max_seq": cfg["MAX_SEQ"], "epochs": cfg["EPOCHS"]},
        versions={"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        extra={"global_step": res.global_step, "train_loss": round(float(res.training_loss), 4), "eval": eval_metrics},
    )
    return trainer


def step_export_and_gguf(cfg: dict, hf_api, model, tokenizer, repos: dict, skip_merge: bool, skip_gguf: bool):
    print("\n=== [7/8] Push Adapter & Merge bf16 ===")
    hf_api.create_repo(repos["adapter"], repo_type="model", private=True, exist_ok=True)
    adapter_dir = cfg["OUT_DIR"] / "adapter_final"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    manifest_p = cfg["OUT_DIR"] / "training_run_manifest.json"
    if manifest_p.exists():
        shutil.copy2(manifest_p, adapter_dir / "training_run_manifest.json")
    hf_api.upload_folder(folder_path=str(adapter_dir), repo_id=repos["adapter"], repo_type="model")
    print(f"  adapter pushed -> {repos['adapter']}")

    if skip_merge:
        print("  --skip-merge set; stopping before merge/GGUF.")
        return

    hf_api.create_repo(repos["merged"], repo_type="model", private=True, exist_ok=True)
    merged_dir = cfg["OUT_DIR"] / "merged_tmp"
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size=cfg["MAX_SHARD"])
    tokenizer.save_pretrained(str(merged_dir))
    for f in sorted(merged_dir.iterdir()):
        if not f.is_file():
            continue
        sz = f.stat().st_size
        hf_api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repos["merged"])
        if not lib.hub_file_ok(hf_api, repos["merged"], f.name, min_bytes=sz):
            raise RuntimeError(f"{f.name} failed Hub verification; kept locally")
        f.unlink()
    shutil.rmtree(merged_dir, ignore_errors=True)
    print(f"  merged bf16 pushed -> {repos['merged']}")

    if skip_gguf:
        print("  --skip-gguf set; skipping GGUF conversion.")
        return

    print("\n=== [8/8] GGUF + mmproj Conversion ===")
    from huggingface_hub import snapshot_download

    dl_merged = cfg["OUT_DIR"] / "merged_dl"
    gguf_out = cfg["OUT_DIR"] / "gguf"
    gguf_out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repos["merged"], local_dir=str(dl_merged))

    gguf_script = _THIS_DIR / "gguf_convert.py"
    cmd = [
        sys.executable, str(gguf_script),
        "--merged", str(dl_merged),
        "--out", str(gguf_out),
        "--quants", ",".join(cfg["QUANTS"]),
        "--work", str(cfg["OUT_DIR"] / "llama.cpp"),
        "--repo", repos["gguf"],
        "--mmproj",
    ]
    if cfg["LLAMA_CPP_REVISION"] != "master":
        cmd.extend(["--revision", cfg["LLAMA_CPP_REVISION"]])
    rc = lib.run_logged(cmd, logger=print)
    shutil.rmtree(dl_merged, ignore_errors=True)
    if rc != 0:
        raise SystemExit(f"gguf_convert.py failed with exit code {rc}")
    print(f"  GGUF (+mmproj) pushed -> {repos['gguf']}")


def main() -> int:
    load_dotenv_file(Path(".env"))
    load_dotenv_file(_THIS_DIR / ".env")

    p = argparse.ArgumentParser(description="Terminal-first Elaina Gemma 4 bf16 LoRA training & export.")
    p.add_argument("--mode", choices=["smoke", "full"], help="override ELAINA_MODE")
    p.add_argument("--workdir", help="override ELAINA_WORKDIR (default: ./work)")
    p.add_argument("--quants", help="comma-separated GGUF quants (e.g. Q4_K_M,Q6_K)")
    p.add_argument("--skip-merge", action="store_true", help="push adapter only, skip merged bf16 & GGUF")
    p.add_argument("--skip-gguf", action="store_true", help="skip GGUF conversion after merge")
    args = p.parse_args()

    cfg = build_cfg(args)
    step_env_gate(cfg)
    hf_api, _user, repos = step_hf_login_and_repos(cfg)
    train_raw, val_raw, data_info = step_load_dataset(cfg, repos)
    tokenizer, parts = step_tokenizer_and_proof(cfg, train_raw)
    model = step_load_and_lora(cfg)
    step_tokenize_and_train(cfg, model, tokenizer, parts, train_raw, val_raw, repos, data_info)
    step_export_and_gguf(cfg, hf_api, model, tokenizer, repos, args.skip_merge, args.skip_gguf)
    print("\n[ALL DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
