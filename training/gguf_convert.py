#!/usr/bin/env python3
"""
gguf_convert.py

Standalone llama.cpp runner: merged HF checkpoint -> GGUF (f16 or bf16) ->
quantised GGUF -> optionally uploaded to a HuggingFace repo.

Kept out of the marimo notebook so the most fragile step can be run on its own,
inside a container, or on HF Jobs without marimo:

    python training/gguf_convert.py \
        --merged /root/work/merged_dl \
        --out /root/work/gguf \
        --quants Q4_K_M,Q6_K \
        --work /root/work/llama.cpp \
        --repo you/elaina-gemma4-31b-GGUF

Design notes that matter:

  * CPU-only build by default (`-DGGML_CUDA=OFF`). Quantisation needs no GPU,
    and this sidesteps sm_120 kernel issues on Blackwell.
  * llama.cpp is built from a *pinned commit* or master, because gemma4 support
    landed recently and release tags lag behind it.
  * The f16/bf16 intermediate is deleted as soon as all quantisations that
    need it are produced, so peak disk is intermediate + largest quant.
  * Each quant is uploaded, verified on the Hub, and then deleted locally,
    so peak disk never holds all quantisations at once.
  * Multimodal: gemma4 is a VLM. `--mmproj` is attempted and its failure is a
    warning, not an error -- the text model is the deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"

# Quantisations llama.cpp accepts from an f16/bf16 GGUF. Q1_* does not exist;
# Q2_K is the smallest mainstream one.
VALID_QUANTS = {
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
}

# Rough output size relative to the f16 intermediate, for disk pre-flight.
QUANT_RATIO = {
    "Q2_K": 0.19, "Q3_K_S": 0.23, "Q3_K_M": 0.25, "Q3_K_L": 0.27,
    "Q4_0": 0.30, "Q4_1": 0.31, "Q4_K_S": 0.30, "Q4_K_M": 0.32,
    "Q5_0": 0.35, "Q5_1": 0.36, "Q5_K_S": 0.35, "Q5_K_M": 0.36,
    "Q6_K": 0.41, "Q8_0": 0.53,
}

GIB = 2 ** 30


def log(msg: str) -> None:
    print(f"[gguf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None, check=True, env=None) -> int:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(map(str, cmd))}")
    return rc


def free_gib(path: Path) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(str(target)).free / GIB


def ensure_llama(work: Path, revision: str | None, cpu_only: bool) -> tuple[Path, Path]:
    """Clone + build llama.cpp. Returns (build_dir, convert_script)."""
    work.mkdir(parents=True, exist_ok=True)

    if not (work / ".git").exists():
        log(f"cloning llama.cpp into {work}")
        run(["git", "clone", "--depth", "1", LLAMA_REPO, str(work)])
    else:
        log("reusing existing llama.cpp checkout")

    if revision:
        log(f"checking out pinned revision {revision}")
        run(["git", "fetch", "--depth", "1", "origin", revision], cwd=work)
        run(["git", "checkout", revision], cwd=work)

    build = work / "build"
    flags = ["-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"]
    if cpu_only:
        flags.append("-DGGML_CUDA=OFF")

    if not (build / "CMakeCache.txt").exists():
        run(["cmake", "-B", str(build), "-S", str(work), *flags], cwd=work)

    run(["cmake", "--build", str(build), "--config", "Release", "-j",
         str(os.cpu_count() or 4), "--target", "llama-quantize"], cwd=work)

    quantize = build / "bin" / "llama-quantize"
    if not quantize.exists():
        quantize = build / "llama-quantize"
    if not quantize.exists():
        raise RuntimeError(f"llama-quantize not found under {build}")

    convert = work / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise RuntimeError(f"convert_hf_to_gguf.py not found in {work}")

    gguf_py = work / "gguf-py"
    if gguf_py.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(gguf_py)], check=False)

    return build, convert


def _convert_env(convert: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra = [str(convert.parent), str(convert.parent / "gguf-py")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def convert_intermediate(convert: Path, merged: Path, out: Path, outtype: str) -> Path:
    """HF merged dir -> one GGUF at `outtype` (f16 / bf16 / q8_0)."""
    cmd = [sys.executable, str(convert), str(merged),
           "--outfile", str(out), "--outtype", outtype]
    log(f"converting to {outtype} -> {out.name}")
    try:
        run(cmd, cwd=convert.parent, env=_convert_env(convert))
    except RuntimeError as exc:
        if outtype == "bf16":
            raise
        log(f"{outtype} outtype failed ({exc}); retrying with bf16")
        return convert_intermediate(convert, merged, out, "bf16")
    if not out.exists():
        raise RuntimeError(f"converter reported success but {out} is missing")
    log(f"{out.name}: {out.stat().st_size / GIB:.2f} GB")
    return out


def try_mmproj(convert: Path, merged: Path, out_dir: Path) -> Path | None:
    """Best-effort vision projector export; never fatal."""
    target = out_dir / "mmproj-F16.gguf"
    try:
        run([sys.executable, str(convert), str(merged),
             "--mmproj", "--outfile", str(target), "--outtype", "f16"],
            cwd=convert.parent, env=_convert_env(convert), check=True)
    except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
        log(f"mmproj export skipped: {type(exc).__name__}: {exc}")
        return None
    if target.exists():
        log(f"{target.name}: {target.stat().st_size / GIB:.2f} GB")
        return target
    return None


def quantize(quantize_bin: Path, src: Path, dst: Path, quant: str) -> Path:
    run([str(quantize_bin), str(src), str(dst), quant])
    if not dst.exists():
        raise RuntimeError(f"quantisation {quant} produced no file at {dst}")
    log(f"{dst.name}: {dst.stat().st_size / GIB:.2f} GB")
    return dst


def upload_and_delete(api, repo_id: str, path: Path, repo_type: str = "model") -> bool:
    size = path.stat().st_size
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id)
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=True)
    ok = any(s.rfilename == path.name and (getattr(s, "size", None) or size) >= size
             for s in (info.siblings or []))
    if ok:
        path.unlink()
        log(f"uploaded and deleted {path.name}")
    else:
        log(f"WARNING: {path.name} not confirmed on the Hub; keeping it locally")
    return ok


def preflight(args, quants: list[str], merged_gib: float) -> None:
    # Both f16 and bf16 are 2 bytes per parameter; only q8_0 is ~1 byte per parameter.
    intermediate = merged_gib * (0.5 if args.outtype == "q8_0" else 1.0)
    largest = max(QUANT_RATIO.get(q, 0.5) for q in quants) * merged_gib
    mmproj_gib = 1.5 if getattr(args, "mmproj", False) else 0.0
    need = intermediate + largest + mmproj_gib + 2.0
    have = free_gib(Path(args.out))
    log(f"disk pre-flight: free {have:.1f} GB, need ~{need:.1f} GB "
        f"(intermediate {intermediate:.1f} + largest quant {largest:.1f})")
    if have < need and not args.force:
        raise RuntimeError(
            f"not enough disk: {have:.1f} GB free, ~{need:.1f} GB needed.\n"
            "Free space, drop Q8_0 from --quants, or pass --force to try anyway."
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", required=True, help="local dir with the merged HF model")
    p.add_argument("--out", required=True, help="output dir for .gguf files")
    p.add_argument("--quants", default="Q4_K_M", help="comma-separated quant list")
    p.add_argument("--work", default="llama.cpp", help="llama.cpp checkout dir")
    p.add_argument("--repo", default=None, help="HF model repo to push into (optional)")
    p.add_argument("--revision", default=None, help="pin llama.cpp commit (recommended)")
    p.add_argument("--outtype", default="f16", choices=["f16", "bf16", "q8_0"],
                   help="type of the intermediate the quants are made from")
    p.add_argument("--gpu", action="store_true", help="build llama.cpp with CUDA (off by default)")
    p.add_argument("--mmproj", action="store_true", help="also export the vision projector")
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--private", action="store_true", help="create HF repo as private (default: public)")
    p.add_argument("--force", action="store_true", help="skip the disk pre-flight abort")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args(argv)

    merged = Path(args.merged).resolve()
    out = Path(args.out).resolve()
    work = Path(args.work).resolve()

    if not merged.exists():
        raise SystemExit(f"--merged does not exist: {merged}")

    quants = [q.strip().upper() for q in args.quants.split(",") if q.strip()]
    bad = [q for q in quants if q not in VALID_QUANTS]
    if bad:
        raise SystemExit(f"unknown quant(s) {bad}; valid: {sorted(VALID_QUANTS)}")

    merged_gib = sum(f.stat().st_size for f in merged.rglob("*") if f.is_file()) / GIB
    log(f"merged model: {merged} ({merged_gib:.1f} GB)")
    log(f"quants: {quants}")

    if args.dry_run:
        preflight(args, quants, merged_gib)
        log("dry-run: nothing built, nothing converted")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    preflight(args, quants, merged_gib)

    build, convert = ensure_llama(work, args.revision, cpu_only=not args.gpu)
    quantize_bin = build / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = build / "llama-quantize"

    api = None
    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        log(f"pushing to {args.repo} (private={args.private})")

    intermediate = out / f"elaina-{args.outtype}.gguf"
    produced: dict[str, float] = {}

    try:
        convert_intermediate(convert, merged, intermediate, args.outtype)

        if args.mmproj:
            mmproj = try_mmproj(convert, merged, out)
            if mmproj and api:
                upload_and_delete(api, args.repo, mmproj)

        for q in quants:
            dst = out / f"elaina-{q}.gguf"
            quantize(quantize_bin, intermediate, dst, q)
            produced[dst.name] = round(dst.stat().st_size / GIB, 2)
            if api:
                upload_and_delete(api, args.repo, dst)
            else:
                log(f"keeping {dst.name} (no --repo given)")
    finally:
        if intermediate.exists() and not args.keep_intermediate:
            intermediate.unlink()
            log(f"deleted intermediate {intermediate.name}")

    summary = {"repo": args.repo, "quants": produced,
               "intermediate_deleted": not args.keep_intermediate}
    log("summary: " + json.dumps(summary))
    (out / "gguf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())