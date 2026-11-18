#!/usr/bin/env python3
"""
push_dataset_to_hub.py

Upload the prepared dataset (training/out/) to a private HuggingFace dataset
repo and verify it by downloading it back and comparing sha256 hashes.

Run this on your own machine -- the Molab notebook pulls from the Hub, so it
never needs the original output/ files.

Usage:
    huggingface-cli login              # once, or export HF_TOKEN=hf_...
    python training/push_dataset_to_hub.py --dir training/out --repo YOU/elaina-gemma4-dataset

    # dry run: show what would be uploaded
    python training/push_dataset_to_hub.py --dir training/out --repo YOU/x --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REQUIRED = ("train.jsonl", "val.jsonl", "manifest.json", "stats.json")
OPTIONAL = ("train_meta.jsonl", "val_meta.jsonl")

CARD_TEMPLATE = """---
license: other
license_name: proprietary-educational-research-only
language:
{langs}
tags:
  - persona
  - roleplay
  - sft
  - elaina
  - not-for-commercial-use
---

# Elaina persona SFT dataset

SFT data for fine-tuning Gemma on the *Wandering Witch: The Journey of Elaina*
persona. Generated from {volumes} EPUB volumes with an LLM extraction pipeline.

## Contents

| File | Rows | Notes |
|---|---|---|
| `train.jsonl` | {train_rows} | `{{"messages":[system,user,assistant]}}` |
| `val.jsonl` | {val_rows} | held out **by chapter**, no leakage |
| `train_meta.jsonl` / `val_meta.jsonl` | {train_rows} / {val_rows} | sidecar: lang, chapter, kind, prompt |
| `manifest.json` | -- | counts, split chapters, params, sha256 |
| `stats.json` | -- | length distributions |

## Prompt

Most rows carry a compact (~{short_chars} char) system prompt; {long_frac:.0%} of **train** rows
carry the full ~{long_chars} char lore prompt so the model does not key on one exact string.
Val always uses the short prompt.

At inference, the recommended system message is:

```
<short prompt>

<full lore prompt>
```

## Mix

{lang_table}

User turns are either verbatim dialogue or third-person scene narration prefixed with
`[Scene] `; the ratio is roughly {dialogue_share:.0%} / {scene_share:.0%}.

## Split method

Whole chapters are assigned to train or val, so no chapter straddles the split.
`manifest.json -> chapters` records exactly which chapters went where.

Built by `training/prepare_dataset.py`.
"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def build_card(out_dir: Path) -> str:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    by_lang = counts.get("by_lang", {})
    prompts = manifest.get("prompts", {})

    lang_rows = sum(by_lang.values()) or 1
    langs = "\n".join(f"  - {code}" for code in sorted(by_lang)) or "  - en"
    lang_table = "| Lang | Rows |\n|---|---|\n" + "\n".join(
        f"| `{code}` | {n} |" for code, n in sorted(by_lang.items())
    )

    train_rows = counts.get("train", 0)
    val_rows = counts.get("val", 0)
    stats_path = out_dir / "stats.json"
    dialogue_share, scene_share = 0.5, 0.5
    if stats_path.exists():
        kinds = json.loads(stats_path.read_text(encoding="utf-8")).get("train", {}).get("by_kind", {})
        total = sum(kinds.values()) or 1
        dialogue_share = (kinds.get("dialogue", 0) + kinds.get("synthetic", 0)) / total
        scene_share = kinds.get("scene", 0) / total

    # Estimate number of volumes dynamically from manifest chapters or metadata (avoid hardcoded 12)
    chapters_dict = manifest.get("chapters", {})
    all_chapters = set()
    for split_key in ("train_chapters", "val_chapters"):
        for ch_list in chapters_dict.get(split_key, {}).values():
            all_chapters.update(ch for ch in ch_list if not str(ch).startswith(("synthetic_", "unknown_")))
    # Each novel volume has ~16-17 chapters on average (~199 chapters across 12 volumes), or explicit volume prefix
    vol_prefixes = {str(ch).split("_")[0] for ch in all_chapters if str(ch).lower().startswith("vol")}
    if manifest.get("volumes"):
        volumes = manifest["volumes"]
    elif vol_prefixes:
        volumes = len(vol_prefixes)
    elif all_chapters:
        volumes = max(1, round(len(all_chapters) / 16.5))
    else:
        volumes = "multiple"

    return CARD_TEMPLATE.format(
        langs=langs,
        volumes=volumes,
        train_rows=train_rows,
        val_rows=val_rows,
        short_chars=prompts.get("short_chars", 0),
        long_chars=prompts.get("long_chars", 0),
        long_frac=manifest.get("params", {}).get("long_prompt_frac", 0.0),
        lang_table=lang_table,
        dialogue_share=dialogue_share,
        scene_share=scene_share,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Upload training/out to a HF dataset repo and verify it.")
    p.add_argument("--dir", default="training/out", help="prepared dataset directory")
    p.add_argument("--repo", required=True, help="dataset repo id, e.g. you/elaina-gemma4-dataset")
    p.add_argument("--private", dest="private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    p.add_argument("--verify", dest="verify", action="store_true", default=True)
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.dir)
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} not found. Run training/prepare_dataset.py first.")

    missing = [name for name in REQUIRED if not (out_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing required file(s) in {out_dir}: {', '.join(missing)}")

    files = [out_dir / n for n in REQUIRED + OPTIONAL if (out_dir / n).exists()]
    local_hashes = {f.name: sha256_of(f) for f in files}

    print(f"[LOCAL] {out_dir}")
    for name, digest in local_hashes.items():
        size = (out_dir / name).stat().st_size
        print(f"  {name:<20} {size:>10,} B  {digest[:16]}...")

    card_path = out_dir / "README.md"
    card_text = build_card(out_dir)
    print(f"\n[CARD] {len(card_text)} chars -> {card_path}")

    if args.dry_run:
        print("[DRY RUN] nothing uploaded")
        return 0

    try:
        from huggingface_hub import HfApi, hf_hub_download, login
    except ImportError:
        raise SystemExit("huggingface_hub is not installed: pip install huggingface_hub hf_transfer")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi()

    try:
        user = api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(f"not logged in to HuggingFace ({type(exc).__name__}: {exc}).\n"
                         f"Run `huggingface-cli login` or set HF_TOKEN.")

    if "/" not in args.repo:
        args.repo = f"{user}/{args.repo}"
    print(f"[AUTH] {user} -> {args.repo} (private={args.private})")

    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    card_path.write_text(card_text, encoding="utf-8")

    for f in files + [card_path]:
        print(f"[UPLOAD] {f.name} ...", end="", flush=True)
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(" ok")

    if not args.verify:
        print(f"\n[DONE] https://huggingface.co/datasets/{args.repo}")
        return 0

    print("\n[VERIFY] downloading back and comparing hashes")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, digest in local_hashes.items():
            try:
                got = hf_hub_download(
                    repo_id=args.repo, repo_type="dataset", filename=name,
                    local_dir=tmp, force_download=True,
                )
            except Exception as exc:
                failures.append(f"{name}: download failed ({type(exc).__name__}: {exc})")
                continue
            remote = sha256_of(Path(got))
            if remote != digest:
                failures.append(f"{name}: hash mismatch (local {digest[:12]} vs remote {remote[:12]})")
            else:
                print(f"  {name:<20} OK")

    if failures:
        print("\n[FAIL] verification failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\n[OK] https://huggingface.co/datasets/{args.repo}")
    print(f"     train {count_lines(out_dir / 'train.jsonl')} rows | "
          f"val {count_lines(out_dir / 'val.jsonl')} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())