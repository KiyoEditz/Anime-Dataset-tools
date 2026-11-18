# Training Elaina on Gemma 4 31B (LoRA bf16 / Blackwell 96 GB)

> 📚 **Dokumentasi Lengkap & Peta Alur:** Lihat [**`training/docs/`**](docs/README.md) untuk panduan operasional, peta alur mermaid, dan cara penggunaan.  
> 🛠️ **Rencana Perbaikan Kode & Audit:** Lihat [**`training/docs/fix/`**](docs/fix/README.md) untuk daftar temuan audit lengkap (C1–C9, H1–H10, QA1–QA11) dan usulan patch.

Runbook for turning `output/raw_dialogue_pairs.jsonl` into a LoRA adapter, a
merged bf16 model, and GGUF files, using Molab (marimo) on a 96 GB Blackwell GPU.

Nothing in this folder modifies the extraction pipeline. `prepare_dataset.py`
is a *variant* of `build_dataset.py`, not a replacement, and the only step that
needs the original `output/` files.

```
training/
  docs/                           comprehensive guides & mermaid workflow maps
    fix/                          code fix plans & audit findings (C1-C9, H1-H10, QA1-QA11)
  prepare_dataset.py              local: raw pairs -> train/val + manifest + stats
  push_dataset_to_hub.py          local: create the private dataset repo, verify hashes
  molab_train_elaina.py           marimo notebook: gate -> train -> merge -> GGUF
  (no separate smoke file: ELAINA_MODE=smoke does it)
  elaina_train_lib.py             shared code (env gate, masking, Hub helpers)
  gguf_convert.py                 standalone llama.cpp convert + quantise runner
  elaina_system_prompt_short.txt  the prompt used in training (~470 chars)
  sources.json                    language slots: en (ready), id/jp (pending)
  requirements-training.txt       reference environment, not auto-installed
  out/                            gitignored: train.jsonl, val.jsonl, manifest.json, stats.json
```

---

## 0. Why the dataset is rebuilt, not reused

`output/elaina_finetune_dataset.jsonl` has 3,879 rows, and ~97% of its bytes are
the same 8.8 KB system prompt repeated. It also carries no `chapter` field, so a
leakage-free train/val split is impossible from that file alone.

`prepare_dataset.py` fixes both:

* reads `output/raw_dialogue_pairs.jsonl` directly, so `chapter` survives;
* splits **by chapter** (train and val chapter sets are asserted disjoint);
* swaps the long prompt for `elaina_system_prompt_short.txt`, keeping the long
  one on 10% of train rows as augmentation;
* tags narration user-turns with `[Scene] ` so dialogue and scene framing are
  distinguishable at inference time.

---

## 1. Local preparation (no GPU, ~10 seconds)

```bash
# preview only: builds everything in memory, writes nothing
python training/prepare_dataset.py --sources-file training/sources.json --check

# real run
python training/prepare_dataset.py --sources-file training/sources.json
```

Options that matter:

| Flag | Default | Meaning |
|---|---|---|
| `--val-frac` | 0.06 | fraction of rows held out, sampled **by chapter** |
| `--long-prompt-frac` | 0.10 | share of *train* rows that keep the 8.8 KB prompt |
| `--fuzzy-threshold` | 92 | `difflib` ratio above which two replies are duplicates |
| `--min/max-response-chars` | 8 / 1500 | length filter on the assistant turn |
| `--drop-scene` | off | remove `[Scene]`-tagged narration turns entirely |
| `--tokenizer MODEL_ID` | off | exact token counts (needs `transformers` locally) |

Expected result on today's corpus (verified):

```
train 3,528   val 252   chapter overlap: none
narration rows tagged [Scene]: 382
```

Check `training/out/stats.json` and `training/out/manifest.json` after the run.
`manifest.json` records the chapter lists, per-language counts, and the sha256
of every output file.

### Adding Indonesian / Japanese

`training/data_id/` and `training/data_jp/` are the drop points. Put
`raw_dialogue_pairs.jsonl` there, flip the slot in `sources.json` to
`"status": "ready"`, rerun. Missing files are skipped with a warning, never a
crash — so the EN-only run today and the 3-language run later are the same
command.

---

## 2. Push the dataset to the Hub

```bash
python training/push_dataset_to_hub.py \
  --dir training/out --repo <you>/elaina-gemma4-dataset --private
```

The script uploads, then re-downloads and compares sha256 of every file. It
refuses to report success if a hash differs.

---

## 3. Molab (marimo) run

Open `training/molab_train_elaina.py` in Molab. Cells run top to bottom; every
expensive cell is behind a run button, so you can stop after any of them and
resume later — the Hub holds everything that finished.

Order: `gate → login → repos → install → model → data → template proof →
load → LoRA → tokenise → trainer → train → eval → samples → adapter →
merge → GGUF → cleanup`.

Two cells are non-negotiable checkpoints:

* **Cell 4 (gate)** measures disk, RAM, GPU and prints the tier. If the volume
  is too small it prints why and blocks every later cell — nothing is
  downloaded first.
* **Cell 10 (chat-template proof)** derives the assistant-turn markers from the
  tokenizer's own template and shows the masked labels. If it cannot prove only
  the assistant span is supervised, do not train.

### Disk tiers

| Tier | Free disk | VRAM | Path |
|---|---|---|---|
| `full31b` | ≥ 150 GB | ≥ 80 GB | LoRA bf16 + streaming merge + all quants incl. Q8_0 + mmproj |
| `small31b` | ≥ 90 GB | ≥ 80 GB | LoRA bf16 + streaming merge + mmproj, **Q8_0 skipped** |
| `12b` | ≥ 45 GB | ≥ 24 GB | smoke validation tier (`ELAINA_MODE=smoke` with `google/gemma-4-E2B-it`) |
| `blocked` | below | — | run on HF Jobs instead (same dataset + adapter repos) |

If `shutil.disk_usage()` reports `8589934592`, that is **8 GiB**, not 8 billion
GB. The gate prints every number in GiB (`/2**30`) for exactly this reason.

### Memory / speed expectations

| Setting | Value |
|---|---|
| base | `KiyoEditz/Gemma-4-Dark-Gemistry-31B` (Roleplay Specialist VLM, `Gemma4ForConditionalGeneration`, exclusive target — no generic fallback) |
| precision | **bfloat16 LoRA** (`load_in_4bit=False`, ~62.5 GB base in VRAM on 96 GB Blackwell GPU) |
| LoRA | r=32, alpha=64, dropout 0.05, language decoder layers only (vision tower & projector frozen) |
| batch | 2 × grad-accum 8 = effective 16 |
| LR / epochs | 1e-4 cosine, 2 epochs, early stop patience 1, `adamw_torch_fused` |
| seq length | 4096 (512 in smoke mode) |

Trainable share lands around 0.3–0.9% of 31.3B parameters. If it is outside
0.05–3%, check `target_modules`. For detailed VRAM analysis and methodology, see [**`training/docs/04-metode-training.md`**](docs/04-metode-training.md).

---

## 4. Smoke test before renting GPU hours

```bash
ELAINA_MODE=smoke python -m marimo run training/molab_train_elaina.py
```

Smoke mode: `google/gemma-4-E2B-it` (same Gemma 4 VLM architecture and chat template), 100 train rows, 5 optimizer steps.
Its job is to prove every cell runs and every Hub push path works
end to end before the real run.

---

## 5. GGUF export (can run separately)

The notebook calls `gguf_convert.py`; you can also run it alone, e.g. on HF Jobs:

```bash
python training/gguf_convert.py \
  --merged /root/work/merged_dl --out /root/work/gguf \
  --quants Q4_K_M,Q6_K --work /root/work/llama.cpp \
  --repo <you>/elaina-gemma4-31b-GGUF --mmproj --revision <pinned-commit>
```

Defaults: llama.cpp built **CPU-only** (quantisation needs no GPU, and this
avoids sm_120 kernel issues), intermediate deleted after quantising, each quant
uploaded and verified before its local copy is removed. `Q1_*` does not exist in
llama.cpp; `Q2_K` is the smallest supported.

---

## 6. Inference

Training uses the **short** prompt. At inference, prepend it and append the lore:

```python
system = open("training/elaina_system_prompt_short.txt").read() \
       + "\n\n" + open("output/elaina_system_prompt.txt").read()
```

This keeps the trained voice while restoring lore detail. Pure short-prompt
inference also works and matches the training distribution most closely.

With llama.cpp:

```bash
llama-server -hf <you>/elaina-gemma4-31b-GGUF --hf-file elaina-Q4_K_M.gguf -c 4096
```

---

## 7. Troubleshooting

For the comprehensive diagnosis matrix, root-cause explanations, and quick fixes, see [**`training/docs/09-troubleshooting.md`**](docs/09-troubleshooting.md) and [**`training/docs/fix/`**](docs/fix/README.md).

| Symptom | Cause | Fix |
|---|---|---|
| gate says `blocked` | volume too small for 31B | use the 12B branch or HF Jobs (see [docs/fix/05 § M2](docs/fix/05-env-gate-dan-library.md#m2--gate-disk-ikut-menghitung-tmp)) |
| `RepositoryNotFoundError` on the model id | wrong id, or licence not accepted for *this* account | accept the licence on the model page; a write token is not enough |
| `KeyError: 'gemma4'` in `AutoConfig` | transformers too old | `pip install "git+https://github.com/huggingface/transformers.git@main"` |
| `no kernel image available` | torch built for the wrong arch | install the cu128 wheel, restart the session |
| trainable share 0% or >5% | wrong `target_modules` | check the LoRA cell's table and ensure vision tower is frozen (see [docs/fix/02 § C4](docs/fix/02-metode-lora-bf16-dan-vision.md#c4--target-lora-bisa-mengenai-vision-tower)) |
| masking proof fails | chat template changed | do not train; inspect the rendered example (see [docs/06-chat-template-gemma.md](docs/06-chat-template-gemma.md)) |
| eval loss flat while train loss drops | overfitting | expected on 3.5 K rows; judge by sample generations, not eval loss |
| `llama-quantize` fails on `Gemma4ForConditionalGeneration` | llama.cpp lags gemma4 | pin a newer commit; try `--outtype bf16`; run on HF Jobs |

---

## 8. What this training does and does not buy you

**Yes:** Elaina's voice — tics (`Shoot.`, `Ugh.`, ellipses, "Oh-hoh-hoh"),
sarcasm, teasing register, refusal phrasing — and consistent short persona
answers on topics present in the novels.

**No:** lore fidelity outside the training scenarios (use retrieval over
`text/chapter_*.txt` for that), multi-turn state, tool use, or general
assistant ability. 3,879 single-turn pairs at ~215 K conversational tokens is a
style prior, not a new capability. Val loss is optimistic because the base model
has likely seen the novels — judge by the sample generations, not by eval loss.