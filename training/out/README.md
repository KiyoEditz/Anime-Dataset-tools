---
license: other
license_name: proprietary-educational-research-only
language:
  - en
  - id
  - jp
tags:
  - persona
  - roleplay
  - sft
  - elaina
  - not-for-commercial-use
---

# Elaina persona SFT dataset

SFT data for fine-tuning Gemma on the *Wandering Witch: The Journey of Elaina*
persona. Generated from 44 EPUB volumes with an LLM extraction pipeline.

## Contents

| File | Rows | Notes |
|---|---|---|
| `train.jsonl` | 23988 | `{"messages":[system,user,assistant]}` |
| `val.jsonl` | 1825 | held out **by chapter**, no leakage |
| `train_meta.jsonl` / `val_meta.jsonl` | 23988 / 1825 | sidecar: lang, chapter, kind, prompt |
| `manifest.json` | -- | counts, split chapters, params, sha256 |
| `stats.json` | -- | length distributions |

## Prompt

Most rows carry a compact (~581 char) system prompt; 10% of **train** rows
carry the full ~0 char lore prompt so the model does not key on one exact string.
Val always uses the short prompt.

At inference, the recommended system message is:

```
<short prompt>

<full lore prompt>
```

## Mix

| Lang | Rows |
|---|---|
| `en` | 8342 |
| `id` | 8179 |
| `jp` | 7467 |

User turns are either verbatim dialogue or third-person scene narration prefixed with
`[Scene] `; the ratio is roughly 85% / 5%.

## Split method

Whole chapters are assigned to train or val, so no chapter straddles the split.
`manifest.json -> chapters` records exactly which chapters went where.

Built by `training/prepare_dataset.py`.
