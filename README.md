# Elaina Dataset Tool

A pipeline that turns the novel *Majo no Tabitabi / The Journey of
Elaina* (your own EPUB/PDF copy) into a JSONL chatbot fine-tuning
dataset, capturing Elaina's personality, speech style, and memories.

This pipeline works with any OpenAI-compatible LLM provider. See
`.env.example` for 5 ready-to-use configurations. **Recommended**:
Claude Sonnet (via the Anthropic API + LiteLLM or OpenRouter) for the
best extraction quality.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Open .env, pick ONE provider block, uncomment its lines,
# and fill in the matching key and URL.
export $(grep -v '^#' .env | xargs)   # load env vars (skip comment lines)
```

Test the connection to your provider before starting the pipeline:
```bash
curl $LLM_BASE_URL/models -H "Authorization: Bearer $LLM_API_KEY"
# Should return a list of models, not a 401/403/connection-refused error
```

## Setting up LiteLLM + Claude (Option 1 — Best)

If you already have LiteLLM running in Colab via a Cloudflare Tunnel and
want to add Claude, edit your LiteLLM `config.yaml`:

```yaml
model_list:
  - model_name: claude-sonnet          # any alias you like
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: hermes-4-14b           # your local model stays available too
    litellm_params:
      model: ollama/hermes-4-14b-abliterated
      api_base: http://localhost:11434
```

Then in Colab, set the env var **before** starting LiteLLM:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
litellm --config config.yaml
```

Anthropic API key: sign up at `console.anthropic.com` → *API Keys* menu.

## Folder structure

```
elaina-dataset-tool/
  input/          <- put your novel .epub / .pdf files here
  confirm_xhtml/  <- (step 1) unpacked XHTML in true reading order for verification
    Volume_01/    <- candidate story chapters
    _excluded_junk/ <- automatically isolated covers, TOC, copyright, ads, images
  text/           <- (step 2) clean plain-text chapters (chapter_0001.txt, ...)
  output/         <- (auto) all final & intermediate results
```

## How to use it

1. Put your novel's EPUB/PDF files in the `input/` folder. Name them so
   they sort in order (e.g. `01_vol1.epub`, `02_vol2.epub` or `Volume 01.epub`).

2. **Stage 1: Two-Step Extraction with Confirmation** (prevents junk from leaking into the dataset):

   - **Step 1A: Unpack & Triage to `confirm_xhtml/`**:
     ```bash
     python extract_text.py --step unpack
     ```
     This reads the EPUB spine in exact reading order, maps real TOC titles,
     places real story chapters into `confirm_xhtml/Volume_XX/`, and moves
     covers, table of contents, copyright, and image pages into `_excluded_junk/`.
     Inspect `confirm_xhtml/` and check `confirm_xhtml/SUMMARY.md`. Delete any
     file you don't want, or move back any file from `_excluded_junk/`.

   - **Step 1B: Convert Confirmed XHTML to Clean Text**:
     ```bash
     python extract_text.py --step text
     ```
     This converts the confirmed XHTML files into sequentially numbered text files:
     `text/chapter_0001.txt`, `text/chapter_0002.txt`, ..., with volume and title headers.
     Spot-check `text/` to confirm the contents and file names match your expectations.

3. **Stage 2 to 5: Run the LLM Extraction Pipeline**:
   The chapters will be processed strictly chronologically by chapter name:

   ```bash
   python persona_extractor.py     # 2. build the personality & memory profile
   python dialogue_extractor.py    # 3. extract dialogue pairs from the novel
   python augment_synthetic.py 10  # 4. (optional) add original synthetic Q&A
   python build_dataset.py         # 5. assemble the final JSONL
   python validate_dataset.py      # check the final output's quality & structure
   ```

   *(Or run `python run_all.py` once your text/ folder is confirmed!)*

4. Final output: `output/elaina_finetune_dataset.jsonl`, format:

   ```json
   {"messages": [
     {"role": "system", "content": "You are Elaina, ..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "Elaina's response..."}
   ]}
   ```

   This format is directly compatible with most chat fine-tuning
   frameworks (Axolotl, LLaMA-Factory, the OpenAI fine-tuning format, etc.).


## Why is this staged?

- **persona_extractor.py** runs first so there's a single "character
  bible" (`output/elaina_profile.json` + `output/elaina_system_prompt.txt`)
  that's used consistently as the system prompt across every example.
- **dialogue_extractor.py** pulls genuine scenes from the novel -- this
  is what makes Elaina's responses feel authentic to the source.
- **augment_synthetic.py** is deliberately separate because its content
  is NOT quoted from the novel -- it's new Q&A the model generates based
  on the profile. This helps the Elaina bot answer questions that aren't
  exactly in the novel, and reduces a dataset that's just verbatim
  copies of the original text.

## Choosing a provider & model

This tool talks to models through a single OpenAI-compatible wrapper
(`llm_client.py`), so the provider can be swapped without changing any
code -- just change 3 variables in `.env`: `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL`. See `.env.example` for ready-made options:

- **Your own LiteLLM** (Ollama + Hermes-4-14B abliterated via Cloudflare
  Tunnel) -- private, but limited context window (~40k tokens) and speed
  depends on your Colab GPU.
- **NVIDIA Build / NIM** (https://build.nvidia.com) -- free, no credit
  card, OpenAI-compatible endpoint, many open-weight models with a much
  larger context window (typically 128k). Especially well-suited for
  this dataset-extraction stage since it's not used for interactive
  chat, so the free tier's ~40 req/minute rate limit isn't an issue. You
  can use it for the whole pipeline, or mix and match: NVIDIA Build for
  `persona_extractor.py`/`dialogue_extractor.py` (which need large
  context & consistent JSON), then switch back to your local Hermes for
  other needs.

  Models well-suited to this kind of structured extraction task (strong
  instruction-following + large context + consistent JSON):
  `meta/llama-3.3-70b-instruct`, `qwen/qwen2.5-72b-instruct`, or
  `mistralai/mistral-large`. Avoid "reasoning" models that wrap answers
  in `<think>...</think>` text (e.g. some DeepSeek-R1 variants) unless
  you add logic to strip that part before JSON parsing.

  Note: NVIDIA Build's free tier is meant for prototyping/personal use,
  not production traffic -- a good fit for building this dataset.

## Tuning chunk size vs. context window

Each call to the model contains: the system prompt (~500-800 tokens) +
a chunk of chapter text + (for persona_extractor specifically) the
accumulated profile so far + room for the JSON output. Rule of thumb:
total input should stay under ~60-70% of the model's context window,
leaving the rest for output.

| Model context window | PERSONA_CHUNK_CHARS | DIALOGUE_CHUNK_CHARS |
|---|---|---|
| ~40k (local Hermes-4-14B) | 6000 (default) | 4000 (default) |
| ~128k (Llama-3.3-70B, Qwen2.5-72B via NVIDIA Build) | 30000-40000 | 20000-25000 |

Increase these values in `.env` if you're using a large-context model --
this results in fewer API calls, and the model gets fuller scene/chapter
context (extraction quality is usually better). For a 350+ page novel,
this also significantly reduces the total number of calls compared to
sticking with small chunks.

`persona_extractor.py` also automatically compresses (`compress_profile`)
Elaina's profile once it crosses `MAX_PROFILE_CHARS`, so the profile
that keeps accumulating chapter by chapter doesn't balloon and eat up
the context window on novels with many chapters.

## If the script stops due to a rate limit / error

As of the latest update, `persona_extractor.py` and
`dialogue_extractor.py` automatically save a checkpoint
(`output/persona_checkpoint.json` and `output/dialogue_checkpoint.json`)
after every chapter is processed.

- **SHORT rate limit** (e.g. a per-minute limit) → the script
  automatically waits and continues on its own, no action needed.
- **LONG rate limit** (e.g. a daily token limit like Groq's free tier,
  which is only 100,000 tokens/day) → the script will STOP with a clear
  message about how long to wait. Your progress is **safe**, nothing is
  lost. Wait until the stated time (or the next day, if it's a daily
  limit), then re-run the exact same command:

  ```bash
  python persona_extractor.py     # or python run_all.py
  ```

  The script will resume from the last successfully completed chapter,
  not restart from chapter 1.

**Tokens-per-minute (TPM) limits**: some providers (Groq in particular)
also cap total tokens *per minute*, separately from the daily cap. If
you hit this, set `LLM_TPM_LIMIT` in `.env` (see the Groq block in
`.env.example`) -- the client will then proactively pace requests to
stay under that budget using a sliding 60-second window, instead of
firing requests and reacting to 429s after the fact. This is on by
default in the Groq config block, set a bit below the real limit as a
safety margin.

**A realistic note about Groq's free quota**: the 100,000 tokens/day
limit for `llama-3.3-70b-versatile` is fairly small for a long novel --
in practice, just ~14 chapters of profile extraction can exhaust the
daily quota. For an 80+ chapter novel, you'll likely need several
consecutive days (the checkpoint handles this safely), OR consider
switching to Claude (via OpenRouter/LiteLLM) to finish much faster
without being repeatedly interrupted by daily limits.

**If you add a new volume** and old checkpoints already exist, delete
both checkpoint files before rerunning (so the whole pipeline is
reprocessed chronologically and consistently from the start, as
explained in the "How to use it" section above):

```bash
rm -f output/persona_checkpoint.json output/dialogue_checkpoint.json
rm -f output/raw_dialogue_pairs.jsonl output/elaina_profile.json
```

## Improving quality

- If a PDF's extracted chapters look messy (headers/footers leaking
  in, oddly broken words), check the files in `text/` before moving on
  to the next stage -- EPUB is usually much cleaner than PDF.
- If you see a lot of `[WARN] Failed to get valid JSON`, the model
  probably needs a stricter prompt or a lower temperature. Try lowering
  `temperature` in the `llm_client.chat_json` calls in each script.
- Run `validate_dataset.py` every time after `build_dataset.py` -- it
  also prints 3 random examples so you can spot-check quality.

## Usage notes

This dataset is built from copyrighted material (the novel *Majo no
Tabitabi* by Jougi Shiraishi). This pipeline is intended for
personal/research use on your own machine, using a copy of the novel
you own. If you plan to share the resulting fine-tuned model or dataset
publicly/commercially, consider the copyright implications first.
