"""
run_all.py

Run the entire pipeline in sequence with one command:
    extract_text -> persona_extractor -> dialogue_extractor -> augment_synthetic -> build_dataset -> validate_dataset

Usage:
    py -3.14 run_all.py --EN               # run full English pipeline (confirm_xhtml-EN, text-EN, output-EN)
    py -3.14 run_all.py --JP               # run full Japanese pipeline (confirm_xhtml-JP, text-JP, output-JP)
    py -3.14 run_all.py --JP --synthetic 20
    py -3.14 run_all.py --JP --skip-synthetic
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], step_name: str, env: dict = None):
    print(f"\n{'='*60}\n>> {step_name}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd, env=env)
    if result.returncode != 0:
        print(f"\n[STOP] Step '{step_name}' failed (exit code {result.returncode}). "
              f"Fix this before continuing.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run complete Elaina dataset pipeline")
    parser.add_argument("--lang", choices=["EN", "JP", "ID", "en", "jp", "id"], default=None, help="Target language mode")
    parser.add_argument("--EN", dest="lang_en", action="store_true", help="Process English novel dataset (uses *-EN folders)")
    parser.add_argument("--JP", dest="lang_jp", action="store_true", help="Process Japanese novel dataset (uses *-JP folders)")
    parser.add_argument("--ID", dest="lang_id", action="store_true", help="Process Indonesian novel dataset (uses *-ID folders)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip text extraction (use existing confirmed text/ files)")
    parser.add_argument("--synthetic", type=int, default=10, help="Number of synthetic Q&A batches")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic augmentation")
    args = parser.parse_args()

    # Determine language mode
    if args.lang_jp:
        target_lang = "JP"
    elif args.lang_en:
        target_lang = "EN"
    elif args.lang_id:
        target_lang = "ID"
    elif args.lang:
        target_lang = args.lang.upper()
    else:
        target_lang = os.environ.get("DATASET_LANG", "EN").upper()

    print(f"[PIPELINE] Selected mode: [{target_lang}]")
    print(f"  confirm folder : confirm_xhtml-{target_lang}")
    print(f"  text folder    : text-{target_lang}")
    print(f"  output folder  : output-{target_lang}")

    env = os.environ.copy()
    env["DATASET_LANG"] = target_lang
    lang_flag = f"--{target_lang}"

    if not args.skip_extract:
        run(["extract_text.py", "--all", lang_flag], f"1/5 Extracting text from EPUB/PDF [{target_lang}]", env=env)
    else:
        print(f"\n>> 1/5 Skipped extraction (using existing text-{target_lang}/ files)")

    run(["persona_extractor.py", lang_flag], f"2/5 Building Elaina's profile & memories [{target_lang}]", env=env)

    run(["dialogue_extractor.py", lang_flag], f"3/5 Extracting dialogue pairs from novel [{target_lang}]", env=env)

    if not args.skip_synthetic:
        run(["augment_synthetic.py", str(args.synthetic), lang_flag], f"4/5 Generating additional synthetic Q&A [{target_lang}]", env=env)
    else:
        print(f"\n>> 4/5 Skipped (--skip-synthetic)")

    run(["build_dataset.py", lang_flag], f"5/5 Assembling the final dataset [{target_lang}]", env=env)
    run(["validate_dataset.py", lang_flag], f"Final validation [{target_lang}]", env=env)

    print(f"\nAll stages complete. Final dataset is at output-{target_lang}/elaina_finetune_dataset.jsonl")


if __name__ == "__main__":
    main()
