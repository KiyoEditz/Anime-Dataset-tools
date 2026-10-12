"""
run_all.py

Run the entire pipeline in sequence with one command:
    extract_text -> persona_extractor -> dialogue_extractor -> augment_synthetic -> build_dataset -> validate_dataset

Usage:
    python run_all.py                  # use the default of 10 synthetic batches
    python run_all.py --synthetic 20   # change the number of synthetic Q&A batches
    python run_all.py --skip-synthetic # skip the synthetic augmentation stage
"""

import argparse
import subprocess
import sys


def run(cmd: list[str], step_name: str):
    print(f"\n{'='*60}\n>> {step_name}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n[STOP] Step '{step_name}' failed (exit code {result.returncode}). "
              f"Fix this before continuing.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true", help="Skip text extraction (use existing confirmed text/ files)")

    parser.add_argument("--synthetic", type=int, default=10, help="Number of synthetic Q&A batches")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic augmentation")
    args = parser.parse_args()

    if not args.skip_extract:
        # If confirm_xhtml already exists, run step text; otherwise run --all
        run(["extract_text.py", "--all"], "1/5 Extracting text from EPUB/PDF")
    else:
        print("\n>> 1/5 Skipped extraction (using existing text/ files)")

    run(["persona_extractor.py"], "2/5 Building Elaina's profile & memories")

    run(["dialogue_extractor.py"], "3/5 Extracting dialogue pairs from the novel")

    if not args.skip_synthetic:
        run(["augment_synthetic.py", str(args.synthetic)], "4/5 Generating additional synthetic Q&A")
    else:
        print("\n>> 4/5 Skipped (--skip-synthetic)")

    run(["build_dataset.py"], "5/5 Assembling the final dataset")
    run(["validate_dataset.py"], "Final validation")

    print("\nAll stages complete. Final dataset is at output/elaina_finetune_dataset.jsonl")


if __name__ == "__main__":
    main()
