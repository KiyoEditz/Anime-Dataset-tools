"""
run_all.py

Jalankan seluruh pipeline secara berurutan dalam satu perintah:
    extract_text -> persona_extractor -> dialogue_extractor -> augment_synthetic -> build_dataset -> validate_dataset

Cara pakai:
    python run_all.py                  # pakai default 10 batch sintetis
    python run_all.py --synthetic 20   # ubah jumlah batch Q&A sintetis
    python run_all.py --skip-synthetic # lewati tahap augmentasi sintetis
"""

import argparse
import subprocess
import sys


def run(cmd: list[str], step_name: str):
    print(f"\n{'='*60}\n>> {step_name}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n[STOP] Tahap '{step_name}' gagal (exit code {result.returncode}). "
              f"Perbaiki dulu sebelum lanjut.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, default=10, help="Jumlah batch Q&A sintetis")
    parser.add_argument("--skip-synthetic", action="store_true", help="Lewati augmentasi sintetis")
    args = parser.parse_args()

    run(["extract_text.py"], "1/5 Ekstraksi teks dari EPUB/PDF")
    run(["persona_extractor.py"], "2/5 Membangun profil & ingatan Elaina")
    run(["dialogue_extractor.py"], "3/5 Ekstraksi pasangan dialog dari novel")

    if not args.skip_synthetic:
        run(["augment_synthetic.py", str(args.synthetic)], "4/5 Membuat Q&A sintetis tambahan")
    else:
        print("\n>> 4/5 Dilewati (--skip-synthetic)")

    run(["build_dataset.py"], "5/5 Menggabungkan dataset final")
    run(["validate_dataset.py"], "Validasi akhir")

    print("\nSemua tahap selesai. Dataset final ada di output/elaina_finetune_dataset.jsonl")


if __name__ == "__main__":
    main()
