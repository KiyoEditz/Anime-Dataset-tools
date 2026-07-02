"""
augment_synthetic.py

Tahap 4 (opsional tapi disarankan): hasilkan pasangan tanya-jawab TAMBAHAN
yang orisinal (bukan kutipan dari novel), berdasarkan profil Elaina yang
sudah dibangun. Ini membantu model generalisasi (menjawab pertanyaan yang
TIDAK ada persis di novel) dan mengurangi ketergantungan pada teks asli
secara verbatim.

Jalankan setelah persona_extractor.py:
    python augment_synthetic.py [jumlah_batch]

Hasil:
    output/synthetic_pairs.jsonl
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json

OUTPUT_DIR = Path("output")
PROFILE_FILE = OUTPUT_DIR / "elaina_profile.json"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_pairs.jsonl"

PAIRS_PER_BATCH = 8

SYSTEM_PROMPT = """Kamu membantu menyiapkan data fine-tuning chatbot untuk tokoh Elaina \
dari novel "Majo no Tabitabi / The Journey of Elaina", berdasarkan character bible \
yang diberikan.

Buat {n} pasangan tanya-jawab BARU dan ORISINAL (bukan kutipan/adegan persis dari \
novel) yang menguji apakah seseorang bisa berperan sebagai Elaina dengan konsisten. \
Variasikan jenis pertanyaan: tentang masa lalunya, pendapatnya soal sesuatu, \
reaksinya terhadap situasi hipotetis baru, hal-hal kecil sehari-hari, dst. \
Jawaban HARUS konsisten dengan kepribadian, nilai, gaya bicara, dan ingatan di \
character bible berikut:

{profile}

Format setiap entri:
{{"user_message": "pertanyaan/situasi dalam bahasa Inggris", "elaina_response": "jawaban Elaina dalam bahasa Inggris, sesuai gaya bicaranya"}}

Balas dengan JSON array berisi {n} entri tersebut. HANYA JSON, tanpa teks lain.
"""


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not PROFILE_FILE.exists():
        raise SystemExit("output/elaina_profile.json tidak ditemukan. Jalankan persona_extractor.py dulu.")

    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    profile_str = json.dumps(profile, ensure_ascii=False)

    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for _ in tqdm(range(n_batches), desc="Membuat Q&A sintetis"):
            prompt = SYSTEM_PROMPT.format(n=PAIRS_PER_BATCH, profile=profile_str)
            try:
                pairs = chat_json(prompt, "Hasilkan batch berikutnya, hindari duplikasi tema dari batch sebelumnya.", temperature=0.8)
            except ValueError as e:
                print(f"[WARN] Lewati satu batch: {e}")
                continue

            if not isinstance(pairs, list):
                continue

            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if not {"user_message", "elaina_response"}.issubset(pair.keys()):
                    continue
                out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nSelesai. {total} pasangan sintetis tersimpan di {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
