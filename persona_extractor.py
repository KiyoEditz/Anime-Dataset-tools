"""
persona_extractor.py

Tahap 2: Bangun "character bible" Elaina secara bertahap (rolling profile),
membaca bab demi bab dan memperbarui profil tiap kali ada info baru
(kepribadian, ingatan/peristiwa penting, hubungan dengan karakter lain,
gaya bicara, nilai/pandangan hidup).

Jalankan setelah extract_text.py:
    python persona_extractor.py

Hasil:
    output/elaina_profile.json     -> profil terstruktur (bisa diaudit manual)
    output/elaina_system_prompt.txt -> system prompt siap pakai untuk fine-tuning
"""

import json
import os
from pathlib import Path

from tqdm import tqdm

from llm_client import chat_json, chat

TEXT_DIR = Path("text")
OUTPUT_DIR = Path("output")

# Berapa karakter teks bab yang dikirim per panggilan. Naikkan kalau model
# kamu punya context window besar (mis. 128k via NVIDIA Build) supaya jumlah
# panggilan lebih sedikit & konteks per panggilan lebih utuh. Untuk model
# 40k seperti Hermes-4-14B, nilai default ini sudah aman.
PERSONA_CHUNK_CHARS = int(os.environ.get("PERSONA_CHUNK_CHARS", "6000"))

# Profil JSON terus diakumulasi tiap bab dan dikirim PENUH di tiap panggilan.
# Untuk novel panjang (banyak bab), ini bisa makin besar dan mendekati batas
# context window. Begitu ukurannya (dalam karakter) melewati ambang ini,
# profil otomatis "dipadatkan" (duplikasi/redundansi dirapikan oleh model)
# sebelum lanjut ke bab berikutnya.
MAX_PROFILE_CHARS = int(os.environ.get("MAX_PROFILE_CHARS", "8000"))

# Checkpoint: menyimpan profil + nama bab terakhir yang selesai diproses,
# supaya kalau script terhenti di tengah jalan (error, rate limit harian,
# listrik mati, dll), rerun berikutnya MELANJUTKAN dari titik terakhir --
# bukan mengulang semua bab dari awal (yang buang-buang kuota API).
CHECKPOINT_FILE = OUTPUT_DIR / "persona_checkpoint.json"

PROFILE_SCHEMA_HINT = """
Format JSON yang HARUS kamu hasilkan (object tunggal):
{
  "personality_traits": ["..."],          // daftar sifat, gabungkan/perbarui yang sudah ada
  "speech_patterns": ["..."],              // ciri khas cara bicara, ungkapan favorit, nada bicara
  "values_and_worldview": ["..."],         // prinsip hidup, sikap terhadap dunia/orang lain
  "key_memories": [                        // peristiwa penting yang ia alami/ingat
    {"event": "...", "emotional_impact": "..."}
  ],
  "relationships": [
    {"name": "...", "description": "..."}
  ],
  "physical_description": "...",
  "notes": "..."                           // hal lain yang relevan tapi tidak masuk kategori di atas
}
"""

SYSTEM_PROMPT = f"""Kamu adalah asisten riset sastra yang membantu menyusun "character bible" \
untuk tokoh Elaina dari novel "Majo no Tabitabi / The Journey of Elaina".

Tugasmu: baca cuplikan bab berikut, lalu PERBARUI profil karakter yang sudah ada \
(jika ada info baru) dengan informasi dari bab ini. Jangan menghapus info lama yang \
masih valid, gabungkan/rapikan jika ada duplikasi atau kontradiksi (pakai versi yang \
lebih spesifik/terbaru). Fokus HANYA pada tokoh Elaina (bukan karakter lain), kecuali \
untuk bagian "relationships" yang menjelaskan hubungan Elaina dengan karakter tsb.

{PROFILE_SCHEMA_HINT}

Balas HANYA dengan JSON, tanpa teks pembuka/penutup, tanpa markdown code fence.
"""


COMPRESSION_SYSTEM_PROMPT = """Kamu membantu memadatkan character bible Elaina (dari novel \
'Majo no Tabitabi / The Journey of Elaina') yang sudah mulai panjang, TANPA menghilangkan \
informasi penting. Gabungkan trait/memory yang mirip atau redundan jadi satu entri yang \
lebih ringkas, buang yang murni duplikat, tapi PERTAHANKAN semua peristiwa/ingatan dan sifat \
yang berbeda maknanya. Pertahankan struktur schema JSON yang sama persis seperti input.

Balas HANYA dengan JSON, tanpa teks lain.
"""


def compress_profile(profile: dict) -> dict:
    """Padatkan profil kalau sudah terlalu besar, supaya tidak membengkak ke context window."""
    current_size = len(json.dumps(profile, ensure_ascii=False))
    if current_size <= MAX_PROFILE_CHARS:
        return profile

    try:
        compressed = chat_json(
            COMPRESSION_SYSTEM_PROMPT,
            json.dumps(profile, ensure_ascii=False),
            temperature=0.2,
        )
        if isinstance(compressed, dict):
            new_size = len(json.dumps(compressed, ensure_ascii=False))
            print(f"[INFO] Profil dipadatkan: {current_size} -> {new_size} karakter")
            return compressed
    except ValueError as e:
        print(f"[WARN] Gagal memadatkan profil ({e}), lanjut pakai versi sebelumnya.")

    return profile


def load_checkpoint():
    """Kembalikan (nama_bab_terakhir_selesai, profil) dari checkpoint, atau (None, {}) kalau belum ada."""
    if not CHECKPOINT_FILE.exists():
        return None, {}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return data.get("last_completed_chapter"), data.get("profile", {})
    except json.JSONDecodeError:
        print("[WARN] Checkpoint rusak/tidak terbaca, mulai dari awal.")
        return None, {}


def save_checkpoint(last_completed_chapter: str, profile: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {"last_completed_chapter": last_completed_chapter, "profile": profile},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def merge_profile(existing: dict, update: dict) -> dict:
    """Gabungkan profil lama dengan hasil update dari satu bab."""
    if not existing:
        return update

    merged = dict(existing)
    for list_key in ("personality_traits", "speech_patterns", "values_and_worldview"):
        merged[list_key] = list(
            dict.fromkeys(existing.get(list_key, []) + update.get(list_key, []))
        )

    merged["key_memories"] = existing.get("key_memories", []) + update.get("key_memories", [])

    # relationships: gabung berdasarkan nama, deskripsi terbaru menang
    rel_map = {r["name"]: r for r in existing.get("relationships", [])}
    for r in update.get("relationships", []):
        rel_map[r["name"]] = r
    merged["relationships"] = list(rel_map.values())

    merged["physical_description"] = (
        update.get("physical_description") or existing.get("physical_description", "")
    )
    merged["notes"] = (existing.get("notes", "") + "\n" + update.get("notes", "")).strip()

    return merged


def build_profile():
    chapter_files = sorted(TEXT_DIR.glob("chapter_*.txt"))
    if not chapter_files:
        raise SystemExit("Tidak ada file bab di text/. Jalankan extract_text.py dulu.")

    last_completed, profile = load_checkpoint()

    if last_completed:
        chapter_names = [p.name for p in chapter_files]
        if last_completed in chapter_names:
            resume_idx = chapter_names.index(last_completed)
            skipped = chapter_files[: resume_idx + 1]
            chapter_files = chapter_files[resume_idx + 1 :]
            print(
                f"[INFO] Melanjutkan dari checkpoint: {len(skipped)} bab sudah "
                f"selesai sebelumnya (terakhir: {last_completed}), "
                f"sisa {len(chapter_files)} bab."
            )
        else:
            print(
                f"[WARN] Checkpoint menyebut '{last_completed}' tapi file itu tidak "
                f"ditemukan lagi di text/ (mungkin daftar bab berubah karena "
                f"nambah/ubah volume). Memproses ULANG SEMUA bab dari awal supaya "
                f"konsisten. Hapus output/persona_checkpoint.json dulu kalau ini "
                f"tidak diinginkan."
            )
            profile = {}

    if not chapter_files:
        print("[INFO] Semua bab sudah selesai diproses (sesuai checkpoint).")

    for chap_path in tqdm(chapter_files, desc="Membangun profil Elaina"):
        chapter_text = chap_path.read_text(encoding="utf-8")

        user_prompt = (
            "Profil yang sudah ada sejauh ini (JSON, bisa kosong kalau ini bab pertama):\n"
            f"{json.dumps(profile, ensure_ascii=False)}\n\n"
            "Cuplikan bab berikutnya:\n"
            f"{chapter_text[:PERSONA_CHUNK_CHARS]}"
        )

        try:
            update = chat_json(SYSTEM_PROMPT, user_prompt)
            if isinstance(update, dict):
                profile = merge_profile(profile, update)
                profile = compress_profile(profile)
        except ValueError as e:
            print(f"[WARN] Lewati {chap_path.name}: {e}")

        # Simpan checkpoint SETELAH tiap bab (berhasil ataupun di-skip karena
        # ValueError) supaya kalau bab BERIKUTNYA gagal dengan error yang tidak
        # tertangani (mis. rate limit harian), progress sampai bab ini tetap aman.
        save_checkpoint(chap_path.name, profile)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "elaina_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print("Profil tersimpan di output/elaina_profile.json")
    return profile


def write_system_prompt(profile: dict):
    """Minta model merangkum profil JSON menjadi system prompt siap pakai."""
    summarizer_prompt = (
        "Berikut adalah character bible terstruktur untuk Elaina dari "
        "'Majo no Tabitabi / The Journey of Elaina', hasil ekstraksi dari novel:\n\n"
        f"{json.dumps(profile, indent=2, ensure_ascii=False)}\n\n"
        "Tulis ulang ini menjadi SATU system prompt (3-6 paragraf, bahasa Inggris) "
        "yang menginstruksikan sebuah AI untuk berperan sebagai Elaina secara konsisten: "
        "kepribadiannya, gaya bicaranya, nilai-nilai yang ia pegang, dan ingatan/latar "
        "belakang penting yang harus ia ingat. Tulis dalam sudut pandang instruksi ke AI "
        "(\"You are Elaina...\"), bukan dalam sudut pandang Elaina sendiri."
    )
    system_prompt_text = chat(
        "Kamu adalah asisten yang ahli menulis system prompt karakter AI yang detail dan konsisten.",
        summarizer_prompt,
        temperature=0.3,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "elaina_system_prompt.txt").write_text(system_prompt_text.strip(), encoding="utf-8")
    print("System prompt tersimpan di output/elaina_system_prompt.txt")


if __name__ == "__main__":
    final_profile = build_profile()
    write_system_prompt(final_profile)
