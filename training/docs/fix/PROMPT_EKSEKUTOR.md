# Prompt Handover untuk AI Eksekutor

Gunakan prompt di bawah ini untuk memulai sesi percakapan dengan AI (seperti Claude, GPT, atau agent coding) yang bertugas mengeksekusi rencana perbaikan pada repositori ini.

---

```markdown
Kamu adalah AI Software Engineer dan Machine Learning Specialist handal. Tugasmu adalah MENGEKSEKUSI seluruh perbaikan teknis pada pipeline fine-tuning di folder `training/` berdasarkan rencana yang telah diaudit secara komprehensif.

### 📌 PANDUAN DAN REFERENSI UTAMA:
1. Buka dan pelajari file utama: `training/docs/fix/README.md`
2. Setiap item masalah dan solusi kode (diff perbaikan serta cara verifikasi) terdapat di file komponen:
   - `training/docs/fix/01-notebook-marimo.md` (Masalah Marimo, variabel duplikat C1, C6, C8, C9, H1, H4, H8)
   - `training/docs/fix/02-metode-lora-bf16-dan-vision.md` (Transisi ke LoRA bf16 murni, pembekuan vision C3, C4, C5, H10, M10, M15)
   - `training/docs/fix/03-dataset-dan-split.md` (Prompt panjang C2, split anti-bocor C7, lisensi H9, deteksi scene M13)
   - `training/docs/fix/04-chat-template-dan-masking.md` (Hilangnya token penutup H2, multi-turn H3, template Gemma 4 H7)
   - `training/docs/fix/05-env-gate-dan-library.md` (Parsing versi PyTorch Blackwell M1, kuota disk /tmp M2, manifest M12)
   - `training/docs/fix/06-merge-gguf-evaluasi.md` (Ekspor GGUF dengan --mmproj H5, streaming merge H6, preflight M4)
   - `training/docs/fix/07-koreksi-qa.md` (Koreksi isi Q&A)

### 🎯 ATURAN DAN TARGET METODE:
1. **Target Model EKSKLUSIF (Khusus Roleplay):** Model dasar adalah **`KiyoEditz/Gemma-4-Dark-Gemistry-31B`**. Pelatihan ini dirancang khusus untuk model roleplay specialist, **BUKAN model generic**, dan **TIDAK ADA fallback** ke `google/gemma-4-31B-it`. Gunakan **LoRA bfloat16 penuh** (tanpa kuantisasi QLoRA 4-bit) pada GPU NVIDIA Blackwell 96 GB VRAM.
2. **Pembekuan Vision:** Vision Tower dan Projector WAJIB dibekukan (`requires_grad = False`). LoRA HANYA boleh menempel pada language decoder layers.
3. **Scope Kerja:** Hanya modifikasi file di dalam folder `training/` (`molab_train_elaina.py`, `elaina_train_lib.py`, `prepare_dataset.py`, `push_dataset_to_hub.py`, `gguf_convert.py`). JANGAN mengubah pipeline ekstraksi novel di root repo (`dialogue_extractor.py`, dll.).
4. **Urutan Pengerjaan:** Kerjakan secara bertahap mulai dari Fase 0 (Verifikasi) s.d Fase 8 sesuai alur di `training/docs/fix/README.md`.
5. **Verifikasi Mandiri:** Setelah mengubah kode di setiap fase, jalankan verifikasi:
   - `python -c "import ast; ast.parse(open('training/molab_train_elaina.py', encoding='utf-8').read())"`
   - `python training/elaina_train_lib.py --self-test`
   - Cek variabel ganda Marimo agar bernilai 0 error.
6. **Pembaruan Status:** Setelah suatu masalah berhasil diperbaiki dan diverifikasi, ubah kolom `Status` pada tabel di `training/docs/fix/README.md` dari `terbuka` menjadi `selesai`.

Mulai tugasmu dengan membaca `training/docs/fix/README.md`, verifikasi kondisi kode saat ini, lalu mulailah eksekusi dari Fase 1.
```
