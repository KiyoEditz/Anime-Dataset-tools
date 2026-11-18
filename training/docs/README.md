# Dokumentasi Lengkap Fine-Tuning Persona Elaina (Gemma 4 31B)

Selamat datang di repositori dokumentasi dan panduan operasional fine-tuning persona Elaina (*Wandering Witch: The Journey of Elaina*) pada model dasar multimodal **KiyoEditz/Gemma-4-Dark-Gemistry-31B** menggunakan **LoRA bfloat16** pada GPU cloud NVIDIA Blackwell 96 GB.

---

## 🗺️ Peta Navigasi Dokumen

Pilih dokumen sesuai dengan kebutuhan tugas Anda:

| Dokumen | Topik & Tujuan |
|---|---|
| [**01. Peta Alur & Diagram (Map Cara)**](01-peta-alur.md) | Diagram visual alur end-to-end, dependensi cell marimo, siklus hidup memori/disk, dan state machine training. |
| [**02. Persiapan Dataset**](02-persiapan-dataset.md) | Cara kerja `prepare_dataset.py`, filtering panjang, deduplikasi leksikal/fuzzy, split bab anti-bocor, dan format 3 bahasa (EN, ID, JP). |
| [**03. Unggah Dataset ke Hub**](03-upload-dataset-hub.md) | Cara mengunggah artefak dataset ke Hugging Face Hub dengan verifikasi SHA256 dua arah menggunakan `push_dataset_to_hub.py`. |
| [**04. Metode Training & Arsitektur**](04-metode-training.md) | Konsep LoRA bf16 vs QLoRA, pembekuan vision tower, response-only masking, dan kalkulasi memori GPU 96 GB. |
| [**05. Panduan Notebook Molab**](05-menjalankan-notebook-molab.md) | Langkah-demi-langkah menjalankan notebook Marimo (`molab_train_elaina.py`), variabel lingkungan, dan prosedur resume. |
| [**06. Chat Template Gemma**](06-chat-template-gemma.md) | Analisis template Jinja Gemma 4, penanganan role `system` native, token kontrol `<start_of_turn>`, dan pembuktian masking. |
| [**07. Evaluasi & Benchmark Persona**](07-evaluasi.md) | Kumpulan prompt uji tetap (multi-bahasa, out-of-domain, vision), metrik loss vs kualitas generasi nyata, dan paritas adapter. |
| [**08. Merge, GGUF & Inferensi**](08-merge-gguf-inferensi.md) | Prosedur penggabungan bobot LoRA, kompilasi GGUF quant, vision projector (`mmproj`), serta deployment via `llama-server`. |
| [**09. Panduan Troubleshooting**](09-troubleshooting.md) | Matriks penanganan kendala error umum: OOM, template mismatch, NaN loss, build error sm_120, dan kuota disk. |
| [**📁 Dokumen Perbaikan Kode (`fix/`)**](fix/README.md) | Rencana teknis komprehensif untuk AI/developer mengenai perbaikan bug kode, notebook marimo, dan inkonsistensi skrip. |

---

## 📖 Glosarium Konsep Penting

* **LoRA (Low-Rank Adaptation):** Metode pelatihan parameter-efisien dengan menyuntikkan matriks rank rendah ($W = W_0 + \frac{\alpha}{r} BA$) pada layer proyeksi model tanpa memodifikasi bobot asli base model selama pelatihan.
* **bf16 (bfloat16):** Format floating point 16-bit dengan 8-bit eksponen (rentang dinamis setara float32) dan 7-bit mantissa. Menghindari underflow/overflow gradien tanpa memerlukan loss scaling yang rumit.
* **Response-Only Loss Masking:** Teknik supervisi di mana nilai target (`labels`) pada token giliran sistem dan pengguna diisi `-100` (diabaikan oleh fungsi `CrossEntropyLoss`). Model hanya dilatih memprediksi token balasan dari Elaina.
* **End-of-Turn (`<end_of_turn>`):** Token kontrol penutup giliran bicara pada Gemma. Wajib ikut disupervisi agar model mengetahui kapan harus berhenti berbicara.
* **Vision Tower & Multimodal Projector:** Komponen visual pada Gemma 4 VLM. Dalam fine-tuning persona teks murni, komponen ini **dibekukan** (*frozen*) agar kemampuan visual aslinya tidak rusak (*catastrophic forgetting*).
* **`mmproj` (Multimodal Projector GGUF):** File bobot terpisah pada llama.cpp yang memetakan fitur gambar dari encoder visual ke ruang embedding bahasa model.
* **Leakage-Free Chapter Split:** Pembagian data train dan evaluasi/validasi berdasarkan keutuhan bab novel (bukan baris acak), memastikan teks cerita dan fakta adegan tidak pernah bocor antara train dan val di semua bahasa.
* **Environment Gate:** Pemeriksaan pre-flight otomatis di awal notebook yang mengukur kapasitas disk, RAM host, dan VRAM GPU sebelum mengizinkan pengunduhan model besar.

