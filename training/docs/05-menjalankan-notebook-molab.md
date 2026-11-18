# 05 — Panduan Menjalankan Notebook di Molab (`molab_train_elaina.py`)

Dokumen ini adalah *runbook* langkah-demi-langkah untuk mengeksekusi pelatihan di platform cloud **Molab** menggunakan notebook interaktif berbasis **Marimo** pada GPU NVIDIA Blackwell 96 GB.

---

## 1. Mengenal Arsitektur Marimo Notebook

`molab_train_elaina.py` adalah notebook Python berbasis Marimo (bukan Jupyter `.ipynb` tradisional):
* Seluruh cell adalah fungsi Python murni yang diatur dalam graf dependensi (*Directed Acyclic Graph*).
* Eksekusi cell berlangsung secara **reaktif**: perubahan pada cell atas secara otomatis memicu cell di bawahnya yang bergantung padanya.
* Operasi komputasi berat dan berbiaya VRAM diletakkan di balik tombol antarmuka (`mo.ui.run_button`), sehingga proses berat hanya berjalan saat pengguna secara sadar menekan tombol tersebut.

---

## 2. Variabel Lingkungan (*Environment Variables*)

Sebelum meluncurkan notebook, Anda dapat mengatur parameter melalui variabel lingkungan:

| Variabel | Default | Fungsi |
|---|---|---|
| `ELAINA_MODE` | `full` | Mode eksekusi. Isi `smoke` untuk uji coba kilat dengan model kecil (5 langkah optimizer). |
| `ELAINA_WORKDIR` | `/root/work` | Direktori kerja penyimpanan sementara artefak model dan cache di Molab. |
| `ELAINA_REPO_SLUG` | `KiyoEditz/elaina-dataset-tool` | Repositori GitHub untuk mengunduh modul shared helper `elaina_train_lib.py`. |
| `HF_TOKEN` | *(Kosong)* | Token Hugging Face write role untuk bypass login manual. |

---

## 3. Alur Eksekusi Langkah-demi-Langkah (Per Cell)

### Bagian A: Inisialisasi & Verifikasi Perangkat Keras
1. **Cell 1–3 (Setup & Konfigurasi):** Mengimpor library standar, memuat konfigurasi `CFG`, dan menampilkan tabel estimasi kebutuhan disk.
2. **Cell 4 (Environment Gate - Titik Kritis 1):**
   * Mengukur kapasitas disk bebas pada `/root/work`, RAM sistem, dan spesifikasi GPU.
   * Memastikan GPU Blackwell terdeteksi dengan kapabilitas komputasi yang valid.
   * *Perhatian:* Jika status menunjukkan `tier: blocked`, jangan lanjutkan! Minta penambahan volume disk Molab (minimal 150 GB untuk alur 31B).

### Bagian B: Otentikasi & Dataset
3. **Cell 5–6 (Hugging Face Login & Repositories):**
   * Masukkan Hugging Face Token (wajib memiliki **Write scope**), lalu tekan tombol **Log in**.
   * Tentukan nama 4 repositori privat yang akan dibuat otomatis:
     * Dataset repo: `username/elaina-gemma4-dataset`
     * Adapter repo: `username/elaina-gemma4-31b-lora`
     * Merged repo: `username/elaina-gemma4-31b`
     * GGUF repo: `username/elaina-gemma4-31b-GGUF`
4. **Cell 7 (Instalasi Stack):** Tekan tombol untuk menginstal/memverifikasi ketergantungan PyTorch cu128, transformers, PEFT, TRL, dan Unsloth.
5. **Cell 8–9 (Resolve Model & Tarik Dataset):**
   * Memeriksa akses ke model dasar roleplay `KiyoEditz/Gemma-4-Dark-Gemistry-31B`.
   * Mengunduh dataset bersih yang telah disiapkan sebelumnya dari Hugging Face Hub.

### Bagian C: Validasi Template & Pemuatan Model
6. **Cell 10–11 (Tokenizer & Chat Template Proof - Titik Kritis 2):**
   * Membedah Jinja template model dan memverifikasi token pembuka/penutup.
   * Menampilkan bukti bahwa token prompt sistem & user ter-masking dengan nilai `-100`, dan token jawaban asisten memiliki rentang supervisi kontigu.
   * *Aturan Keras:* Jika cell ini melaporkan error pada template, **jangan lanjutkan ke tahap training**.
7. **Cell 12 (Load Base Model):** Tekan tombol **Load base model** untuk memuat bobot 31B dalam presisi `bfloat16`.
8. **Cell 13 (Pasang Adapter LoRA):**
   * Membekukan parameter vision tower dan multimodal projector.
   * Menyuntikkan adapter LoRA ($r=32, \alpha=64$) pada language model layers.
   * Memastikan rasio bobot trainable berada di kisaran 0,1% – 1,0%.

### Bagian D: Pelatihan & Pemantauan
9. **Cell 14–15 (Tokenisasi & Konfigurasi Trainer):**
   * Mengonversi teks dataset menjadi token input dan tensor label yang telah di-masking.
   * Mengonfigurasi `Trainer` dengan optimizer `adamw_torch_fused`, cosine scheduler, dan sinkronisasi checkpoint berkala ke Hugging Face Hub.
10. **Cell 16 (Eksekusi Training):**
    * Centang kotak konfirmasi biaya GPU hours.
    * Tekan tombol **Start / resume training**.
    * Trainer akan berjalan selama 2 epoch. Setiap 100 step, checkpoint otomatis disimpan dan diunggah ke Hub.

### Bagian E: Evaluasi, Ekspor & Pembersihan
11. **Cell 17–18 (Evaluasi & Sampel Inferensi):**
    * Mengukur loss pada bab evaluasi yang di-hold-out.
    * Menghasilkan sampel teks berdampingan antara Base Model vs LoRA Model terhadap 6 prompt uji.
12. **Cell 19 (Upload Final Adapter):** Mengunggah adapter final beserta `adapter_config.json` dan README model card ke Hugging Face.
13. **Cell 20 (Merge Shards ke bf16):**
    * Menyatukan bobot LoRA ke bobot base model.
    * Mengunggah file safetensors per shard (~4 GB) dan langsung menghapus file lokal untuk menjaga ketersediaan disk.
14. **Cell 21 (Konversi GGUF & Vision Projector):** Mengompilasi `llama.cpp`, membuat intermediate GGUF, mengonversi ke varian quant (Q4_K_M, Q6_K), serta mengekstrak `mmproj-F16.gguf`.
15. **Cell 22–23 (Pembersihan Disk & Laporan Selesai):** Menghapus artefak lokal berukuran besar dan menampilkan ringkasan tautan 4 repositori Hugging Face yang siap digunakan.

---

## 4. Prosedur Melanjutkan Sesi yang Terputus (*Resume*)

Jika sesi Molab terputus akibat batas waktu (*session timeout*) atau koneksi internet terputus:
1. Luncurkan kembali notebook Molab pada instance GPU baru.
2. Jalankan kembali cell atas hingga **Cell 6** (pastikan nama repositori adapter diisi sama dengan sesi sebelumnya).
3. Jalankan pemuatan model dan pembuatan trainer (**Cell 12–15**).
4. Tekan tombol **Start / resume training** di **Cell 16**:
   * Skrip secara otomatis mencari subfolder `checkpoint-*` terbaru di repositori Hugging Face Adapter.
   * Checkpoint diunduh dan pelatihan otomatis dilanjutkan persis dari langkah terakhir yang tersimpan tanpa mengulang dari awal!

