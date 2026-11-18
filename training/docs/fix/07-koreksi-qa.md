# 07 — Koreksi & Penyelarasan Dokumen Q&A

Kembali ke [dokumen utama](README.md).

File [`lora_gemma_elaina_qa.txt`](../../../lora_gemma_elaina_qa.txt) berisi tanya-jawab dan daftar kesalahan umum dalam fine-tuning persona Elaina pada Gemma. Berdasarkan evaluasi teknis mendalam (terutama catatan pada baris 10 dokumen tersebut), berikut adalah koreksi teknis dan klarifikasi akurasi yang harus dipahami oleh tim/AI:

---

### QA1 — Ambiguitas Rujukan "Nomor 5" pada Masalah Validasi
* **Teks Asli (Baris 21):**
  > `- Kebocoran data validasi (lihat jawaban nomor 5).`
* **Analisis Masalah:**
  Pada bagian Q&A di atas, Q5 memang membahas dataset 3 bahasa dan kebocoran validasi. Namun pada bagian daftar "TAMBAHAN" di bawah (baris 64), nomor 5 adalah *"Learning rate terlalu tinggi"*, sedangkan masalah kebocoran data validasi bernomor 7 (*"Data bocor ke validasi"*). Hal ini menimbulkan kebingungan bagi pembaca.
* **Koreksi:**
  Perjelas rujukan menjadi: `(lihat jawaban Q5 di atas atau poin 7 di daftar tambahan)`.

---

### QA2 — Penanganan Role System & Chat Template Gemma Antar-Versi
* **Teks Asli (Baris 26):**
  > `Instruksi persona digabung ke giliran user pertama bila versi Gemma yang dipakai tidak punya role system native. Cek chat template tokenizer dari versi yang dipakai, karena ini bisa berbeda antar versi.`
* **Akurasi Teknis Antar Versi:**
  1. **Gemma 1 & Gemma 2:** Tidak memiliki role `system`. Template Hugging Face akan melempar error jika ada role `system`. Solusinya adalah memformat instruksi sistem ke awal giliran `user` pertama: `f"{system_prompt}\n\n{user_message}"`.
  2. **Gemma 3:** Tokenizer menerima role `system`, namun di dalam template Jinja secara otomatis menggabungkannya ke dalam giliran user pertama di bawah tag `<start_of_turn>user`.
  3. **Gemma 4:** **Mendukung role `system` secara NATIVE.** Terdapat tag kontrol `<|turn>system\n...<turn|>\n` (`sot_token="<|turn>"`, `eot_token="<turn|>"`). Penggabungan manual ke user turn tidak lagi diperlukan untuk Gemma 4, namun jika digabungkan tetap kompatibel. Selain itu, Gemma 4 memperkenalkan kanal token khusus pemikiran (`<|channel>thought\n<channel|>` yang disuntikkan saat `add_generation_prompt=True`), sehingga `tokenize_with_mask` di `elaina_train_lib.py` telah diperbarui agar menyertakan blok pemikiran kosong tersebut ke dalam span prompt yang di-mask (`-100`) dan mendukung sampel multi-turn (`H3`, `H7`, `M7`).

---

### QA3 — Kelayakan LoRA bf16 Penuh pada GPU Blackwell 96 GB
* **Teks Asli (Baris 31):**
  > `Lebih dari cukup. LoRA bf16 penuh muat tanpa perlu QLoRA. Yang perlu dicek: pastikan PyTorch dan library pendukungnya versi baru yang mendukung Blackwell (CUDA 12.8 ke atas)...`
* **Klarifikasi Teknis:**
  Pernyataan ini secara teoritis **benar**, dengan rincian:
  * Base 31B bf16 membutuhkan ~62,5 GB VRAM statis.
  * Sisa ~33,5 GB VRAM cukup untuk menampung bobot adapter LoRA, optimizer states adapter (hanya ~1 GB), dan aktivasi.
  * **Syarat Mutlak:** *Gradient checkpointing* (`use_reentrant=False`) wajib aktif! Tanpa gradient checkpointing, memori aktivasi untuk sequence length 2048/4096 akan melebihi sisa 33,5 GB dan memicu OOM CUDA.
  * **Inkonsistensi Kode:** Walaupun Q&A menyarankan LoRA bf16, skrip `molab_train_elaina.py` yang ada di repositori justru mengimplementasikan QLoRA 4-bit (C3).

---

### QA4 — Target Modules LoRA & Pembekuan Vision Tower
* **Teks Asli (Baris 36):**
  > `LoRA tidak mengubah bobot asli. Risikonya ada di target module. Kalau memakai all-linear, vision tower dan projector ikut dilatih. Solusi: batasi target hanya ke layer language model (misalnya dengan regex pada nama modul) dan bekukan vision tower...`
* **Klarifikasi Teknis:**
  * Kalimat *"LoRA tidak mengubah bobot asli"* hanya berlaku selama adapter tersimpan sebagai layer tambahan terpisah (`adapter_model.safetensors`). Begitu adapter disatukan ke base model (*merge & unload*), bobot dasar berubah permanen.
  * Menghindari `all-linear` saja tidak cukup jika daftar modul target (`q_proj`, `v_proj`, dll.) juga ada di modul visual encoder. Wajib menggunakan regex spesifik (misal `.*language_model.*`) dan melakukan pengecekan `requires_grad = False` pada seluruh layer vision.
  * Saat ekspor ke GGUF untuk inferensi lokal, flag `--mmproj` wajib disertakan agar vision projector ikut diekspor, jika tidak model GGUF akan kehilangan kemampuan vision.

---

### QA5 — Kebocoran Validasi pada Dataset Tiga Bahasa
* **Teks Asli (Baris 41):**
  > `...Yang lebih berbahaya adalah kebocoran validasi: kalau satu adegan versi JP ada di training dan versi ID-nya di validasi, loss validasi terlihat bagus padahal menipu. Pisahkan validasi per adegan atau sumber, bukan per bahasa...`
* **Catatan Status:**
  Penjelasan konsep pada Q&A ini **sangat tepat dan kritis**. Namun, implementasi di [`prepare_dataset.py` L384-L404](../../prepare_dataset.py#L384-L404) justru melakukan pelanggaran langsung terhadap peringatan ini dengan membagi bab train/val menggunakan generator acak terpisah per bahasa (lihat [03-dataset-dan-split.md § C7](03-dataset-dan-split.md#c7--split-per-bahasa-membocorkan-validasi-lintas-bahasa)).

---

### QA6 — Dampak fp16 pada Gemma
* **Teks Asli (Baris 60-62):**
  > `4. Memakai fp16, bukan bf16. Gejala: loss menjadi NaN atau tidak stabil. Solusi: pakai bf16 di GPU yang mendukung.`
* **Catatan Status:**
  Benar. Di skrip training terdapat bug (C5) di mana flag bf16 jatuh ke `False` sehingga pelatihan secara otomatis berjalan di fp16 tanpa disadari user.

---

### QA7 — Evaluasi dan Pemilihan Checkpoint Terbaik
* **Teks Asli (Baris 68-70 & 100-102):**
  > `...pilih yang terbaik (bukan otomatis yang terakhir).` dan `14. Mengevaluasi hanya dari loss... siapkan 15-20 prompt uji tetap...`
* **Catatan Kode:**
  Skrip saat ini menggunakan argumen `load_best_model_at_end=True` dengan `metric_for_best_model="eval_loss"`, yang berarti skrip secara otomatis memilih checkpoint berdasarkan loss validasi novel (yang menurut poin 14 menipu). Sebaiknya simpan 3 checkpoint teratas dan pilih manual berdasarkan pengujian sampel inferensi.

---

### QA8 — Pemotongan Respons (Truncation) & Token End-of-Turn
* **Teks Asli (Baris 56-58):**
  > `3. Token EOS/end-of-turn hilang. Gejala: model terus bicara tanpa berhenti.`
* **Tambahan:**
  Penyebab hilangnya token EOS tidak hanya dari template yang salah, tetapi juga dari proses pemotongan teks (*token truncation*) di collator jika panjang respons melebihi batas `MAX_SEQ`.

---

### QA9 — Dry Run pada Keluarga Model yang Sama
* **Teks Asli (Baris 96-98):**
  > `13. Tidak melakukan dry run. Solusi: jalankan 50-100 sampel selama beberapa puluh step...`
* **Tambahan:**
  Dry run (smoke test) tidak ada gunanya jika menggunakan arsitektur model yang berbeda (misal Qwen untuk menguji Gemma), karena format token kontrol, nama layer vision, dan konfigurasi chat template sama sekali tidak teruji. Dry run harus memakai varian mini dari keluarga yang sama (misal `gemma-4-E2B-it`).

---

### QA10 — Status Hak Cipta & Lisensi Dataset
* **Teks Asli (Baris 116-117):**
  > `18. Hak cipta dataset. Catatan: teks novel berhak cipta. Gunakan untuk keperluan pribadi dan jangan bagikan dataset atau model hasilnya.`
* **Catatan Kode:**
  Skrip `push_dataset_to_hub.py` secara tidak sengaja menyertakan label `license: apache-2.0` pada template metadata dataset card, yang bertentangan dengan klausul ini.

---

### QA11 — Ekspektasi "Work on the First Try"
* **Teks Asli (Baris 6):**
  > `- Method: LoRA, bf16, single run that should work on the first try.`
* **Koreksi Realitas:**
  Dengan kondisi skrip yang ada saat ini sebelum audit (mengandung 7 bug fatal seperti variabel duplikat di marimo, prompt overflow, dan kebocoran validasi), eksekusi sekali jalan pasti gagal. Setelah seluruh checklist perbaikan di [dokumen utama](README.md) diimplementasikan, barulah target run pertama berhasil dapat dicapai secara realistis.

