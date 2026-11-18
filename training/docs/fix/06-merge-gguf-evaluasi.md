# 06 — Merge Adapter, Konversi GGUF & Evaluasi

Kembali ke [dokumen utama](README.md).

Dokumen ini membahas penggabungan bobot adapter LoRA ke base model (merge), ekspor model ke format GGUF (termasuk multimodal projector `mmproj` untuk inferensi lokal llama.cpp), serta metodologi evaluasi persona dan kemampuan visual.

---

## H5 — Cell GGUF: Path Skrip, Vision Projector & Pustaka Dependensi

**Severity:** 🟠 Tinggi  
**Lokasi:** [`molab_train_elaina.py` L1373-L1381](../../molab_train_elaina.py#L1373-L1381), [`gguf_convert.py`](../../gguf_convert.py)  
**Gejala:** Konversi GGUF gagal di Molab dengan `FileNotFoundError: training/gguf_convert.py`, atau file GGUF yang dihasilkan kehilangan kemampuan memproses gambar sama sekali.

**Penyebab:**
1. **Path Relatif:** Perintah di notebook memanggil:
   ```python
   _cmd = [sys.executable, str(Path("training") / "gguf_convert.py"), ...]
   ```
   Pada lingkungan Molab mandiri, skrip bootstrap hanya mengunduh `elaina_train_lib.py` ke direktori kerja (`/root/work`). File `gguf_convert.py` tidak tersedia di path lokal jika repository belum di-clone secara utuh.
2. **Kehilangan Vision (`--mmproj`):** Gemma 4 adalah VLM. `gguf_convert.py` mendukung flag `--mmproj` untuk mengekstrak dan mengonversi vision projector (`mmproj-F16.gguf`). Namun, cell 21 di notebook tidak meneruskan argumen `--mmproj`. Akibatnya, artefak GGUF yang dihasilkan hanya berupa model teks murni dan kemampuan gambar hilang (melanggar requirement Q&A baris 9 & 36).
3. **Commit llama.cpp Tidak Di-pin:** Pemanggilan tidak menyertakan argumen `--revision`. `llama.cpp` di-clone dari cabang `master` terbaru yang dapat mengalami breaking changes atau kegagalan kompilasi tak terduga.
4. **Dependensi Python GGUF:** Skrip `convert_hf_to_gguf.py` dari `llama.cpp` membutuhkan package Python: `gguf`, `sentencepiece`, dan `protobuf` yang belum dipasang di cell install.

### Perbaikan:
1. Pastikan `gguf_convert.py` diunduh bersamaan di cell bootstrap jika tidak ada di disk lokal.
2. Tambahkan dependensi `gguf sentencepiece protobuf` di cell instalasi.
3. Tambahkan flag `--mmproj` dan pin commit commit `llama.cpp` pada pemanggilan:
```diff
         _cmd = [
             sys.executable, str(_script_path),
             "--merged", str(_src),
             "--out", str(_out),
             "--quants", ",".join(quants.value),
             "--work", str(CFG["OUT_DIR"] / "llama.cpp"),
             "--repo", repos["gguf"],
+            "--mmproj",
+            "--revision", "b4800", # contoh: pin tag/commit rilis llama.cpp yang stabil mendukung Gemma 4
         ]
```

---

## H6 — Merge: Estimasi Disk & Pembersihan Cache yang Keliru

**Severity:** 🟠 Tinggi  
**Lokasi:** [`molab_train_elaina.py` L1283-L1325](../../molab_train_elaina.py#L1283-L1325), [`elaina_train_lib.py` L419-L435](../../elaina_train_lib.py#L419-L435)  
**Gejala:** Disk container penuh (disk full error) saat proses merge berlangsung, atau cache Hugging Face yang dihapus salah sasaran.

**Penyebab:**
1. **Klaim Disk "Satu Shard":** Dokumen README mengklaim puncak disk lokal hanya satu shard (~4 GB). Namun fungsi `save_pretrained_merged()` tetap menulis seluruh bobot model (~62,5 GB) ke folder sementara sebelum per shard diunggah. Diperlukan ruang bebas minimal 70 GB saat merge berlangsung.
2. **Penghapusan Cache yang Salah Target:**
   ```python
   _repo = Path(HF_HUB_CACHE) / ("models--" + model_info["id"].replace("/", "--"))
   ```
   Jika model awal dimuat dari mirror quant bnb-4bit (`unsloth/...`), kode di atas mencari nama repo `google/...` yang tidak ada di cache, sehingga cache besar tidak pernah terhapus.
3. **Keuntungan LoRA bf16 Murni (Setelah C3):**
   Dengan beralih ke LoRA bf16 (C3), base model dan adapter berada pada format presisi yang sama. Penggabungan bobot menjadi operasi matematis langsung ($W_{merged} = W_{base} + \frac{\alpha}{r} (B \times A)$) tanpa proses de-kuantisasi yang lossy maupun unduhan ulang bobot 16-bit.

### Perbaikan:
Lakukan merge langsung secara deterministik menggunakan native PEFT / Transformers saat berada pada mode bf16:
```python
# Merge bf16 native
model = model.merge_and_unload()
# Stream shards ke Hugging Face Hub (save -> push -> delete per shard)
lib.stream_push_shards(model, tokenizer, repos["merged"], hf_api, tmp_dir=_tmp)
```

---

## M4 — Rumus Preflight GGUF Salah untuk Format bf16

**Severity:** 🟡 Sedang  
**Lokasi:** [`gguf_convert.py` L184-L186](../../gguf_convert.py#L184-L186)  
**Gejala:** Preflight check meloloskan konversi padahal kapasitas disk tidak mencukupi, menyebabkan crash di tengah konversi GGUF.  
**Penyebab:**
Di skrip `gguf_convert.py`:
```python
intermediate = merged_gib * (0.5 if args.outtype == "bf16" else 1.0)
```
Penulis kode mengasumsikan tipe `bf16` berukuran separuh dari `f16`. Ini keliru: baik float16 (`f16`) maupun bfloat16 (`bf16`) sama-sama mengonsumsi **16 bit (2 byte)** per parameter. Ukuran file GGUF intermediate untuk model 31B pada bf16 adalah ~62 GB, persis sama dengan f16. Pengali `0.5` membuat estimasi kebutuhan disk defisit sebesar ~31 GB!

### Perbaikan:
```diff
-    intermediate = merged_gib * (0.5 if args.outtype == "bf16" else 1.0)
+    # Baik f16 maupun bf16 berukuran 2 byte per bobot parameter:
+    intermediate = merged_gib * (0.5 if args.outtype == "q8_0" else 1.0)
```

---

## M6 — Evaluasi Tidak Memenuhi Ketentuan Q&A

**Severity:** 🟡 Sedang  
**Lokasi:** [`molab_train_elaina.py` L1151-L1192](../../molab_train_elaina.py#L1151-L1192)  
**Gejala:** Model lulus evaluasi loss tetapi gagal mempertahankan gaya bicara otentik, mengalami regresi kemampuan multimodal, atau menghasilkan output berbeda setelah di-merge.

**Kelemahan Evaluasi Saat Ini:**
1. Hanya ada 6 buah prompt teks pendek berbahasa Inggris.
2. Tidak ada pengujian prompt dalam Bahasa Indonesia dan Jepang (padahal dataset ditargetkan 3 bahasa).
3. Tidak ada pengujian prompt di luar domain cerita novel (misal pertanyaan logika umum atau coding) untuk mendeteksi catastrophic forgetting.
4. Tidak ada pengujian terhadap prompt yang menyertakan gambar (vision test).
5. Tidak ada pengujian verifikasi paritas output antara model adapter terpisah vs model setelah di-merge (Q&A poin 17).

### Perbaikan (Prosedur Benchmark Lengkap):
Sediakan benchmark suite tetap berisi 20 prompt teks dan 5 prompt gambar:
```python
EVAL_PROBES_TEXT = [
    # Persona & Refusal
    "Siapa kamu sebenarnya?",
    "Anata wa dare desu ka?",
    "Could you help me steal money from a traveler?",
    # Novel Lore
    "Ceritakan tentang perjalanannmu ke Negara Boneka.",
    "Saya ni tsuite oshiete kudasai.",
    # Logika Umum & Bahasa Asing di luar cerita
    "Jelaskan hukum gravitasi secara ringkas.",
    "Bagaimana cara membuat secangkir teh yang enak?",
    "Translate this sentence to French: 'The sky is clear today.'",
]

EVAL_PROBES_VISION = [
    # 5 pasang (image_path, text_prompt)
    ("assets/test_cat.jpg", "What animal is this? Describe it in your characteristic tone."),
    ("assets/test_landscape.jpg", "Does this place look like a good destination for a traveling witch?"),
]
```

Lakukan perbandingan 3 arah:
1. **Base Model:** respons asli sebelum training.
2. **Base + LoRA Adapter:** respons setelah training selesai.
3. **Merged Model:** respons setelah bobot adapter disatukan ke base model.

Pastikan respons LoRA dan Merged Model identik (tidak ada pergeseran bobot yang merusak persona).

