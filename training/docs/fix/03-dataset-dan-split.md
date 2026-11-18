# 03 — Dataset, Kebocoran Validasi Lintas Bahasa & Split

Kembali ke [dokumen utama](README.md).

Dokumen ini menjelaskan analisis akar masalah dan solusi untuk pembagian dataset train/val (leakage-free across languages), pemotongan sequence length akibat prompt panjang, deteksi adegan narasi (scene vs dialogue), serta metadata dataset.

---

## C2 — Prompt Panjang Melebihi MAX_SEQ

**Severity:** 🔴 Kritis · **[VERIFIKASI DULU: V1]**  
**Lokasi:** [`elaina_train_lib.py` L321-L327](../../elaina_train_lib.py#L321-L327), [`molab_train_elaina.py` L902-L906](../../molab_train_elaina.py#L902-L906)  
**Gejala:** Tokenisasi crash dengan error:
```
RuntimeError: prompt alone is 2140 tokens, leaving -92 for the reply (max_len=2048)
... 346/3528 train rows failed to tokenise; first: ...
```
Akibatnya, cell pembuatan dataset TRL gagal total dan training tidak bisa dimulai.

**Penyebab:**
Di [`prepare_dataset.py` L65](../../prepare_dataset.py#L65), flag `--long-prompt-frac 0.10` menyuntikkan prompt lore lengkap (8.791 karakter) pada 10% baris training (346 baris). Prompt ini sendiri setara ~2.100–2.400 token pada tokenizer Gemma. Sementara itu, `CFG["MAX_SEQ"]` di notebook di-set `2048`. Di fungsi `tokenize_with_mask`:
```python
budget = max_len - len(prompt_ids)
if budget < 8:
    raise RuntimeError(f"prompt alone is {len(prompt_ids)} tokens, leaving {budget} for the reply...")
```

### Snippet Verifikasi V1:
Cek jumlah token sesungguhnya dari prompt panjang dengan tokenizer Gemma:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
long_text = open("output/elaina_system_prompt.txt", encoding="utf-8").read()
ids = tok.encode(long_text)
print(f"Panjang token prompt panjang: {len(ids)}")
```

### Opsi Perbaikan (Pilih Salah Satu / Kombinasi):
1. **Opsi Rekomendasi A (Naikkan MAX_SEQ ke 4096):**
   Pada VRAM 96 GB dengan LoRA bf16 (hanya melatih adapter), memori aktivasi untuk sequence 4096 sangat memadai.
   ```diff
   # Di CFG (Cell 1):
   -        "MAX_SEQ": 512 if SMOKE else 2048,
   +        "MAX_SEQ": 512 if SMOKE else 4096,
   ```
2. **Opsi B (Matikan Augmentasi Long Prompt pada Training):**
   Gunakan short prompt (~580 karakter, ~140 token) untuk 100% baris training. Hal ini sesuai dengan panduan inferensi: model dilatih gaya bicaranya dengan short prompt, dan saat inferensi prompt panjang dapat ditempelkan tanpa merusak token distribution.
   ```bash
   python training/prepare_dataset.py --long-prompt-frac 0.0
   ```
3. **Opsi C (Perlindungan Dinamis di Tokenizer):**
   Jangan lempar runtime error yang menggagalkan seluruh dataset jika ada baris yang melampaui budget; log baris tersebut dan lewati (skip) atau gunakan short prompt sebagai fallback untuk baris yang overflow.

---

## C7 — Split Per Bahasa Membocorkan Validasi Lintas Bahasa

**Severity:** 🔴 Kritis  
**Lokasi:** [`prepare_dataset.py` L384-L432](../../prepare_dataset.py#L384-L432)  
**Gejala:** Data kebocoran validasi (validation data leakage). Metrik eval loss tampak sangat bagus (misal < 0.8), namun kemampuan generalisasi model palsu karena model telah melihat isi percakapan adegan tersebut dalam bahasa lain selama training.

**Penyebab:**
Fungsi `chapter_split()` mengiterasi setiap bahasa secara terpisah dengan seed berbeda:
```python
for lang, recs in sorted(by_lang.items()):
    rng = random.Random(f"{seed}:{lang}")  # <-- RNG berbeda tiap bahasa!
    rng.shuffle(chapters)
    # ... memilih val_chapters[lang] secara independen
```
Akibatnya, bab `chapter_0012` bisa terpilih masuk `val_chapters["jp"]`, namun masuk `train_chapters["en"]` dan `train_chapters["id"]`. Karena dataset berasal dari konten novel yang sama di ketiga bahasa (Q&A Q5), model menghafal fakta cerita dari versi bahasa Inggris saat ditest pada versi Jepang!

Pengecekan overlap saat ini juga salah karena hanya memeriksa di dalam bahasa yang sama:
```python
for lang in by_lang:
    train_set = {r.chapter for r in train_rows if r.lang == lang}
    val_set = {r.chapter for r in val_rows if r.lang == lang}
    overlap = train_set & val_set  # <-- Tidak pernah mendeteksi overlap antar-bahasa!
```

### Perbaikan:
Bagi bab secara seragam pada tingkat novel induk (global chapter ID) sebelum mendistribusikan baris bahasa:

```diff
 def chapter_split(
     records: list[Record], val_frac: float, seed: int
 ) -> tuple[list[Record], list[Record], dict]:
-    by_lang: dict[str, list[Record]] = defaultdict(list)
-    for r in records:
-        by_lang[r.lang].append(r)
-    val_chapters: dict[str, list[str]] = {}
-    train_chapters: dict[str, list[str]] = {}
-    for lang, recs in sorted(by_lang.items()):
-        ... (pemilihan independen per bahasa) ...
+    # 1. Normalisasi nama bab acuan (misal: "chapter_0012.txt" atau normalized key)
+    all_chapters = sorted({r.chapter for r in records if not r.is_synthetic})
+    rng = random.Random(seed)
+    shuffled = list(all_chapters)
+    rng.shuffle(shuffled)
+
+    n_val = max(1, int(round(val_frac * len(shuffled))))
+    global_val_chapters = set(shuffled[:n_val])
+    global_train_chapters = set(shuffled[n_val:])
+
+    # 2. Assign baris berdasarkan keanggotaan bab global:
+    train_rows, val_rows = [], []
+    for r in records:
+        if r.is_synthetic:
+            # Synthetic pairs dibagi deterministik berdasarkan seed global
+            if rng.random() < val_frac:
+                val_rows.append(r)
+            else:
+                train_rows.append(r)
+        elif r.chapter in global_val_chapters:
+            val_rows.append(r)
+        else:
+            train_rows.append(r)
+
+    # 3. Assert leakage-free LINTAS SELURUH BAHASA:
+    all_train_chs = {r.chapter for r in train_rows if not r.is_synthetic}
+    all_val_chs = {r.chapter for r in val_rows if not r.is_synthetic}
+    cross_overlap = all_train_chs & all_val_chs
+    if cross_overlap:
+        raise AssertionError(f"FATAL CROSS-LINGUAL LEAK: Bab berikut muncul di train & val lintas bahasa: {cross_overlap}")
```

**Kriteria Selesai:**
Uji coba dengan dataset multi-bahasa mock menunjukkan 0 overlap bab antara `train_rows` (apapun bahasanya) dan `val_rows`.

---

## H9 — Lisensi Dataset Card Salah

**Severity:** 🟠 Tinggi  
**Lokasi:** [`push_dataset_to_hub.py` L32-L41](../../push_dataset_to_hub.py#L32-L41)  
**Gejala:** Dataset repo di Hugging Face mengklaim `license: apache-2.0` pada dataset card metadata.  
**Penyebab:** Teks dialog novel *Majo no Tabitabi* (Wandering Witch) memiliki hak cipta dari penerbit/penulis (Jougi Shiraishi / SB Creative). Mengklaim lisensi open-source Apache-2.0 dapat memicu pelanggaran hak cipta (DMCA take-down) atau redistribusi ilegal (Q&A poin 18).

### Perbaikan:
Ganti metadata lisensi menjadi non-komersial / riset terbatas / copyright note:
```diff
 CARD_TEMPLATE = """---
-license: apache-2.0
+license: other
+license_name: proprietary-educational-research-only
 language:
 {langs}
 tags:
   - persona
   - roleplay
   - sft
   - elaina
+  - not-for-commercial-use
 ---
```

---

## M7 — Dataset 100% Single-Turn Tanpa Data Umum

**Severity:** 🟡 Sedang  
**Gejala:** Model menjadi kaku ketika diajak mengobrol dalam giliran percakapan panjang (multi-turn), dan kemampuan penalaran umum menurun drastis karena semua data hanya berisi tanya-jawab satu giliran tentang Elaina (Q&A Q2 & poin 10).  
**Perbaikan:**
1. Tambahkan 10–20% percakapan 2–4 giliran (multi-turn) dari bab yang sama (dialog berurutan antara Elaina dan lawan bicaranya).
2. Sisipkan 5–10% dataset percakapan/instruksi umum (misalnya subset UltraChat / ShareGPT multi-turn yang bersih) agar model tetap mampu menjawab pertanyaan logika dan instruksi umum tanpa kehilangan kepribadian.

---

## M8 — Perhitungan Exact Tokens pada Transformers v5

**Severity:** 🟡 Sedang  
**Lokasi:** [`prepare_dataset.py` L510-L513](../../prepare_dataset.py#L510-L513)  
**Gejala:** `stats.json` melempar `TypeError: object of type 'BatchEncoding' has no len()` saat `--tokenizer` diaktifkan.  
**Penyebab:** Pada transformers versi baru, `tokenizer.apply_chat_template(..., tokenize=True)` dapat mengembalikan `BatchEncoding` atau dictionary berisi `input_ids` jika return_dict default aktif.  
**Perbaikan:**
```python
rendered = tokenizer.apply_chat_template(m["messages"], tokenize=True, add_generation_prompt=False)
if isinstance(rendered, dict) and "input_ids" in rendered:
    rendered = rendered["input_ids"]
total += len(rendered)
```

---

## M9 — Folder `output/` Tidak Ada di Workspace Lokal

**Severity:** 🟡 Sedang  
**Gejala:** Menjalankan `prepare_dataset.py` menghasilkan `SystemExit: No usable pairs found. Check your --source paths.`  
**Penyebab:** Folder `output/` di-gitignore atau belum diekstraksi di mesin lokal ini (hanya hasil kompilasi `training/out/` yang ada).  
**Panduan untuk AI Eksekutor:**
Jika ingin membangun ulang dataset:
- Pastikan ekstraksi dijalankan terlebih dahulu atau unduh `output/raw_dialogue_pairs.jsonl` dari storage/Google Drive/Hugging Face private dataset.
- Jika hanya menguji kode persiapan dataset, gunakan opsi mock/dry-run atau buat skrip sintetis pengujian di `scratch/`.

---

## M13 — Deteksi Scene & System Prompt untuk ID / JP

**Severity:** 🟡 Sedang  
**Lokasi:** [`prepare_dataset.py` L72-L78](../../prepare_dataset.py#L72-L78), [`elaina_system_prompt_short.txt`](../../elaina_system_prompt_short.txt)  
**Gejala:** Ketika file bahasa Indonesia dan Jepang dimasukkan:
1. Giliran user yang berupa narasi adegan tidak terdeteksi sebagai `scene` (karena regex `NARRATION_OPEN` hanya mencocokkan kata ganti bahasa Inggris seperti "She", "He", "Elaina"), sehingga tag `[Scene]` tidak diberikan.
2. System prompt hanya tersedia dalam bahasa Inggris.

### Perbaikan:
1. Sediakan regex penanda narasi adegan untuk bahasa Indonesia (misal: "Dia", "Ia", "Elaina", "Seseorang", "Tiba-tiba") dan Jepang (penanda kalimat naratif tanpa tanda kutip `「...」`).
2. Buat file `elaina_system_prompt_short_id.txt` dan `elaina_system_prompt_short_jp.txt` atau gunakan system prompt bahasa Inggris yang menginstruksikan respons dalam bahasa yang diminta pengguna.

---

## L4 — Nilai `volumes=12` Di-hardcode pada Dataset Card

**Severity:** ⚪ Rendah  
**Lokasi:** [`push_dataset_to_hub.py` L127](../../push_dataset_to_hub.py#L127)  
**Perbaikan:** Hitung jumlah volume unik secara dinamis dari nama file bab atau metadata manifest, bukan konstanta angka 12.

