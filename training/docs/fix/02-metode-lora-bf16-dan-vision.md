# 02 — Metode LoRA bf16 & Pembekuan Vision

Kembali ke [dokumen utama](README.md).

Dokumen ini merinci perbaikan untuk aspek metode pemodelan, presisi pelatihan, penargetan modul LoRA, pembekuan komponen multimodal (vision tower & projector), konfigurasi optimizer, dan manajemen VRAM pada GPU NVIDIA Blackwell 96 GB.

---

## C3 — Loader Memakai QLoRA 4-bit, Requirement LoRA bf16

**Severity:** 🔴 Kritis  
**Lokasi:** [`molab_train_elaina.py` L741-L777](../../molab_train_elaina.py#L741-L777)  
**Gejala:** Model dimuat dalam kuantisasi 4-bit (`load_in_4bit=True`, `BitsAndBytesConfig`), menghasilkan adapter QLoRA bukan LoRA bf16 murni.  
**Penyebab:** Skrip saat ini dirancang untuk alur hemat VRAM (~20 GB) dengan mirror bnb-4bit, mengabaikan ketersediaan VRAM 96 GB Blackwell yang mampu menampung base 31B bf16 (~62,5 GB). Merge dari model 4-bit juga bersifat lossy.

### Perbaikan:
Ubah konfigurasi dan logika pemuatan model agar memuat bobot dalam presisi `bfloat16` penuh tanpa kuantisasi:

```diff
# Di CFG (Cell 1):
-        "LOAD_IN_4BIT": True,
+        "TORCH_DTYPE": "bfloat16",
+        "LOAD_IN_4BIT": False,
```

```diff
# Di Cell 12 (Load base model):
-        try:
-            from unsloth import FastModel as _FastModel
-            base_model, _ = _FastModel.from_pretrained(
-                model_name=_source,
-                max_seq_length=CFG["MAX_SEQ"],
-                load_in_4bit=True,
-                full_finetuning=False,
-            )
-        except Exception as exc:
+        # Jika menggunakan Unsloth (pastikan mendukung Gemma 4 VLM bf16):
+        try:
+            from unsloth import FastModel as _FastModel
+            base_model, _ = _FastModel.from_pretrained(
+                model_name=model_info["id"],
+                max_seq_length=CFG["MAX_SEQ"],
+                load_in_4bit=False,
+                torch_dtype=_torch.bfloat16,
+            )
+        except Exception as exc:
+            # Fallback native transformers (AutoModelForMultimodalLM / Gemma4ForConditionalGeneration)
+            from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM
+            base_model = AutoModelForImageTextToText.from_pretrained(
+                model_info["id"],
+                torch_dtype=_torch.bfloat16,
+                device_map="auto",
+            )
```

**Kriteria Selesai:**
Eksekusi verifikasi di notebook menghasilkan:
```python
assert next(base_model.parameters()).dtype == torch.bfloat16
assert not hasattr(base_model, "is_loaded_in_4bit") or not base_model.is_loaded_in_4bit
```

---

## C4 — Target LoRA Bisa Mengenai Vision Tower

**Severity:** 🔴 Kritis · **[VERIFIKASI DULU: V2]**  
**Lokasi:** [`molab_train_elaina.py` L62-L63](../../molab_train_elaina.py#L62-L63), [L825-L844](../../molab_train_elaina.py#L825-L844)  
**Gejala:** Kemampuan pemrosesan gambar dan penalaran visual Gemma 4 mengalami degradasi atau rusak (catastrophic forgetting visual) setelah fine-tuning teks.  
**Penyebab:** 
1. `TARGET_MODULES` berisi daftar nama sederhana: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. PEFT mencocokkan nama-nama ini dengan *suffix matching*. Jika modul di vision encoder (misal SigLIP/ViT) atau projector menggunakan nama layer yang sama, modul visual akan ikut disuntik adapter LoRA dan ikut ter-update.
2. Unsloth `FastModel.get_peft_model` pada arsitektur VLM secara default sering kali menargetkan linear layer di seluruh arsitektur termasuk multimodal.

### Snippet Verifikasi V2:
Jalankan inspeksi hierarki nama modul model Gemma 4:
```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-31B-it")
print("Arsitektur:", cfg.architectures)
# Muat meta model atau inspeksi nama layer jika model ada:
for name, _ in base_model.named_modules():
    if any(k in name for k in ["vision", "visual", "projector", "image"]):
        print("Visual module:", name)
```

### Perbaikan:
Gunakan regex eksplisit pada `target_modules` yang hanya mencocokkan transformer decoder language model dan membekukan seluruh modul visual:

```diff
# Di CFG:
-        "TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj",
-                           "gate_proj", "up_proj", "down_proj"],
+        # Regex hanya menargetkan language model decoder layer:
+        "TARGET_MODULES": r".*(?:language_model|model)\.layers\.\d+\.(?:self_attn|mlp)\.(?:q|k|v|o|gate|up|down)_proj.*",
```

Tambahkan pembekuan manual (freeze) untuk seluruh parameter non-language model sebelum adapter di-attach:
```python
# Bekukan modul vision dan projector secara eksplisit:
for name, param in base_model.named_parameters():
    if any(k in name.lower() for k in ["vision", "visual", "img", "projector", "mm_"]):
        param.requires_grad = False
```

Tambahkan pengecekan pasca-attach LoRA:
```python
# Verifikasi tidak ada adapter LoRA yang menempel pada komponen visual:
visual_lora_params = [
    n for n, p in model.named_parameters() 
    if p.requires_grad and any(k in n.lower() for k in ["vision", "visual", "projector", "image"])
]
if visual_lora_params:
    raise RuntimeError(f"FATAL: LoRA aktif di modul visual: {visual_lora_params[:5]}")
```

**Kriteria Selesai:**
Daftar `visual_lora_params` kosong (`[]`), dan rasio bobot trainable berada di kisaran 0.1% – 1.0% dari total parameter 31B.

---

## C5 — Fallback Diam-diam ke fp16

**Severity:** 🔴 Kritis  
**Lokasi:** [`molab_train_elaina.py` L950-L962](../../molab_train_elaina.py#L950-L962)  
**Gejala:** Training crash dengan error gradien NaN, underflow/overflow numerik, atau error PyTorch `Attempting to unscale FP16 gradients`.  
**Penyebab:** Nilai `_bf16 = bool(gate["gpu"].get("bf16"))` diambil dari hasil cell `gate` yang dijalankan di awal notebook sebelum sesi PyTorch cu128 terpasang atau terinisialisasi. Akibatnya `_bf16` bernilai `False`, sehingga `TrainingArguments` otomatis menyalakan `fp16=True`. Model Gemma sangat sensitif terhadap fp16 dan wajib menggunakan bfloat16.

### Perbaikan:
Evaluasi dukungan bfloat16 langsung dari runtime PyTorch aktif saat membangun `TrainingArguments`:

```diff
-        _bf16 = bool(gate["gpu"].get("bf16"))
+        import torch as _torch
+        _bf16 = _torch.cuda.is_available() and _torch.cuda.is_bf16_supported()
+        if not _bf16:
+            raise RuntimeError("CRITICAL: GPU tidak mendukung bfloat16. Gemma 4 tidak dapat di-train secara stabil pada FP16!")
         _common = dict(
             ...
             bf16=_bf16,
-            fp16=not _bf16,
+            fp16=False,
```

**Kriteria Selesai:**
Objek `trainer.args.bf16` bernilai `True` dan `trainer.args.fp16` bernilai `False`.

---

## H10 — Penanganan `token_type_ids` / Multimodal Input pada Gemma VLM

**Severity:** 🟠 Tinggi · **[VERIFIKASI DULU: V5]**  
**Lokasi:** [`elaina_train_lib.py` L344-L365](../../elaina_train_lib.py#L344-L365) (`ResponseOnlyCollator`)  
**Gejala:** `Trainer.train()` crash dengan `TypeError: forward() got an unexpected keyword argument` atau error terkait masking multimodal embeddings.  
**Penyebab:** Gemma 4 VLM berbasis arsitektur pemrosesan teks dan citra terintegrasi. Beberapa varian processor menyertakan `token_type_ids` atau penanda posisi khusus multimodal, sementara collator custom saat ini hanya mereturn `input_ids`, `labels`, dan `attention_mask`.

### Perbaikan:
Jalankan uji dummy forward pass (V5):
```python
# Uji tokenisasi batch teks murni ke model
inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
out = model(**inputs)  # pastikan tidak melempar pengecualian token_type_ids
```
Bila model memerlukan kolom ekstra atau `remove_unused_columns=False` bermasalah, sesuaikan `ResponseOnlyCollator` agar meneruskan kwargs yang valid atau biarkan processor menangani collation dengan masking yang disuntikkan.

---

## M10 — Konfigurasi Optimizer & Gradient Checkpointing

**Severity:** 🟡 Sedang  
**Lokasi:** [`molab_train_elaina.py` L963](../../molab_train_elaina.py#L963), [L830-L832](../../molab_train_elaina.py#L830-L832)  
**Gejala:** OOM (Out Of Memory) saat backward pass atau warning ketidakcocokan kernel bitsandbytes pada arsitektur Blackwell sm_120.  
**Penyebab:** Skrip memakai `optim="adamw_8bit"`. Pada arsitektur GPU baru (Blackwell), binary CUDA kernel bitsandbytes 8-bit belum tentu teroptimasi atau kompatibel. Di sisi lain, pada 96 GB VRAM, training LoRA bf16 murni (hanya parameter adapter yang butuh optimizer states) sangat hemat memori sehingga optimizer AdamW native bf16 atau `adamw_torch_fused` dapat digunakan dengan aman.

### Perbaikan:
```diff
         _common = dict(
             ...
-            optim="adamw_8bit",
+            optim="adamw_torch_fused" if _torch.cuda.is_available() else "adamw_torch",
```
Dan pastikan gradient checkpointing selalu aktif:
```python
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

---

## M14 — Pemilihan Checkpoint Hanya Berdasarkan Eval Loss

**Severity:** 🟡 Sedang  
**Lokasi:** [`molab_train_elaina.py` L968-L971](../../molab_train_elaina.py#L968-L971)  
**Gejala:** Model overfitted terpilih secara otomatis karena eval loss pada novel terlihat bagus, namun gaya bicara persona dan fleksibilitas inferensi menjadi kaku.  
**Penyebab:** Sesuai peringatan di Q&A poin 14 dan 20: loss validasi pada domain novel sering menipu (base model sudah melihat teks novel saat pre-training). `load_best_model_at_end=True` dengan `metric_for_best_model="eval_loss"` dan `save_total_limit=1` menghapus checkpoint lain dan memaksa checkpoint terakhir/loss terendah.

### Perbaikan:
Simpan beberapa checkpoint berkala (misal 3 checkpoint):
```diff
-            save_total_limit=1,
+            save_total_limit=3,
-            load_best_model_at_end=True,
+            load_best_model_at_end=False,
```
Evaluasi output persona menggunakan set benchmark prompt tetap (lihat [06-merge-gguf-evaluasi.md](06-merge-gguf-evaluasi.md)) pada setiap checkpoint sebelum memutuskan adapter yang akan di-merge.

---

## M15 — Anggaran VRAM 31B bf16 pada 96 GB GPU Blackwell

**Severity:** 🟡 Sedang  
**Analisis Memori (Budgeting):**
Untuk model 31.3 miliar parameter pada presisi bf16:
1. **Bobot Statis Base Model:** 31.3B × 2 byte = ~62.6 GB VRAM.
2. **Bobot LoRA Adapter (r=32):** ~100–200 MB.
3. **Optimizer States (AdamW untuk LoRA saja):** ~400–800 MB.
4. **Gradien (hanya adapter):** ~200 MB.
5. **Aktivasi (Forward/Backward):**
   - Dengan context length 2048, batch size 2, gradient accumulation 4: ~4–6 GB (dengan activation/gradient checkpointing aktif).
   - *Catatan Penting:* Vocab size Gemma mencapai ~262.144 token. Layer `lm_head` logits menghasilkan tensor `[batch_size * seq_len, 262144]`. Pada bf16, logits untuk 4096 token membutuhkan ~2 GB VRAM saat penghitungan CrossEntropyLoss.
6. **Total Kebutuhan Puncak:** ~62.6 + 1.2 + 6 + 2 = **~71.8 GB VRAM**.
7. **Sisa Margin Bebas pada 96 GB:** ~24.2 GB headroom aman!

Rekomendasi Hyperparameter Aman:
- `per_device_train_batch_size`: **2** (efektif batch 16 dengan `gradient_accumulation_steps=8`)
- `max_seq_length`: **2048** (atau 4096 jika prompt panjang dipertahankan)
- `gradient_checkpointing`: **Wajib aktif**

