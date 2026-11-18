# 01 — Notebook marimo (`molab_train_elaina.py`)

Kembali ke [dokumen utama](README.md).

Latar belakang aturan marimo yang relevan:

- Setiap nama global (variabel, fungsi, **import**) hanya boleh didefinisikan di **satu** cell. Nama berawalan `_` bersifat lokal per cell.
- Output visual cell adalah **ekspresi terakhir** sebelum `return`. Panggilan `mo.md(...)` di tengah cell, atau di dalam `if`/`for`, **tidak ditampilkan**.
- `mo.ui.run_button().value` bernilai `True` hanya pada eksekusi yang dipicu klik. Saat cell dijalankan ulang karena upstream berubah, nilainya kembali `False`.
- Cell berjalan reaktif: begitu dependensinya berubah, cell langsung dijalankan ulang.

---

## C1 — Global marimo didefinisikan di lebih dari satu cell

**Severity:** 🔴 Kritis

**Lokasi** (hasil analisis AST):

| Nama | Cell yang mendefinisikan |
|---|---|
| `shutil` | cell 1 [L40](../../molab_train_elaina.py#L40), merge [L1291](../../molab_train_elaina.py#L1291), GGUF [L1387](../../molab_train_elaina.py#L1387) |
| `importlib` | cell 1 [L35](../../molab_train_elaina.py#L35), install [L371](../../molab_train_elaina.py#L371) (`import importlib.util` mengikat nama `importlib`) |
| `snapshot_download` | data [L511](../../molab_train_elaina.py#L511), train [L1054](../../molab_train_elaina.py#L1054), GGUF [L1369](../../molab_train_elaina.py#L1369) |
| `torch` | load [L764](../../molab_train_elaina.py#L764), samples [L1161](../../molab_train_elaina.py#L1161) |

**Gejala:** marimo menampilkan error *"This cell redefines variables from other cells"* dan tidak menjalankan cell yang bersangkutan, beserta semua cell turunannya.

**Penyebab:** import lokal di dalam cell dianggap sebagai definisi global.

**Perbaikan:**

```diff
 # cell 1
-    import importlib
+    import importlib
+    import importlib.util  # dipakai di cell install
```

```diff
 # cell install (L371)
-        import importlib.util
-
         unsloth_ready = importlib.util.find_spec("unsloth") is not None
```

```diff
 # cell data (L511), train (L1054), GGUF (L1369) — pakai alias lokal
-            from huggingface_hub import snapshot_download
+            from huggingface_hub import snapshot_download as _snapshot_download
 ...
-            _src = Path(snapshot_download(
+            _src = Path(_snapshot_download(
```

```diff
 # cell load (L764) dan samples (L1161)
-                import torch
+                import torch as _torch
 ...
-                        bnb_4bit_compute_dtype=torch.bfloat16,
+                        bnb_4bit_compute_dtype=_torch.bfloat16,   # (akan diganti oleh C3)
```

```diff
 # cell merge (L1279): tambahkan `shutil` ke parameter, lalu hapus import lokal
-def _(CFG, Path, drop_cache, hf_api, lib, merge_button, mo, model, model_info, repos, tokenizer):
+def _(CFG, Path, drop_cache, hf_api, lib, merge_button, mo, model, model_info, repos, shutil, tokenizer):
 ...
-                import shutil
 ...
-        import shutil as _sh
-
-        _sh.rmtree(_tmp, ignore_errors=True)
+        shutil.rmtree(_tmp, ignore_errors=True)
```

```diff
 # cell GGUF (L1362): sama, tambahkan `shutil` ke parameter dan hapus `import shutil` di L1387
```

**Kriteria selesai:** skrip berikut mencetak `{}`.

```python
import ast, collections
src = open("training/molab_train_elaina.py", encoding="utf-8").read()
defs = collections.defaultdict(list)
for node in ast.parse(src).body:
    if not (isinstance(node, ast.FunctionDef) and any(getattr(d, "attr", "") == "cell" for d in node.decorator_list)):
        continue
    names = set()
    stack = list(node.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name); continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    for nm in names:
        if not nm.startswith("_"):
            defs[nm].append(node.lineno)
print({k: v for k, v in defs.items() if len(v) > 1})
```

---

## C6 — Resume dari Hub tidak pernah menemukan checkpoint

**Severity:** 🔴 Kritis

**Lokasi:** [L973-L976](../../molab_train_elaina.py#L973-L976) (`hub_strategy="checkpoint"`) dan [L1052-L1068](../../molab_train_elaina.py#L1052-L1068) (resume).

**Gejala:** setelah sesi putus, tombol "Start / resume training" selalu mulai dari awal (`resumed_from: scratch`).

**Penyebab:** dengan `hub_strategy="checkpoint"`, `transformers` mengunggah checkpoint terakhir ke subfolder **`last-checkpoint/`**. Kode resume justru mencari `checkpoint-*/**` dan mengurutkan berdasarkan angka di nama folder.

**Perbaikan (pilih satu, rekomendasi A):**

*A. Simpan semua checkpoint di Hub* (sekaligus memperbaiki M14: checkpoint bisa dipilih berdasarkan sampel generasi):

```diff
-            save_total_limit=1,
+            save_total_limit=3,
 ...
-            hub_strategy="checkpoint",
+            hub_strategy="all_checkpoints",
```

Logika resume yang ada (`checkpoint-*`) sudah cocok dengan opsi ini. Perlu ditambah validasi bahwa folder benar-benar berisi `trainer_state.json`:

```diff
             _cks = sorted(
-                (p for p in _dir.rglob("checkpoint-*") if p.is_dir()),
+                (p for p in _dir.glob("checkpoint-*")
+                 if p.is_dir() and (p / "trainer_state.json").exists()),
                 key=lambda p: int(p.name.split("-")[-1]),
             )
```

*B. Tetap `checkpoint`:* ganti `allow_patterns="last-checkpoint/**"` dan `_resume = str(_dir / "last-checkpoint")` bila `trainer_state.json` ada.

**Catatan:** resume juga gagal kalau config LoRA, `MAX_SEQ`, atau dataset berbeda dari run sebelumnya. Bandingkan hash dataset di run manifest (M12) sebelum melanjutkan.

**Kriteria selesai:** di smoke test, latih 3 step, matikan kernel, lalu jalankan ulang. Laporan harus menampilkan `resumed_from: .../checkpoint-N` dan `global_step` dilanjutkan dari N.

---

## C8 — Trainer dibangun tanpa LoRA

**Severity:** 🔴 Kritis

**Lokasi:** cell LoRA [L817](../../molab_train_elaina.py#L817) (`model = base_model`) dan cell trainer [L947](../../molab_train_elaina.py#L947).

**Gejala:**
- QLoRA (kode saat ini): `ValueError` karena model terkuantisasi tidak boleh dilatih tanpa adapter.
- bf16 (setelah C3): **full fine-tuning 31B**, langsung OOM. Atau, bila lolos, bobot asli berubah, dan ini melanggar requirement "jangan override base model".

**Penyebab:** cell LoRA mengekspor `model = base_model` sebelum tombol ditekan. Cell trainer reaktif terhadap `model`, jadi trainer langsung dibangun begitu base model selesai dimuat.

**Perbaikan:**

```diff
 def _(CFG, base_model, lora_button, mo):
-    model = base_model
+    model = None
     lora_report = {}
```

```diff
-def _(CFG, gate, lib, mo, model, repos, tokenizer, train_ds, val_ds):
+def _(CFG, gate, lib, lora_report, mo, model, repos, tokenizer, train_ds, val_ds):
     trainer = None
     trainer_report = {}
 
-    if model is not None and train_ds is not None and val_ds is not None:
+    if (model is not None and lora_report.get("ok")
+            and train_ds is not None and val_ds is not None):
```

Tambahkan juga assert di cell trainer:

```python
_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
_total = sum(p.numel() for p in model.parameters())
mo.stop(_trainable / _total > 0.03, mo.md("Trainable > 3 % — kemungkinan full fine-tune. Berhenti."))
```

**Kriteria selesai:** setelah base model dimuat tapi tombol LoRA belum ditekan, cell trainer menampilkan *"Trainer not built yet"*.

---

## C9 — Output cell marimo tidak tampil

**Severity:** 🔴 Kritis (gate dan bukti masking, yang di README disebut *non-negotiable*, tidak terlihat oleh user)

**Lokasi:** hampir semua cell logika, contohnya:
- readme [L149-L181](../../molab_train_elaina.py#L149-L181): `mo.md` utama lalu `if SMOKE: mo.md(...)`. Ekspresi terakhirnya adalah `if`, sehingga **tidak ada** output.
- gate [L208-L233](../../molab_train_elaina.py#L208-L233): tabel gate, peringatan dalam `for`, dan pesan blokir dalam `if`. Tidak ada yang tampil.
- login, model resolve, data, tokenizer, proof, load, LoRA, tokenise, trainer, train, eval, adapter, merge, GGUF, cleanup: semuanya ditutup dengan blok `if/else`.

**Penyebab:** di marimo, `mo.md()` hanya membuat objek. Objek itu baru tampil kalau menjadi ekspresi terakhir cell, atau dikirim lewat `mo.output.append()` / `mo.output.replace()`.

**Perbaikan (pola seragam):**

```python
_out = []                       # di awal cell
...
_out.append(mo.md("..."))       # menggantikan setiap mo.md(...) yang berdiri sendiri
...
mo.vstack(_out)                 # ekspresi terakhir sebelum return
return (gate,)
```

Untuk proses panjang (install, train, GGUF) yang perlu progres langsung, gunakan `mo.output.append(mo.md(...))`.

**Kriteria selesai:** di Molab setiap cell menampilkan tabel atau pesannya. Uji khusus: set `ELAINA_WORKDIR` ke disk kecil, lalu pastikan pesan "ENV GATE: tidak bisa lanjut" muncul.

---

## H1 — Bukti masking dan validasi dataset tidak memblokir

**Severity:** 🟠 Tinggi

**Lokasi:** cell tokenise [L877-L883](../../molab_train_elaina.py#L877-L883), cell data [L599-L604](../../molab_train_elaina.py#L599-L604).

**Penyebab:** `_ready` hanya mengecek `"error" not in parts`. Nilai `masking_evidence["ok"]` (span supervisi kontigu dan dimulai tepat di batas prompt) dan `data_report["ok"]` (termasuk kebocoran bab) dihitung tapi tidak pernah dipakai.

**Perbaikan:**

```diff
-def _(CFG, lib, mo, parts, tokenizer, train_raw, val_raw):
+def _(CFG, data_report, lib, masking_evidence, mo, parts, tokenizer, train_raw, val_raw):
 ...
+    mo.stop(not data_report.get("ok"),
+            mo.md("**Dataset tidak valid** (lihat cell Dataset). Tokenisasi dihentikan."))
+    mo.stop(not (masking_evidence and masking_evidence.get("ok")),
+            mo.md("**Bukti masking gagal.** Jangan latih model; periksa cell Chat template proof."))
```

Tambahkan juga pemblokiran bila `data_report["chapter_check"] == "unverified"`, kecuali di mode smoke.

**Kriteria selesai:** memanipulasi `train_raw[0]` (misalnya menghapus role system) membuat tokenisasi berhenti dengan pesan yang jelas.

---

## H4 — Smoke test tidak menguji jalur Gemma 4

**Severity:** 🟠 Tinggi · **[VERIFIKASI DULU: V4]**

**Lokasi:** [L52-L56](../../molab_train_elaina.py#L52-L56).

**Penyebab:** `Qwen/Qwen3-0.6B` adalah model teks murni dengan chat template berbeda (ada blok `<think>`), tanpa vision tower, dan nama modulnya berbeda. Jadi smoke test **tidak** menguji C4 (vision beku), H7 (processor), H10 (`token_type_ids`), maupun template Gemma 4. Ini bertentangan dengan Q&A poin 13: dry run harus menguji alur lengkap.

**Perbaikan:**

```diff
         "MODEL_ID": (
-            os.environ.get("ELAINA_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
+            os.environ.get("ELAINA_SMOKE_MODEL", "google/gemma-4-E2B-it")   # varian Gemma 4 terkecil, cek V4
             if SMOKE
```

Pastikan juga mode smoke menjalankan **merge kecil** dan **satu konversi GGUF** (keduanya cepat untuk E2B), karena merge dan GGUF adalah langkah yang paling sering gagal. Saat ini `SKIP_MERGE: SMOKE` didefinisikan tapi tidak pernah dipakai.

**Kriteria selesai:** mode smoke lolos dari gate sampai GGUF memakai model Gemma 4, dalam waktu kurang dari 30 menit.

---

## H8 — Cell install tidak sesuai teksnya

**Severity:** 🟠 Tinggi · **[VERIFIKASI DULU: V6]**

**Lokasi:** [L329-L388](../../molab_train_elaina.py#L329-L388).

**Penyebab:**
- Teks menyebut torch cu128 dan transformers dari git, tapi kodenya tidak memasang keduanya.
- `pip install --upgrade unsloth unsloth_zoo` dapat menarik versi torch lain dari PyPI, yang bisa tidak punya kernel sm_120.
- Fallback git nightly memakai `--no-deps`, sehingga dependensinya tidak lengkap.
- `bitsandbytes` dipasang padahal setelah C3 tidak diperlukan, kecuali optimizer 8-bit (lihat M10).

**Perbaikan (urutan aman):**

```python
# 1. torch: hanya pasang/ganti bila kernel sm_120 tidak ada
import torch
need_torch = (torch.cuda.is_available()
              and torch.cuda.get_device_capability(0) >= (12, 0)
              and "sm_120" not in torch.cuda.get_arch_list())
if need_torch:
    _pip("install", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cu128",
         "torch", "torchvision")

# 2. transformers yang mengenal gemma4 (cek dulu apakah rilis stabil sudah cukup)
_pip("install", "--upgrade", "transformers>=<versi-yang-mendukung-gemma4>")  # atau git+...@main

# 3. sisanya, tanpa membiarkan pip mengganti torch
_pip("install", "--upgrade", "peft", "trl", "accelerate", "datasets", "hf_transfer", "psutil")
_pip("install", "--upgrade", "unsloth", "unsloth_zoo")
# 4. cek ulang torch setelah semua install; bila berubah -> tampilkan peringatan restart
```

Setelah install, tampilkan `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_arch_list()`, `transformers.__version__`, `peft.__version__`, `trl.__version__`, `unsloth.__version__`.

**Snippet V6:**

```python
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print((x @ x).float().abs().mean())
```

**Kriteria selesai:** `sm_120` ada di arch list (atau arch list mencakup kapabilitas GPU), matmul bf16 berjalan, dan `AutoConfig.from_pretrained("google/gemma-4-31B-it")` tidak melempar `KeyError: 'gemma4'`.

---

## L1–L3, L6 — Item rendah

| ID | Lokasi | Masalah | Perbaikan |
|---|---|---|---|
| L1 | [L591-L597](../../molab_train_elaina.py#L591-L597) | cek urutan role hanya `train_raw[:200]`, val tidak dicek | cek seluruh train dan val (cepat, karena hanya list comprehension) |
| L2 | [L674-L675](../../molab_train_elaina.py#L674-L675) | `masking_report(..., 512, ...)` tetap | pakai `CFG["MAX_SEQ"]`, lalu jalankan bukti pada baris prompt-pendek **dan** prompt-panjang (cari baris pertama dengan system > 2000 karakter) |
| L3 | [L1174-L1178](../../molab_train_elaina.py#L1174-L1178) | sampel acak tanpa seed, sehingga base vs LoRA tidak sebanding | `torch.manual_seed(CFG["SEED"])` sebelum setiap generate, atau gunakan greedy untuk perbandingan |
| L6 | [L25](../../molab_train_elaina.py#L25), [L959](../../molab_train_elaina.py#L959) | `__generated_with = "0.9.25"` usang, dan `warmup_ratio` mungkin deprecated di transformers v5 | perbarui sesuai versi marimo Molab, lalu periksa peringatan deprecation saat `TrainingArguments` dibuat |

