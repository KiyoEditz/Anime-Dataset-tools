# 05 — Environment Gate, Versi Library & Run Manifest

Kembali ke [dokumen utama](README.md).

Dokumen ini mencakup perbaikan pada mekanisme pengukuran sistem (*environment gate*), parsing versi PyTorch/CUDA untuk GPU Blackwell, optimasi transfer artefak, serta pencatatan audit *reproducibility* (run manifest).

---

## M1 — Parsing Versi PyTorch & CUDA yang Rapuh

**Severity:** 🟡 Sedang  
**Lokasi:** [`elaina_train_lib.py` L184-L194](../../elaina_train_lib.py#L184-L194)  
**Gejala:** Muncul peringatan palsu (*false warning*) bahwa kernel PyTorch tidak tersedia pada sistem yang sebenarnya sudah kompatibel.  
**Penyebab:**
Kode saat ini memeriksa versi dengan cara berikut:
```python
torch_ok = int(str(torch_v).split(".")[0]) >= 2 and int(str(torch_v).split(".")[1]) >= 7
cuda_ok = str(cuda_v).startswith("12.8") or str(cuda_v).startswith("12.9")
```
Kelemahan logika ini:
1. Jika PyTorch merilis versi mayor `3.0.0`, `split(".")[0] >= 2` bernilai benar, tetapi `split(".")[1] >= 7` (`0 >= 7`) bernilai **False**, sehingga dianggap tidak kompatibel.
2. Jika CUDA Toolkit versi `13.0` digunakan, `startswith("12.8")` bernilai **False**.
3. Jika versi PyTorch mengandung postfix nightly (misal `2.8.0.dev20260501`), parsing string integer langsung rentan melempar pengecualian.

### Perbaikan:
Gunakan pustaka parsing versi standar (`packaging.version.parse`) atau tuple perbandingan numerik yang kokoh:
```python
def check_torch_cuda_compat(torch_v: str, cuda_v: str) -> bool:
    try:
        from packaging import version
        t_ver = version.parse(torch_v.split("+")[0])
        c_ver = version.parse(cuda_v)
        # Butuh torch >= 2.7 dan cuda >= 12.8 untuk Blackwell sm_120
        return t_ver >= version.parse("2.7.0") and c_ver >= version.parse("12.8")
    except Exception:
        # Fallback tanpa packaging
        parts = [int(p) for p in torch_v.split("+")[0].split(".")[:2] if p.isdigit()]
        if len(parts) >= 2:
            major, minor = parts[0], parts[1]
            torch_ok = (major > 2) or (major == 2 and minor >= 7)
        else:
            torch_ok = False
        cuda_ok = "12.8" in cuda_v or "12.9" in cuda_v or "13." in cuda_v
        return torch_ok and cuda_ok
```

---

## M2 — Gate Disk Ikut Menghitung `/tmp`

**Severity:** 🟡 Sedang  
**Lokasi:** [`molab_train_elaina.py` L190-L193](../../molab_train_elaina.py#L190-L193)  
**Gejala:** Gate melaporkan status `tier: blocked` padahal disk kerja utama (`/root/work` atau mount volume Molab) memiliki ruang bebas 150–200 GB.  
**Penyebab:**
Cell 4 mengumpulkan path berikut:
```python
_paths = [CFG["WORKDIR"], os.path.expanduser("~"), "/tmp"]
```
Lalu `lib.env_gate` mengambil `min_free_gib(report)` (kapasitas tersempit). Pada banyak container cloud dan sistem Linux, `/tmp` dipasang sebagai tmpfs berbasis RAM dengan ukuran terbatas (misal 5–10 GB). Akibatnya, kapasitas tersempit jatuh pada `/tmp` dan memblokir notebook secara keliru.

### Perbaikan:
Hanya ukur path tempat artefak besar benar-benar ditulis, atau bedakan antara partisi kerja utama dan partisi temporary:
```diff
-    _paths = [CFG["WORKDIR"], os.path.expanduser("~"), "/tmp"]
+    _paths = [CFG["WORKDIR"]]
+    if os.environ.get("HF_HOME"):
+        _paths.append(os.environ["HF_HOME"])
+    # Ukur /tmp hanya sebagai informasi tambahan, bukan penentu kuota blokir
```

---

## M3 — `HF_HUB_ENABLE_HF_TRANSFER=1` Diaktifkan Sebelum Terpasang

**Severity:** 🟡 Sedang  
**Lokasi:** [`elaina_train_lib.py` L199](../../elaina_train_lib.py#L199)  
**Gejala:** Warning atau error saat mengunduh/mengunggah file besar lewat huggingface_hub jika modul `hf_transfer` belum terpasang.  
**Perbaikan:**
Aktifkan variabel lingkungan transfer hanya jika package `hf_transfer` benar-benar terdeteksi:
```python
import importlib.util
if importlib.util.find_spec("hf_transfer") is not None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
```

---

## M5 — Model ID Fallback & Inkonsistensi Ambang Tier

**Severity:** 🟡 Sedang · **[VERIFIKASI DULU: V4]**  
**Lokasi:** [`molab_train_elaina.py` L57](../../molab_train_elaina.py#L57), [`elaina_train_lib.py` L125](../../elaina_train_lib.py#L125), [`README.md` L122](../../README.md#L122)  
**Gejala:** Pilihan fallback model 12B mengarah ke model base non-instruksi atau error repo not found. Terdapat perbedaan batas VRAM yang membingungkan antara README (24 GB) dan kode (20 GB).

### Perbaikan:
1. Pastikan model fallback menggunakan varian instruction-tuned (`-it`):
   `google/gemma-4-12B-it` (bukan `google/gemma-4-12B`).
2. Sinkronkan aturan batas VRAM tier:
   ```python
   TIER_RULES = [
       ("full31b", 150.0, 80.0),
       ("small31b", 90.0, 80.0),
       ("12b", 45.0, 24.0), # seragamkan pada 24.0 GB
   ]
   ```

---

## M12 — Run Manifest Belum Menyimpan Informasi Lengkap

**Severity:** 🟡 Sedang  
**Lokasi:** [`molab_train_elaina.py` L1078-L1083](../../molab_train_elaina.py#L1078-L1083)  
**Gejala:** Hasil training tidak dapat direproduksi secara persis di masa depan karena versi pustaka, seed, hash dataset, dan konfigurasi adapter tidak tersimpan (Q&A poin 15).

### Perbaikan:
Lengkapi `write_run_manifest` agar mencatat seluruh metadata penting:
```python
lib.write_run_manifest(
    CFG["OUT_DIR"] / "training_run_manifest.json",
    base_model=CFG["MODEL_ID"],
    adapter_repo=repos["adapter"],
    dataset_repo=repos["dataset"],
    dataset_sha256=getattr(data_report, "manifest_sha256", None),
    hyperparameters={
        "lora_r": CFG["LORA_R"],
        "lora_alpha": CFG["LORA_ALPHA"],
        "lora_dropout": CFG["LORA_DROPOUT"],
        "target_modules": CFG["TARGET_MODULES"],
        "learning_rate": CFG["LR"],
        "batch_size": CFG["BATCH"],
        "gradient_accumulation_steps": CFG["GRAD_ACCUM"],
        "max_seq_length": CFG["MAX_SEQ"],
        "epochs": CFG["EPOCHS"],
        "seed": CFG["SEED"],
    },
    environment={
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "peft_version": getattr(peft, "__version__", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    training_results=train_report,
)
```

