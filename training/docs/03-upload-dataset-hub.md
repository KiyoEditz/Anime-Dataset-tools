# 03 — Panduan Unggah Dataset ke Hugging Face Hub

Dokumen ini menjelaskan prosedur mendistribusikan dataset yang telah dikurasi dari mesin lokal (`training/out/`) ke Hugging Face Hub private repository menggunakan [`training/push_dataset_to_hub.py`](../../push_dataset_to_hub.py).

---

## 1. Mengapa Dataset Harus Diunggah ke Hub?

1. **Isolasi Lingkungan:** Lingkungan GPU cloud (Molab/Kaggle/HF Jobs) hanya membutuhkan file training yang bersih (`train.jsonl` dan `val.jsonl`). Pipeline ekstraksi ratusan bab novel tidak perlu dijalankan atau diunggah ke cloud.
2. **Integritas Data:** Skrip memastikan setiap file diverifikasi menggunakan checksum SHA256 ganda (diunggah, diunduh kembali secara independen, dan dicocokkan bit demi bit).
3. **Pemberian Versi (Versioning):** Memungkinkan pelacakan commit dataset yang pasti saat model di-fine-tune, sehingga hasil eksperimen dapat direproduksi (*reproducible*).

---

## 2. Prasyarat & Autentikasi

Pastikan token Hugging Face Anda memiliki izin **Write Role**:

1. Login via Hugging Face CLI:
   ```bash
   huggingface-cli login
   ```
   *Atau set environment variable:*
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```
2. Pastikan pustaka transfer cepat terpasang:
   ```bash
   pip install huggingface_hub hf_transfer
   ```

---

## 3. Cara Penggunaan CLI

```bash
# 1. Uji coba (Dry Run: menampilkan daftar file dan ukuran tanpa mengunggah)
python training/push_dataset_to_hub.py \
    --dir training/out \
    --repo username/elaina-gemma4-dataset \
    --dry-run

# 2. Eksekusi unggah riil (Secara default bersifat PRIVATE dan terverifikasi)
python training/push_dataset_to_hub.py \
    --dir training/out \
    --repo username/elaina-gemma4-dataset
```

### Parameter Skrip:
| Flag | Default | Deskripsi |
|---|---|---|
| `--dir` | `training/out` | Direktori artefak dataset lokal yang akan diunggah. |
| `--repo` | *(Wajib)* | ID repositori Hugging Face target (`<user>/<nama-repo>`). |
| `--private` | `True` | Membuat repositori berstatus **Private** (wajib untuk data berhak cipta). |
| `--verify` | `True` | Mengunduh kembali seluruh file ke folder temporer untuk memastikan hash cocok. |
| `--dry-run` | `False` | Menampilkan rencana upload tanpa menghubungi API Hugging Face. |

---

## 4. Proses Verifikasi Hash Otomatis

Skrip `push_dataset_to_hub.py` menjalankan alur ketat sebelum menyatakan proses berhasil:
1. Menghitung nilai SHA256 dari `train.jsonl`, `val.jsonl`, `manifest.json`, `stats.json`, `train_meta.jsonl`, dan `val_meta.jsonl`.
2. Menghasilkan file dokumentasi otomatis (`README.md` dataset card) yang memuat tabel ringkasan bahasa, jumlah baris, dan metode split.
3. Mengunggah seluruh artefak ke repositori target.
4. Membuat folder sementara (`tempfile.TemporaryDirectory`), lalu mengunduh ulang setiap file langsung dari Hugging Face Hub.
5. Membandingkan hash file lokal dengan hash file yang diunduh. Jika ditemukan perbedaan satu byte saja, skrip mengembalikan exit code `1` dan membatalkan status keberhasilan.

---

## 5. Menghubungkan Dataset ke Notebook Training

Setelah upload selesai, nama repositori dataset Anda (misal `username/elaina-gemma4-dataset`) dimasukkan pada **Cell 6** di notebook `molab_train_elaina.py`. Notebook akan menarik data langsung dari repositori tersebut.

