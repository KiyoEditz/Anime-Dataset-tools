# 02 — Panduan Persiapan Dataset (`prepare_dataset.py`)

Dokumen ini menjelaskan cara mengubah pasangan dialog hasil ekstraksi novel menjadi dataset SFT (*Supervised Fine-Tuning*) terstruktur dalam format pesan percakapan standar:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## 1. Mengapa Dataset Harus Dipersiapkan Ulang?

File kompilasi awal di `output/elaina_finetune_dataset.jsonl` memiliki kelemahan mendasar:
1. **Redundansi Prompt:** ~97% dari total ukuran file adalah pengulangan teks prompt panjang (~8,8 KB) di setiap baris.
2. **Ketiadaan Metadata Bab:** Kolom `chapter` dihilangkan, sehingga mustahil melakukan pembagian train dan validation yang bebas bocor (*leakage-free*).
3. **Pencampuran Tipe Giliran:** Giliran pengguna mencampurkan dialog percakapan langsung dengan narasi sudut pandang ketiga tanpa ada penanda pembeda.

Skrip `prepare_dataset.py` membaca langsung file mentah `output/raw_dialogue_pairs.jsonl` dan `output/synthetic_pairs.jsonl` untuk mempertahankan nama bab asli dan menyusun pembagian train/val yang valid secara ilmiah.

---

## 2. Struktur Input & Output

### Input yang Dibutuhkan:
* `output/raw_dialogue_pairs.jsonl`: Pasangan dialog hasil ekstraksi dari novel. Wajib memiliki field:
  * `user_message`: Perkataan lawan bicara atau narasi situasi.
  * `elaina_response`: Kalimat balasan atau monolog batin Elaina.
  * `chapter`: Nama file bab asal (misal `chapter_0015.txt`).
* `training/elaina_system_prompt_short.txt`: Instruksi persona inti ringkas (~580 karakter / ~140 token).
* `output/elaina_system_prompt.txt` (opsional): Prompt persona lengkap beserta latar belakang novel (~8,8 KB).
* `training/sources.json`: Konfigurasi slot bahasa yang aktif.

### Artefak Output (`training/out/`):
| File | Fungsi |
|---|---|
| `train.jsonl` | Kumpulan data latih format `{"messages": [...]}` untuk trainer TRL/Transformers. |
| `val.jsonl` | Kumpulan data evaluasi yang babnya **100% terpisah** dari data latih. |
| `train_meta.jsonl` | Metadata baris-per-baris data latih (`lang`, `chapter`, `kind`, `prompt`, `source`). |
| `val_meta.jsonl` | Metadata baris-per-baris data evaluasi. |
| `manifest.json` | Rekapitulasi jumlah baris, parameter build, daftar bab train vs val, dan checksum SHA256. |
| `stats.json` | Distribusi statistik panjang karakter, persentase jenis baris, dan estimasi token. |

---

## 3. Alur Pemrosesan Data

1. **Length Filtering:** Membuang baris jika panjang respons kurang dari 8 karakter (menghindari artefak ekstraksi kosong, namun tetap mempertahankan ekspresi khas seperti *"Shoot."*) atau lebih dari 1.500 karakter.
2. **Exact Deduplication:** Menghapus duplikasi eksak pasangan `(user, response)` dan duplikasi respons identik dalam lingkup bahasa yang sama.
3. **Fuzzy Deduplication:** Mengelompokkan respons ke dalam bucket 3 kata pertama dan membuang respons yang memiliki rasio kemiripan leksikal `SequenceMatcher` $\ge 92\%$.
4. **Scene Tagging:** Menganalisis kalimat giliran user. Jika user turn terdeteksi sebagai narasi orang ketiga (bukan percakapan langsung ber-tanda kutip), teks akan diberi awalan `[Scene] ` agar model di tahap inferensi dapat membedakan antara lawan bicara yang mengajak mengobrol vs deskripsi situasi.
5. **Leakage-Free Chapter Split:** Seluruh baris dikelompokkan berdasarkan bab novel. Bab novel diacak dan dipisahkan menjadi kelompok Train dan Val. Bab yang masuk Val tidak akan pernah muncul di Train.

---

## 4. Cara Penggunaan CLI

Jalankan perintah dari root repositori:

```bash
# 1. Preview saja (Dry Run: kompilasi di memori, uji seluruh asersi, tidak menulis file ke disk)
python training/prepare_dataset.py --sources-file training/sources.json --check

# 2. Eksekusi pembuatan file dataset riil
python training/prepare_dataset.py --sources-file training/sources.json
```

### Parameter Penting (Flags):
| Flag | Nilai Default | Keterangan |
|---|---|---|
| `--sources-file` | `None` | Path file JSON definisi sumber data bahasa. |
| `--val-frac` | `0.06` | Fraksi bab yang dialokasikan untuk evaluasi (~6%). |
| `--long-prompt-frac` | `0.10` | Proporsi baris train yang disuntik prompt panjang sebagai augmentasi variasi (disarankan `0.0` jika `MAX_SEQ=2048`). |
| `--fuzzy-threshold` | `92` | Ambang kemiripan string (0–100) untuk deduplikasi fuzzy. |
| `--min-response-chars` | `8` | Batas minimum panjang karakter respons asisten. |
| `--max-response-chars` | `1500` | Batas maksimum panjang karakter respons asisten. |
| `--drop-scene` | `False` | Jika aktif, baris narasi adegan `[Scene]` akan dibuang sepenuhnya. |

---

## 5. Menambahkan Data Bahasa Indonesia (`id`) & Jepang (`jp`)

Dataset dirancang untuk multi-bahasa tanpa perlu mengubah logika kode. Langkah menambahkannya:

1. Letakkan file hasil ekstraksi dialog pada folder drop-point:
   * Bahasa Indonesia: `training/data_id/raw_dialogue_pairs.jsonl`
   * Bahasa Jepang: `training/data_jp/raw_dialogue_pairs.jsonl`
2. Buka file [`training/sources.json`](../sources.json), lalu ubah status bahasa dari `"pending"` menjadi `"ready"`:
   ```json
   {
     "lang": "id",
     "status": "ready",
     "files": ["training/data_id/raw_dialogue_pairs.jsonl"],
     "target_rows": 3000
   }
   ```
3. Pastikan penamaan bab pada metadata JSONL menggunakan penomoran bab standar yang seragam (misal `chapter_0005.txt`) agar pembagian train/val lintas bahasa sinkron dan tidak mengalami kebocoran.
4. Jalankan kembali:
   ```bash
   python training/prepare_dataset.py --sources-file training/sources.json
   ```

