# 06 — Analisis Chat Template Gemma 4 & Pembuktian Masking

Dokumen ini menjelaskan struktur teknis chat template pada keluarga model Gemma, perlakuan role `system`, mekanisme introspeksi dinamis template oleh `elaina_train_lib.py`, serta cara memverifikasi *loss masking proof*.

---

## 1. Evolusi Chat Template Gemma Antar-Generasi

Setiap generasi model Gemma memiliki penanganan format percakapan yang berbeda pada Jinja template-nya:

| Karakteristik | Gemma 1 / Gemma 2 | Gemma 3 | Gemma 4 |
|---|---|---|---|
| **Role `system`** | **Dilarang** (Jinja melempar template error) | Didukung semu (digabung otomatis ke giliran user) | **Didukung Natively** via `<\|turn>system` |
| **Token Pembuka Giliran** | `<start_of_turn>{role}\n` | `<start_of_turn>{role}\n` | `<\|turn>{role}\n` (`sot_token="<\|turn>"`) |
| **Token Penutup Giliran** | `<end_of_turn>\n` | `<end_of_turn>\n` | `<turn\|>\n` (`eot_token="<turn\|>"`) |
| **Modality Input** | Teks murni | Multimodal VLM | Multimodal (Vision, Audio, Text) |
| **Kanal Pemikiran (*Thinking*)** | Tidak ada | Tidak ada | `<\|channel>thought\n<channel\|>` (disuntikkan saat `add_generation_prompt=True`) |

### Format Standar Gemma 4 (Saat Inferensi / `add_generation_prompt=True` + Respons):
```
<bos><|turn>system
You are Elaina, the Ashen Witch...<turn|>
<|turn>user
Siapa namamu?<turn|>
<|turn>model
<|channel>thought
<channel|>Nama saya Elaina, seorang penyihir pengelana.<turn|>
```

---

## 2. Mengapa Template Tidak Boleh Di-hardcode?

Banyak skrip fine-tuning pemula melakukan pemotongan string secara manual, misalnya:
```python
# CARA YANG SALAH:
prompt, response = full_text.split("<start_of_turn>model\n")
```
Pendekatan ini sangat rapuh karena:
1. Spasi, karakter baris baru (`\n`), atau token tersembunyi dapat berbeda antar versi tokenizer.
2. Gemma 4 VLM menggunakan `AutoProcessor` yang dapat menyuntikkan token penanda posisi gambar atau kanal kontrol reasoning.
3. String splitting pada teks mentah mengabaikan boundary tokenization (*subword token merging*), yang dapat menyebabkan label bergeser 1–2 token dari posisi sebenarnya.

---

## 3. Cara Kerja `derive_turn_parts()`

Di [`training/elaina_train_lib.py`](../../elaina_train_lib.py), fungsi `derive_turn_parts()` melakukan introspeksi dinamis langsung terhadap fungsi `apply_chat_template` tokenizer aktif:

1. **Injeksi Probe Marker:** Fungsi mengirimkan data percakapan sintetis dengan penanda unik yang mustahil ada di korpus asli:
   ```python
   probe = [
       {"role": "user", "content": "ZXQ_USER_MARKER_7F3A"},
       {"role": "assistant", "content": "ZXQ_ASSISTANT_MARKER_7F3A"},
   ]
   ```
2. **Pendeteksian Opener:** Mencari posisi indeks marker pada hasil render template untuk mengisolasi token pembuka instruksi dan token pembuka giliran model.
3. **Uji Stabilitas Prefix (*Prefix-Stability Test*):**
   * Memanggil template dengan `add_generation_prompt=False`.
   * Memanggil template dengan `add_generation_prompt=True`.
   * Memastikan bahwa hasil dengan prompt generasi merupakan ekstensi murni dari teks sebelumnya (`with_gen.startswith(plain)`). Jika tidak stabil, sistem menolak melakukan training karena masking token berisiko salah posisi.

---

## 4. Membaca Bukti Masking (*Chat Template Proof*)

Di **Cell 11** notebook `molab_train_elaina.py`, tabel verifikasi ditampilkan:

| Parameter Bukti | Nilai Ideal | Arti & Validasi |
|---|---|---|
| **`user-turn opener`** | `'<start_of_turn>user\n'` | Tag penanda giliran pengguna terdeteksi benar. |
| **`model-turn opener`** | `'<start_of_turn>model\n'` | Tag penanda asisten terdeteksi benar. |
| **`prompt tokens`** | (Contoh: `168`) | Jumlah token sistem + pengguna yang wajib di-mask. |
| **`supervised tokens`** | (Contoh: `35`) | Jumlah token jawaban Elaina yang dihitung loss-nya. |
| **`contiguous span`** | **`True`** | Token yang disupervisi tidak terputus di tengah kalimat. |
| **`span starts at prompt end`**| **`True`** | Indeks token pertama yang disupervisi berada persis setelah prompt penutup user turn. |

> **Perhatian:** Jika salah satu dari `contiguous span` atau `span starts at prompt end` bernilai `False`, fungsi penalti gradien akan melatih model untuk meniru pertanyaan user! Segera periksa format Jinja template tokenizer Anda.

