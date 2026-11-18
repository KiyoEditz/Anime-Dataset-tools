# 08 — Panduan Merge, Ekspor GGUF & Inferensi

Dokumen ini menjelaskan alur penggabungan (*merge*) adapter LoRA ke bobot dasar Gemma 4, konversi ke format GGUF untuk eksekusi lokal berkecepatan tinggi pada `llama.cpp` atau Ollama, serta cara mengonfigurasi system prompt saat inferensi.

---

## 1. Konsep Penggabungan Bobot (*Weight Merge*)

Selama masa pelatihan, bobot asli model dasar ($W_0$) dibekukan dan pembaruan disimpan dalam dua matriks berdimensi rendah ($A$ dan $B$). Pada tahap produksi atau deployment:
$$W_{merged} = W_0 + \frac{\alpha}{r} (B \times A)$$

Keuntungan penggabungan bobot:
* **Zero Inference Latency:** Tidak ada overhead pemrosesan lapisan adapter tambahan saat runtime.
* **Kompatibilitas Penuh:** Model hasil merge berupa folder model Hugging Face standar (`safetensors`) yang kompatibel langsung dengan vLLM, TGI, dan skrip konversi `llama.cpp`.

---

## 2. Konversi GGUF & Multimodal Projector (`gguf_convert.py`)

Skrip [`training/gguf_convert.py`](../../gguf_convert.py) mengotomatisasi kloning `llama.cpp`, kompilasi biner `llama-quantize`, konversi intermediate, serta pengunggahan ke Hugging Face.

### Menjalankan Konversi Mandiri:
```bash
python training/gguf_convert.py \
    --merged /root/work/merged_dl \
    --out /root/work/gguf \
    --quants Q4_K_M,Q6_K \
    --work /root/work/llama.cpp \
    --repo username/elaina-gemma4-31b-GGUF \
    --mmproj \
    --revision b4800
```

### Parameter Utama:
| Flag | Fungsi |
|---|---|
| `--merged` | Direktori lokal berisi model Hugging Face safetensors hasil merge. |
| `--out` | Direktori tempat file `.gguf` akan disimpan. |
| `--quants` | Daftar format kuantisasi yang diinginkan (dipisahkan koma). |
| `--mmproj` | **Wajib disertakan.** Mengekstrak dan mengonversi vision projector (`mmproj-F16.gguf`) agar kemampuan membaca gambar aktif pada `llama.cpp`. |
| `--work` | Lokasi kompilasi kode sumber `llama.cpp`. |
| `--revision` | Pin commit/tag rilis `llama.cpp` untuk kestabilan kompilasi. |

### Panduan Memilih Format Kuantisasi (31B):
* **`Q4_K_M` (~18,8 GB VRAM):** Pilihan paling populer. Keseimbangan terbaik antara penghematan RAM/VRAM GPU desktop dan pelestarian perplexity model.
* **`Q6_K` (~24,5 GB VRAM):** Pilihan kualitas sangat tinggi. Performa bahasa hampir identik dengan model 16-bit asli.
* **`Q2_K` / `Q3_K_M`:** Varian kompresi agresif untuk perangkat dengan memori terbatas (<16 GB RAM).

---

## 3. Menjalankan Server Inferensi Lokal (`llama-server`)

Setelah file GGUF dan `mmproj` tersedia di repositori Hugging Face, Anda dapat menjalankannya langsung di mesin lokal atau server inferensi:

```bash
# Menjalankan server inferensi OpenAI-compatible dengan dukungan teks & gambar
llama-server \
    -hf username/elaina-gemma4-31b-GGUF \
    --hf-file elaina-Q4_K_M.gguf \
    --mmproj mmproj-F16.gguf \
    -c 4096 \
    --port 8080
```

Sekarang Anda dapat mengirimkan request via cURL, Chatbox, SillyTavern, atau Open-WebUI ke `http://localhost:8080/v1/chat/completions`.

---

## 4. Format Prompt Saat Inferensi

Pelatihan SFT menggunakan **short system prompt** (~580 karakter) untuk membentuk fondasi gaya bicara, intonasi, dan pola interaksi Elaina. 

Pada saat inferensi aplikasi akhir, Anda disarankan menggabungkan short prompt dengan ringkasan lore novel panjang untuk memberikan kekayaan konteks pengetahuan dunia:

```python
# Format System Message Inferensi Rekomendasi:
short_prompt = open("training/elaina_system_prompt_short.txt", encoding="utf-8").read()
long_lore = open("output/elaina_system_prompt.txt", encoding="utf-8").read()

system_message = f"{short_prompt}\n\n[LORE CONTEXT & BACKGROUND]\n{long_lore}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "Selamat pagi, Nona Penyihir. Apa rencanamu hari ini?"}
]
```
Format kombinasi ini terbukti paling efektif menjaga nada bicara asli Elaina (*"Dear reader"*, *"Ugh."*, rasa percaya diri tinggi) sembari tetap memiliki akses detail ensiklopedia mengenai karakter dan kota-kota di *Wandering Witch*.

