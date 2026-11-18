# 04 — Metode Training & Arsitektur (LoRA bf16)

Dokumen ini menjelaskan fondasi teoritis dan teknis dari metode **LoRA (Low-Rank Adaptation) bfloat16** yang diterapkan pada arsitektur multimodal **Gemma 4 31B** untuk pembentukan persona karakter Elaina.

---

## 1. Mengapa LoRA bf16 (Bukan QLoRA 4-bit)?

Pada mesin dengan spesifikasi VRAM terbatas (misalnya 16–24 GB VRAM), teknik QLoRA (kuantisasi 4-bit normal float) adalah keharusan. Namun, pada GPU **NVIDIA Blackwell 96 GB VRAM**, kuantisasi 4-bit memiliki kerugian yang tidak perlu:
1. **Kehilangan Presisi Bobot (Quantization Noise):** Matriks bobot yang dipadatkan ke 4-bit mengalami kuantisasi lossy, yang dapat mengurangi ketajaman nuansa bahasa, sarkasme halus, dan kepribadian karakter.
2. **Kompleksitas Merge yang Menurunkan Kualitas:** Menggabungkan adapter LoRA ke model 4-bit untuk diekspor ke format 16-bit atau GGUF melibatkan proses de-kuantisasi perkiraan yang sering kali menghasilkan artefak numerik.
3. **Kapasitas 96 GB Sangat Memadai:** Base model Gemma 4 31B dalam format `bfloat16` murni hanya mengonsumsi **~62,5 GB VRAM**. Terdapat sisa **~33,5 GB VRAM** yang sangat longgar untuk menampung bobot adapter LoRA, optimizer states, dan aktivasi.

---

## 2. Strategi Multimodal: Pembekuan Vision Tower

Gemma 4 adalah arsitektur Vision-Language Model (VLM). Model ini memiliki encoder visual (vision tower) dan modul proyeksi multimodal (*projector*) yang menghubungkan fitur gambar ke lapisan bahasa.

### Risiko Unfrozen Vision:
Jika adapter LoRA ditargetkan secara umum (`all-linear`), layer-layer di dalam vision encoder akan ikut disuntikkan adapter dan mengalami perubahan gradien selama pembelajaran teks dialog novel. Akibatnya:
* Terjadi **catastrophic forgetting visual**: model kehilangan kemampuan memahami gambar umum, grafik, dan objek visual yang telah dipelajari saat pre-training Google.
* Pembengkakan memori komputasi gradien yang sia-sia.

### Kebijakan Pembekuan:
* Parameter Vision Tower dan Multimodal Projector di-set secara tegas ke `requires_grad = False`.
* Target modul LoRA dibatasi hanya pada transformer decoder language model (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` pada `language_model.model.layers`).
* Rasio bobot yang dilatih (*trainable parameters*) dijaga pada kisaran **0,1% – 1,0%** dari total 31,3 miliar parameter.

---

## 3. Response-Only Loss Masking

Pada pelatihan dialog percakapan instruksi/gaya, model **tidak boleh** dilatih untuk memprediksi kata-kata dari pertanyaan user atau system prompt. Jika loss dihitung pada seluruh teks:
* Model cenderung menghafal gaya bicara pengguna atau mengulangi kembali pertanyaan pengguna (echoing).
* Gradien terdistorsi oleh variasi pertanyaan yang acak.

### Mekanisme Masking:
Target label tensor (`labels`) diisi dengan nilai `-100` pada seluruh rentang token system prompt dan user message. Fungsi rugi PyTorch (`CrossEntropyLoss(ignore_index=-100)`) secara otomatis mengabaikan token-token tersebut dalam kalkulasi loss:
```
Token:   [<bos>] [<start_of_turn>] user \n Siapa kamu? [<end_of_turn>] \n [<start_of_turn>] model \n Saya Elaina. [<end_of_turn>] \n
Labels:   -100        -100         -100   -100  -100       -100       -100     -100      -100  -100  Saya Elaina. [<end_of_turn>] \n
```
*Catatan Krusial:* Token penutup `<end_of_turn>` pada giliran model **wajib ikut disupervisi** (tidak boleh di-mask atau terpotong oleh truncation) agar model belajar menghentikan kalimatnya sendiri secara natural.

---

## 4. Hyperparameter Rekomendasi

| Hyperparameter | Nilai | Alasan / Rasional |
|---|---|---|
| **LoRA Rank ($r$)** | `32` | Memberikan kapasitas representasi ekspresif yang cukup untuk tics bicara, gaya sarkastik, dan persona Elaina tanpa overfit. |
| **LoRA Alpha ($\alpha$)** | `64` | Rasio $\alpha / r = 2.0$ merupakan skala adaptasi standar yang stabil. |
| **LoRA Dropout** | `0.05` | Regularisasi ringan untuk mencegah adapter menghafal kalimat verbatim novel. |
| **Learning Rate** | `1e-4` | Nilai ideal untuk LoRA bf16 pada model 31B dengan Cosine Annealing scheduler. |
| **Warmup Ratio** | `0.03` | Menghindari lonjakan gradien tajam di awal langkah pelatihan. |
| **Effective Batch Size** | `16` | Dicapai dengan `per_device_batch_size = 2` dan `gradient_accumulation_steps = 8`. |
| **Epochs** | `2` | Mencegah overfitting pada korpus dataset ~3.800 baris. |
| **Sequence Length** | `2048` | Mencakup p95 token percakapan Elaina (~190 token) dengan headroom yang sangat lapang. |
| **Gradient Checkpointing** | `Aktif` | Wajib untuk menghemat memori aktivasi pada sekuens panjang. |
| **Optimizer** | `adamw_torch_fused` | Komputasi optimizer native fused PyTorch yang optimal untuk GPU Blackwell. |

---

## 5. Alokasi VRAM pada Blackwell 96 GB

```
+-----------------------------------------------------------+
| VRAM Total: 96 GB                                         |
+-----------------------------------------------------------+
| [Base Model 31B bf16]                ~62.5 GB             |
+-----------------------------------------------------------+
| [Aktivasi & Logits Vocab 262K]       ~6.0 - 8.0 GB        |
+-----------------------------------------------------------+
| [LoRA Weights, Gradients, Optimizer] ~1.5 GB              |
+-----------------------------------------------------------+
| [Free Margin / Headroom]             ~24.0 GB (Aman)      |
+-----------------------------------------------------------+
```
Margin bebas sebesar ~24 GB menjamin pelatihan tidak akan mengalami OOM (*Out Of Memory*) mendadak meskipun terjadi variasi panjang konteks percakapan.

