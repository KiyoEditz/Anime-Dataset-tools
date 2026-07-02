# Elaina Dataset Tool

Pipeline untuk mengubah novel *Majo no Tabitabi / The Journey of Elaina*
(format EPUB/PDF milikmu sendiri) menjadi dataset fine-tuning chatbot
berformat JSONL, berisi kepribadian, gaya bicara, dan ingatan Elaina.

Pipeline ini mendukung provider LLM mana pun yang OpenAI-compatible.
Lihat `.env.example` untuk 5 opsi yang sudah dikonfigurasi siap pakai.
**Rekomendasi**: Claude Sonnet (via Anthropic API + LiteLLM atau OpenRouter)
untuk kualitas ekstraksi terbaik.

## Instalasi

```bash
pip install -r requirements.txt
cp .env.example .env
# Buka .env, pilih SATU blok provider, hapus tanda # di baris-barisnya,
# lalu isi key dan URL yang sesuai.
export $(grep -v '^#' .env | xargs)   # load env vars (skip baris komentar)
```

Test koneksi ke provider sebelum mulai pipeline:
```bash
curl $LLM_BASE_URL/models -H "Authorization: Bearer $LLM_API_KEY"
# Harus mengembalikan list model, bukan error 401/403/connection refused
```

## Setup LiteLLM + Claude (Opsi 1 — Terbaik)

Kalau kamu sudah punya LiteLLM jalan di Colab via Cloudflare Tunnel dan
ingin tambahkan Claude, edit `config.yaml` LiteLLM kamu:

```yaml
model_list:
  - model_name: claude-sonnet          # nama alias bebas
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: hermes-4-14b           # model lokal tetap ada
    litellm_params:
      model: ollama/hermes-4-14b-abliterated
      api_base: http://localhost:11434
```

Lalu di Colab, set env var **sebelum** menjalankan LiteLLM:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
litellm --config config.yaml
```

API key Anthropic: daftar di `console.anthropic.com` → menu *API Keys*.

## Struktur folder

```
elaina-dataset-tool/
  input/      <- taruh file .epub / .pdf novel di sini
  text/       <- (auto) hasil ekstraksi teks per-bab
  output/     <- (auto) semua hasil akhir & intermediate
```

## Cara pakai

1. Taruh file EPUB/PDF novel kamu di folder `input/`. Beri nama urut
   (mis. `01_vol1.epub`, `02_vol2.epub`) supaya urutan ceritanya benar —
   ini penting karena ingatan Elaina dibangun secara kronologis.

2. Jalankan seluruh pipeline sekaligus:

   ```bash
   python run_all.py
   ```

   Atau jalankan tahap per tahap (berguna untuk debug / cek hasil tiap
   tahap sebelum lanjut):

   ```bash
   python extract_text.py        # 1. EPUB/PDF -> teks bersih per bab
   python persona_extractor.py   # 2. bangun profil kepribadian & ingatan
   python dialogue_extractor.py  # 3. ekstrak pasangan dialog dari novel
   python augment_synthetic.py 10  # 4. (opsional) tambah Q&A sintetis orisinal
   python build_dataset.py       # 5. gabungkan jadi JSONL final
   python validate_dataset.py    # cek kualitas & struktur hasil akhir
   ```

3. Hasil akhir: `output/elaina_finetune_dataset.jsonl`, format:

   ```json
   {"messages": [
     {"role": "system", "content": "You are Elaina, ..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "Elaina's response..."}
   ]}
   ```

   Format ini langsung kompatibel dengan kebanyakan framework fine-tuning
   chat (Axolotl, LLaMA-Factory, OpenAI fine-tuning format, dst).

## Kenapa prosesnya bertahap?

- **persona_extractor.py** jalan duluan supaya ada satu "character bible"
  (`output/elaina_profile.json` + `output/elaina_system_prompt.txt`) yang
  konsisten dipakai sebagai system prompt di semua contoh.
- **dialogue_extractor.py** mengambil adegan asli dari novel — ini yang
  membuat respons Elaina terasa otentik sesuai sumber.
- **augment_synthetic.py** sengaja dipisah karena isinya BUKAN kutipan
  novel, melainkan Q&A baru yang dibuat model berdasarkan profil. Ini
  membantu Elaina-bot menjawab pertanyaan yang tidak persis ada di novel,
  dan mengurangi dataset yang isinya cuma menyalin ulang teks asli.

## Memilih provider & model

Tool ini bicara ke model lewat satu wrapper OpenAI-compatible
(`llm_client.py`), jadi providernya bisa ditukar tanpa ubah kode -- cukup
ganti 3 variabel di `.env`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
Lihat `.env.example` untuk dua opsi siap pakai:

- **LiteLLM kamu sendiri** (Ollama + Hermes-4-14B abliterated via
  Cloudflare Tunnel) -- privat, tapi context window terbatas (~40k token)
  dan kecepatan tergantung GPU Colab kamu.
- **NVIDIA Build / NIM** (https://build.nvidia.com) -- gratis, tanpa
  kartu kredit, endpoint OpenAI-compatible, banyak model open-weight
  dengan context window jauh lebih besar (umumnya 128k). Cocok khusus
  untuk tahap ekstraksi dataset ini karena bukan dipakai untuk chat
  interaktif, jadi rate limit ~40 req/menit di free tier-nya tidak
  masalah. Bisa pakai untuk seluruh pipeline, atau kombinasikan: NVIDIA
  Build untuk `persona_extractor.py`/`dialogue_extractor.py` (butuh
  konteks besar & JSON konsisten), lalu balik ke Hermes lokal kamu untuk
  keperluan lain.

  Model yang cocok untuk tugas ekstraksi terstruktur seperti ini
  (instruction-following kuat + context besar + JSON konsisten):
  `meta/llama-3.3-70b-instruct`, `qwen/qwen2.5-72b-instruct`, atau
  `mistralai/mistral-large`. Hindari model "reasoning" yang membungkus
  jawaban dengan teks `<think>...</think>` (mis. beberapa varian
  DeepSeek-R1) kecuali kamu menambah logika untuk membuang bagian itu
  sebelum parsing JSON.

  Catatan: free tier NVIDIA Build untuk prototyping/personal use, bukan
  untuk traffic produksi -- pas untuk membangun dataset ini.

## Mengatur ukuran chunk vs context window

Tiap panggilan ke model berisi: system prompt (~500-800 token) + potongan
teks bab + (khusus persona_extractor) profil yang sudah terkumpul +
ruang untuk output JSON. Aturan kasar: total input sebaiknya tidak lebih
dari ~60-70% context window model, sisanya untuk output.

| Context window model | PERSONA_CHUNK_CHARS | DIALOGUE_CHUNK_CHARS |
|---|---|---|
| ~40k (Hermes-4-14B lokal) | 6000 (default) | 4000 (default) |
| ~128k (Llama-3.3-70B, Qwen2.5-72B via NVIDIA Build) | 30000-40000 | 20000-25000 |

Naikkan nilainya di `.env` kalau pakai model context besar -- hasilnya:
lebih sedikit panggilan API, dan model dapat konteks adegan/bab yang
lebih utuh (kualitas ekstraksi biasanya lebih baik). Untuk novel 350+
halaman, ini juga jauh mengurangi jumlah total panggilan dibanding tetap
pakai chunk kecil.

`persona_extractor.py` juga otomatis memadatkan (`compress_profile`)
profil Elaina begitu ukurannya melewati `MAX_PROFILE_CHARS`, supaya
profil yang terus terakumulasi dari bab ke bab tidak membengkak dan
menghabiskan context window di novel yang banyak babnya.

## Kalau script berhenti karena rate limit / error

Sejak update terbaru, `persona_extractor.py` dan `dialogue_extractor.py`
otomatis menyimpan checkpoint (`output/persona_checkpoint.json` dan
`output/dialogue_checkpoint.json`) setelah setiap bab selesai diproses.

- **Rate limit SINGKAT** (mis. limit per-menit) → script otomatis
  menunggu lalu lanjut sendiri, tidak perlu tindakan apapun.
- **Rate limit PANJANG** (mis. limit token harian seperti Groq free tier
  yang cuma 100.000 token/hari) → script akan BERHENTI dengan pesan jelas
  berapa lama harus menunggu. Progress kamu **aman**, tidak hilang.
  Tunggu sampai waktu yang disebutkan (atau besok kalau itu limit harian),
  lalu jalankan ulang perintah yang sama persis:

  ```bash
  python persona_extractor.py     # atau python run_all.py
  ```

  Script akan melanjutkan dari bab terakhir yang berhasil, bukan mengulang
  dari bab 1.

**Catatan realistis soal kuota Groq gratis**: limit 100.000 token/hari
untuk `llama-3.3-70b-versatile` cukup kecil untuk novel panjang — dari
pengalaman, ~14 bab profil ekstraksi saja bisa menghabiskan kuota harian.
Untuk novel 80+ bab, kemungkinan besar kamu butuh beberapa hari berturut
(checkpoint akan menangani ini dengan aman), ATAU pertimbangkan pindah ke
Claude (via OpenRouter/LiteLLM) untuk proses yang jauh lebih cepat selesai
tanpa terputus-putus oleh limit harian.

**Kalau menambah volume baru** dan checkpoint lama sudah ada, hapus dulu
kedua file checkpoint sebelum rerun (supaya seluruh pipeline diproses
ulang secara kronologis dan konsisten dari awal, sesuai penjelasan di
bagian "Cara pakai" di atas):

```bash
rm -f output/persona_checkpoint.json output/dialogue_checkpoint.json
rm -f output/raw_dialogue_pairs.jsonl output/elaina_profile.json
```

## Tuning kualitas

- Kalau hasil ekstraksi bab dari PDF berantakan (header/footer ikut
  kebawa, kata terpotong aneh), cek dulu file di `text/` sebelum lanjut
  ke tahap berikutnya — EPUB biasanya jauh lebih bersih daripada PDF.
- Kalau banyak `[WARN] Gagal mendapatkan JSON valid`, kemungkinan model
  butuh prompt lebih ketat atau temperature lebih rendah. Coba turunkan
  `temperature` di `llm_client.chat_json` call masing-masing script.
- Jalankan `validate_dataset.py` setiap kali habis `build_dataset.py` —
  ini juga menampilkan 3 contoh acak biar kamu bisa spot-check kualitas.

## Catatan penggunaan

Dataset ini dibangun dari materi berhak cipta (novel *Majo no Tabitabi*
karya Jougi Shiraishi). Pipeline ini ditujukan untuk pemakaian
pribadi/riset di mesin kamu sendiri menggunakan salinan novel yang kamu
miliki. Kalau berencana membagikan model hasil fine-tuning atau dataset
ini secara publik/komersial, pertimbangkan implikasi hak ciptanya
terlebih dulu.
