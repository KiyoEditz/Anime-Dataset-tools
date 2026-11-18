# 04 — Chat Template & Response Masking

Kembali ke [dokumen utama](README.md).

Dokumen ini membahas format chat template pada keluarga Gemma (terutama Gemma 4), penanganan token kontrol giliran bicara (`<start_of_turn>`, `<end_of_turn>`), loss masking hanya pada respons asisten (response-only loss masking), serta penghindaran hilangnya token penutup saat pemotongan sekuens (truncation).

---

## H2 — Truncation Memotong Token End-of-Turn

**Severity:** 🟠 Tinggi  
**Lokasi:** [`elaina_train_lib.py` L328-L333](../../elaina_train_lib.py#L328-L333)  
**Gejala:** Model setelah training terus menerus mengoceh (babbling / runaway generation), tidak mau berhenti saat menjawab pertanyaan, atau mengeluarkan teks repetitif tak berujung (Q&A poin 3).

**Penyebab:**
Di fungsi `tokenize_with_mask`:
```python
if len(response_ids) > budget:
    response_ids = response_ids[:budget]
    truncated = 1
```
Jika respons asisten melebihi sisa budget token (`budget`), respons dipotong secara mentah pada token ke-`budget`. Akibatnya, token penutup giliran (misalnya `<end_of_turn>` pada Gemma) yang berada di akhir respons terbuang dari `labels`! Model dilatih untuk memprediksi token teks terakhir tanpa pernah dilatih memprediksi token `<end_of_turn>`, sehingga model kehilangan kemampuan menghentikan generasi secara mandiri.

### Perbaikan:
Ada dua pendekatan yang benar:
1. **Opsi 1 (Buang baris yang melebihi max_seq - Rekomendasi):**
   Alih-alih memotong di tengah kalimat dan membuang token penutup, buang/skip baris yang overflow dari dataset training, atau potong pada tahap preprocessing teks sebelum tokenisasi.
   ```python
   if len(response_ids) > budget:
       # Melebihi budget: jangan latih contoh yang terpotong!
       return None # atau lewati baris ini dalam _encode()
   ```
2. **Opsi 2 (Pertahankan token penutup jika harus memotong):**
   Jika terpaksa dipotong, pastikan token terakhir selalu digantikan dengan token penutup (`eos_token_id` atau `tokenizer.convert_tokens_to_ids("<end_of_turn>")`):
   ```python
   if len(response_ids) > budget:
       eot_token = response_ids[-1] # token penutup asli
       response_ids = response_ids[:budget - 1] + [eot_token]
       truncated = 1
   ```

**Kriteria Selesai:**
Uji pada baris yang melebihi panjang maksimum menunjukkan token terakhir dari `labels` (yang bukan -100) selalu merupakan token `<end_of_turn>` atau `<eos>`.

---

## H3 — Asumsi `messages[:2]` dan Percakapan Multi-Turn

**Severity:** 🟠 Tinggi  
**Lokasi:** [`elaina_train_lib.py` L293-L298](../../elaina_train_lib.py#L293-L298)  
**Gejala:** Masking salah posisi (loss dihitung pada pertanyaan user atau terpotong) jika dataset memiliki baris multi-turn (lebih dari 1 putaran tanya-jawab) atau baris tanpa pesan sistem.

**Penyebab:**
Fungsi `tokenize_with_mask` saat ini melakukan hardcode:
```python
prompt_text = tokenizer.apply_chat_template(
    messages[:2], tokenize=False, add_generation_prompt=True
)
full_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
```
Kode ini mengasumsikan bahwa `messages[0]` selalu `system` dan `messages[1]` selalu `user`, sedangkan `messages[2]` adalah respons asisten. Jika formatnya:
1. Hanya `user` dan `assistant` (2 pesan, single turn tanpa system): `messages[:2]` mencakup seluruh percakapan termasuk balasan asisten! Akibatnya seluruh label bernilai `-100`.
2. Multi-turn (misal 5 pesan: system, user, assistant, user, assistant): `messages[:2]` hanya mengambil prompt pertama, dan asisten giliran pertama malah ikut dihitung sebagai target supervised loss bersama giliran user berikutnya!

### Perbaikan:
Dukung struktur fleksibel dan multi-turn secara dinamis:
Untuk single-turn sembarang (dengan atau tanpa system prompt):
```python
# Prompt adalah semua pesan SEBELUM pesan asisten terakhir
prompt_messages = messages[:-1]
response_message = messages[-1]

prompt_text = tokenizer.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True
)
full_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
```
Untuk percakapan multi-turn penuh (supervisi pada setiap giliran asisten):
Gunakan token traversal: cari pasangan token opener model (`<start_of_turn>model\n`) dan token closer (`<end_of_turn>\n`), lalu set nilai `labels` menjadi `-100` di luar span asisten.

---

## H7 — Chat Template Processor vs Tokenizer pada Gemma 4 VLM

**Severity:** 🟠 Tinggi · **[VERIFIKASI DULU: V3]**  
**Lokasi:** [`molab_train_elaina.py` L631-L641](../../molab_train_elaina.py#L631-L641), [`elaina_train_lib.py` L244-L283](../../elaina_train_lib.py#L244-L283)  
**Latar Belakang Arsitektur Gemma 4:**
1. Gemma 4 adalah Vision-Language Model (VLM). Model ini di-load menggunakan `AutoProcessor` yang membungkus tokenizer teks dan visual feature extractor.
2. Template chat (`chat_template`) dapat terdefinisi pada level `processor` atau `tokenizer`.
3. Gemma 4 memperkenalkan dukungan native untuk role `system`.
4. Gemma 4 memiliki kanal kontrol pemikiran (*thinking / reasoning channels*) dan tag penutup khusus.

### Snippet Verifikasi V3:
Jalankan inspeksi pada lingkungan dengan transformers gemma 4:
```python
from transformers import AutoProcessor, AutoTokenizer

model_id = "google/gemma-4-31B-it"
try:
    proc = AutoProcessor.from_pretrained(model_id)
    tok = proc.tokenizer
    print("Processor chat_template ada:", hasattr(proc, "chat_template") and proc.chat_template is not None)
    print("Tokenizer chat_template ada:", hasattr(tok, "chat_template") and tok.chat_template is not None)
except Exception as e:
    print("Error load processor:", e)

messages = [
    {"role": "system", "content": "You are Elaina."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Fine, I suppose."}
]
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print("Rendered sample:\n", repr(rendered))
```

### Karakteristik Template Antar-Versi Gemma (Terverifikasi V3):
| Fitur | Gemma 1 / 2 | Gemma 3 | Gemma 4 |
|---|---|---|---|
| Role `system` | **Ditolak** (Template error jika role system diberikan) | Digabung otomatis ke giliran user pertama | **Didukung Natively** via tag `<\|turn>system` |
| Token Giliran Model | `<start_of_turn>model\n` | `<start_of_turn>model\n` | `<\|turn>model\n` (`sot_token="<\|turn>"`) |
| Token Penutup Giliran | `<end_of_turn>\n` | `<end_of_turn>\n` | `<turn\|>\n` (`eot_token="<turn\|>"`) |
| Modality | Text-only | VLM multi-modal | Multimodal (Text, Vision, Audio) |
| Kanal Thinking | Tidak ada | Tidak ada | `<\|channel>thought\n<channel\|>` pada `add_generation_prompt=True` |

### Penyesuaian `derive_turn_parts()`:
Pastikan `derive_turn_parts()` memeriksa stabilitas prefix secara ketat:
```python
# Memastikan add_generation_prompt=True benar-benar menghasilkan prefix identik
user_only = [{"role": "user", "content": "TEST_PROBE"}]
plain = apply_template(user_only, tokenize=False, add_generation_prompt=False)
with_gen = apply_template(user_only, tokenize=False, add_generation_prompt=True)
assert with_gen.startswith(plain)
```
Jika tokenizer menyuntikkan token thinking awal (misal `<thought>`), token tersebut harus diperlakukan secara konsisten antara saat training dan inferensi agar model tidak mengalami degradasi respons.

