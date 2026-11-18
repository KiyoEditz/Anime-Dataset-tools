# 01 — Peta Alur & Diagram Sistem (Map Cara)

Dokumen ini memvisualisasikan seluruh alur kerja, arsitektur data, ketergantungan antar-komponen, dan siklus hidup artefak pada sistem fine-tuning persona Elaina.

---

## 1. Peta Alur Utama (End-to-End Pipeline)

Diagram berikut memperlihatkan alur data dari file ekstraksi novel mentah hingga deployment model siap pakai:

```mermaid
flowchart TD
    subgraph S1["1. Tahap Ekstraksi Lokal (CPU)"]
        EPUB["EPUB Novel Asli (EN / ID / JP)"] --> EXT["dialogue_extractor.py / build_dataset.py"]
        EXT --> RAW["output/raw_dialogue_pairs.jsonl"]
        SYN["augment_synthetic.py"] --> SYNP["output/synthetic_pairs.jsonl"]
    end

    subgraph S2["2. Kurasi Dataset (training/prepare_dataset.py)"]
        RAW & SYNP --> PREP["prepare_dataset.py"]
        SRC["sources.json (EN, ID, JP)"] --> PREP
        PROMPT_S["elaina_system_prompt_short.txt"] --> PREP
        PREP --> OUT_DIR["training/out/"]
        OUT_DIR --> TR["train.jsonl (3.5k rows)"]
        OUT_DIR --> VL["val.jsonl (250 rows)"]
        OUT_DIR --> META["manifest.json & stats.json"]
    end

    subgraph S3["3. Distribusi Cloud (Hugging Face Hub)"]
        OUT_DIR --> PUSH["push_dataset_to_hub.py"]
        PUSH --> HF_DATA["HF Dataset Repo (Private)"]
    end

    subgraph S4["4. Eksekusi Training (Molab Blackwell 96 GB)"]
        HF_DATA --> NB["molab_train_elaina.py (Marimo)"]
        BASE["KiyoEditz/Gemma-4-Dark-Gemistry-31B (HF Hub)"] --> NB
        NB --> GATE["Environment Gate (Free Disk & VRAM)"]
        GATE --> LOAD["Load Base Model (bf16)"]
        LOAD --> FREEZE["Freeze Vision Tower & Projector"]
        FREEZE --> LORA["Attach LoRA Adapter (Language Decoder)"]
        LORA --> TRAIN["Trainer: Response-Only Loss Masking"]
        TRAIN --> CKPT["HF Adapter Repo (Checkpoints)"]
    end

    subgraph S5["5. Ekspor & Inferensi"]
        CKPT --> MERGE["Merge Adapter to Base (bf16)"]
        MERGE --> HF_MERGED["HF Merged Repo"]
        HF_MERGED --> GGUF_CONV["gguf_convert.py + llama.cpp"]
        GGUF_CONV --> GGUF_OUT["elaina-Q4_K_M.gguf + mmproj-F16.gguf"]
        GGUF_OUT --> HF_GGUF["HF GGUF Repo"]
        GGUF_OUT --> SERVER["llama-server / Ollama / vLLM"]
    end
```

---

## 2. Peta Ketergantungan Cell Notebook Marimo (`molab_train_elaina.py`)

Notebook Marimo dijalankan secara reaktif sebagai *Directed Acyclic Graph* (DAG). Cell dengan tombol (`mo.ui.run_button`) memerlukan pemicu manual untuk mencegah pemborosan GPU:

```mermaid
flowchart TD
    C1["Cell 1: Imports & CFG Setup"] --> C2["Cell 2: Bootstrap elaina_train_lib"]
    C1 --> C3["Cell 3: Readme & Disk Cost Table"]
    C2 --> C4["Cell 4: Environment Gate Check"]
    C1 --> C5["Cell 5: HF Login [BUTTON]"]
    C5 --> C6["Cell 6: Repo Slug Inputs"]
    C1 --> C7["Cell 7: Install Training Stack [BUTTON]"]
    C7 --> C8["Cell 8: Resolve Base Model Info"]
    C6 & C8 --> C9["Cell 9: Pull Prepared Dataset"]
    C8 --> C10["Cell 10: Load Tokenizer / Processor"]
    C9 & C10 --> C11["Cell 11: Chat Template Proof & Masking Test"]
    C4 & C8 --> C12["Cell 12: Load Base Model bf16 [BUTTON]"]
    C12 --> C13["Cell 13: Attach LoRA Adapter [BUTTON]"]
    C9 & C10 & C11 --> C14["Cell 14: Tokenize Dataset with Masking"]
    C13 & C14 --> C15["Cell 15: Build Trainer Object"]
    C15 --> C16["Cell 16: Train / Resume from Hub [BUTTON]"]
    C16 --> C17["Cell 17: Eval on Held-out Chapters [BUTTON]"]
    C16 --> C18["Cell 18: Generate Sample Probes [BUTTON]"]
    C16 --> C19["Cell 19: Push Final Adapter [BUTTON]"]
    C19 --> C20["Cell 20: Merge bf16 & Stream Shards [BUTTON]"]
    C20 --> C21["Cell 21: GGUF Conversion & mmproj [BUTTON]"]
    C21 --> C22["Cell 22: Disk Cleanup [BUTTON]"]
    C22 --> C23["Cell 23: Final Report & Inference Docs"]

    classDef runBtn fill:#f96,stroke:#333,stroke-width:2px;
    class C5,C7,C12,C13,C16,C17,C18,C19,C20,C21,C22 runBtn;
```

---

## 3. Peta Pemrosesan Data & Split Anti-Bocor (`prepare_dataset.py`)

Bagaimana dialog mentah disaring dan dipisahkan menjadi train dan validation secara *disjoint*:

```mermaid
flowchart LR
    RAW["Raw Dialogue Pairs"] --> LOAD["Load Records per Language"]
    LOAD --> FLT["Length Filter<br/>(Min/Max Char limits)"]
    FLT --> DEDUP1["Exact Pair & Response Dedup<br/>(Language-scoped)"]
    DEDUP1 --> DEDUP2["Fuzzy Dedup<br/>(SequenceMatcher 92%)"]
    DEDUP2 --> CH_SPLIT{"Chapter Splitter<br/>(Global Disjoint Key)"}
    
    CH_SPLIT -->|"Val Chapters (6%)"| VAL_SET["Held-out Val Set<br/>(Short prompt only)"]
    CH_SPLIT -->|"Train Chapters (94%)"| TRN_SET["Train Set<br/>(Short prompt + 10% Long)"]

    TRN_SET --> ASSERT{"Cross-Lingual Leak Check"}
    VAL_SET --> ASSERT
    ASSERT -->|"Pass: 0 Chapter Overlap"| OUT["JSONL & Metadata Writer"]
```

---

## 4. Peta Siklus Hidup Memori & Bounded-Disk Hub Sync

Mekanisme perlindungan kapasitas disk pada model 62,5 GB di volume 150 GB:

```mermaid
sequenceDiagram
    autonumber
    actor User as Pengguna / Molab
    participant Local as Disk Lokal Container (/root/work)
    participant Hub as Hugging Face Hub (Private Repo)

    Note over User,Local: 1. Training Selesai
    User->>Local: Simpan adapter LoRA (~200 MB)
    Local->>Hub: Upload adapter folder
    Hub-->>Local: Verifikasi adapter_config.json ada

    Note over User,Local: 2. Streaming Shards Merge
    User->>Local: Merge base model + adapter di RAM/VRAM
    loop Setiap Shard Safetensors (~4 GB)
        Local->>Local: Tulis model-0000X-of-00016.safetensors
        Local->>Hub: Upload shard ke repo
        Hub-->>Local: Verifikasi ukuran file di Hub
        Local->>Local: Hapus shard lokal (unlink)
    end
    Note over Local: Disk lokal tetap hemat (puncak hanya 1 shard)

    Note over User,Local: 3. GGUF Quantization
    Local->>Hub: Unduh merged model
    Local->>Local: Kompilasi llama.cpp (CPU)
    Local->>Local: Ekspor elaina-f16.gguf & mmproj-F16.gguf
    loop Setiap Quant (Q4_K_M, Q6_K, dll.)
        Local->>Local: Kuantisasi ke elaina-Q4_K_M.gguf
        Local->>Hub: Upload elaina-Q4_K_M.gguf
        Local->>Local: Hapus elaina-Q4_K_M.gguf lokal
    end
    Local->>Local: Hapus intermediate f16 GGUF
```

---

## 5. Pohon Keputusan Environment Gate & Hardware Tier

Aturan otomatis penentuan jalur model berdasarkan sumber daya mesin:

```mermaid
flowchart TD
    START["Mulai Pengukuran Sistem"] --> READ["Ukur Free Disk, Host RAM, VRAM GPU"]
    READ --> VRAM_CHECK{"VRAM GPU?"}
    
    VRAM_CHECK -->|"VRAM >= 80 GB<br/>(NVIDIA Blackwell / A100 / H100)"| DISK_CHECK1{"Free Disk?"}
    VRAM_CHECK -->|"VRAM 24 - 40 GB<br/>(RTX 3090/4090 / A10G)"| DISK_CHECK2{"Free Disk >= 45 GB?"}
    VRAM_CHECK -->|"< 24 GB VRAM"| BLOCK["TIER: BLOCKED<br/>(Alihkan ke HF Jobs / GPU Sewaan)"]
    
    DISK_CHECK1 -->|">= 150 GB"| TIER_A["TIER: full31b<br/>Model: KiyoEditz/Gemma-4-Dark-Gemistry-31B (bf16)<br/>Semua Quant GGUF (Q2_K s.d Q8_0)"]
    DISK_CHECK1 -->|"90 - 149 GB"| TIER_B["TIER: small31b<br/>Model: KiyoEditz/Gemma-4-Dark-Gemistry-31B (bf16)<br/>GGUF Q8_0 dilewati untuk hemat ruang"]
    DISK_CHECK1 -->|"< 90 GB"| BLOCK

    DISK_CHECK2 -->|"Ya"| TIER_C["TIER: 12b<br/>Model: google/gemma-4-12B-it (bf16)<br/>Alur ekspor & inferensi identik"]
    DISK_CHECK2 -->|"Tidak"| BLOCK
```

---

## 6. State Machine: Checkpointing & Resume Pelatihan

Alur penanganan sesi yang terputus di Molab:

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Inisialisasi: Klik Tombol Train
    Inisialisasi --> PeriksaHubCheckpoint: Cari checkpoint-* di Adapter Repo Hub
    
    PeriksaHubCheckpoint --> ResumeTraining: Checkpoint Ditemukan (misal step 300)
    PeriksaHubCheckpoint --> TrainingBaru: Tidak Ada Checkpoint (Mulai dari scratch)

    ResumeTraining --> LoopPelatihan: Muat Optimizer, LR Scheduler & State
    TrainingBaru --> LoopPelatihan: Langkah 1
    
    state LoopPelatihan {
        ForwardPass --> BackwardPass
        BackwardPass --> OptimizerStep
        OptimizerStep --> LogikaSimpan: Tiap SAVE_STEPS (100 step)
        LogikaSimpan --> UploadCheckpointKeHub: Simpan Checkpoint Lokal
        UploadCheckpointKeHub --> ForwardPass: Lanjut Step Berikutnya
    }

    LoopPelatihan --> PutusKoneksi: Sesi Container Timeout / Disconnect
    PutusKoneksi --> Inisialisasi: Restart Notebook & Hubungkan Kembali
    
    LoopPelatihan --> Selesai: Mencapai MAX_STEPS / EPOCHS
    Selesai --> EvaluasiDanExport: Upload Checkpoint Final
```

