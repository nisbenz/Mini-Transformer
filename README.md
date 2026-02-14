#  English → Darija Transformer

An **educational project** built from scratch to learn the Transformer architecture by implementing a lightweight English →  Darija translator, trained on ~16k sentence pairs.

> This is not a production translation system — it's a hands-on learning exercise to deeply understand how attention, positional encoding, encoder-decoder stacks, and beam search work together.

## Dataset

[**English-to-Moroccan-Darija**](https://huggingface.co/datasets/BounharAbdelaziz/English-to-Moroccan-Darija/tree/main/data) by BounharAbdelaziz on HuggingFace — ~16k parallel sentence pairs.

## References & Inspiration

- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (the original Transformer paper)
- 🎥 [Coding a Transformer from Scratch](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=1417s) — 3Blue1Brown

## Architecture Overview

```mermaid
flowchart TB
    subgraph Input
        EN["English Sentence"]
        DA["Darija Sentence (shifted right)"]
    end

    subgraph Encoder["Encoder (×3 Layers)"]
        direction TB
        EE["Token Embedding + Positional Encoding"]
        EL1["Encoder Layer 1"]
        EL2["Encoder Layer 2"]
        EL3["Encoder Layer 3"]
        EN_NORM["Final LayerNorm"]
        EE --> EL1 --> EL2 --> EL3 --> EN_NORM
    end

    subgraph Decoder["Decoder (×3 Layers)"]
        direction TB
        DE["Token Embedding + Positional Encoding"]
        DL1["Decoder Layer 1"]
        DL2["Decoder Layer 2"]
        DL3["Decoder Layer 3"]
        DN_NORM["Final LayerNorm"]
        FC["Linear → Vocab"]
        DE --> DL1 --> DL2 --> DL3 --> DN_NORM --> FC
    end

    EN --> EE
    DA --> DE
    EN_NORM -- "Encoder Output" --> DL1
    EN_NORM -- "Encoder Output" --> DL2
    EN_NORM -- "Encoder Output" --> DL3
    FC --> OUT["Output Probabilities"]
```

## Encoder Layer (Pre-LN)

```mermaid
flowchart TB
    X_IN["Input x"] --> LN1["LayerNorm"]
    LN1 --> SA["Multi-Head Self-Attention"]
    SA --> DROP1["Dropout (0.3)"]
    DROP1 --> ADD1(("+"))
    X_IN --> ADD1

    ADD1 --> LN2["LayerNorm"]
    LN2 --> FFN["FFN (256 → 512 → 256)"]
    FFN --> DROP2["Dropout (0.3)"]
    DROP2 --> ADD2(("+"))
    ADD1 --> ADD2

    ADD2 --> X_OUT["Output"]

    style ADD1 fill:#4CAF50,color:#fff
    style ADD2 fill:#4CAF50,color:#fff
```

## Decoder Layer (Pre-LN)

```mermaid
flowchart TB
    X_IN["Input x"] --> LN1["LayerNorm"]
    LN1 --> MSA["Masked Self-Attention"]
    MSA --> DROP1["Dropout (0.3)"]
    DROP1 --> ADD1(("+"))
    X_IN --> ADD1

    ADD1 --> LN2["LayerNorm"]
    LN2 --> CA["Cross-Attention (Q=dec, K/V=enc)"]
    CA --> DROP2["Dropout (0.3)"]
    DROP2 --> ADD2(("+"))
    ADD1 --> ADD2

    ADD2 --> LN3["LayerNorm"]
    LN3 --> FFN["FFN (256 → 512 → 256)"]
    FFN --> DROP3["Dropout (0.3)"]
    DROP3 --> ADD3(("+"))
    ADD2 --> ADD3

    ADD3 --> X_OUT["Output"]

    ENC_OUT["Encoder Output"] -.-> CA

    style ADD1 fill:#4CAF50,color:#fff
    style ADD2 fill:#4CAF50,color:#fff
    style ADD3 fill:#4CAF50,color:#fff
    style ENC_OUT fill:#2196F3,color:#fff
```

## Multi-Head Attention

```mermaid
flowchart LR
    Q["Q"] --> WQ["W_q Linear"]
    K["K"] --> WK["W_k Linear"]
    V["V"] --> WV["W_v Linear"]

    WQ --> SPLIT_Q["Split into 4 Heads"]
    WK --> SPLIT_K["Split into 4 Heads"]
    WV --> SPLIT_V["Split into 4 Heads"]

    SPLIT_Q --> ATTN["Scaled Dot-Product\nAttention (×4)"]
    SPLIT_K --> ATTN
    SPLIT_V --> ATTN

    ATTN --> CONCAT["Concat Heads"]
    CONCAT --> WO["W_o Linear"]
    WO --> OUT["Output"]
```

## Training Pipeline

```mermaid
flowchart LR
    CSV["CSV Dataset\n~16k pairs"] --> CLEAN["Clean\n• lowercase\n• filter ≥50 words"]
    CLEAN --> TOK["BPE Tokenizer\nvocab=5000"]
    TOK --> DL["DataLoader\nbatch=32"]
    DL --> MODEL["Tiny Transformer\n3L / 256d / 512ff"]
    MODEL --> LOSS["CrossEntropyLoss\nlabel_smoothing=0.1"]
    LOSS --> OPT["Adam + OneCycleLR\nmax_lr=0.0007"]
    OPT --> AMP["Mixed Precision\nGradScaler"]
    AMP -->|20 epochs| MODEL

    style CLEAN fill:#FF9800,color:#fff
    style MODEL fill:#9C27B0,color:#fff
    style AMP fill:#2196F3,color:#fff
```

## Inference: Beam Search

```mermaid
flowchart TB
    IN["English Input"] --> ENC["Encode (frozen)"]
    ENC --> BEAM["Beam Search (k=3)"]

    BEAM --> B1["Beam 1: score=-1.2"]
    BEAM --> B2["Beam 2: score=-1.5"]
    BEAM --> B3["Beam 3: score=-2.1"]

    B1 --> EXPAND["Expand top-k tokens\nper beam"]
    B2 --> EXPAND
    B3 --> EXPAND

    EXPAND --> PRUNE["Keep top 3 beams\nby cumulative log-prob"]
    PRUNE -->|"repeat until </s>"| BEAM
    PRUNE --> BEST["Best Translation"]

    style BEST fill:#4CAF50,color:#fff
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `D_MODEL` | 256 |
| `D_FF` | 512 |
| `N_HEAD` | 4 (64 dims/head) |
| `NUM_LAYERS` | 3 |
| `dropout` | 0.3 |
| `VOCAB_SIZE` | 5000 (shared BPE) |
| `MAX_LEN` | 256 |
| LayerNorm | **Pre-LN** |

## Training Config

| Setting | Value |
|---------|-------|
| Optimizer | Adam (β₁=0.9, β₂=0.98) |
| Scheduler | OneCycleLR |
| `max_lr` | 0.0007 |
| `label_smoothing` | 0.1 |
| Epochs | 20 |
| Batch Size | 32 |
| Mixed Precision | FP16 via `torch.amp` |
| Gradient Clipping | 1.0 |

## Usage

1. Open `Transformer.ipynb` in **Google Colab** (T4 GPU)
2. Upload `train-00000-of-00001.csv` to the runtime
3. Run all cells sequentially (Cell 1 → 6)
4. Test translations in Cell 5:
```python
translate_beam("How are you?", model, tokenizer, device)
```

## Project Structure

```
├── Transformer.ipynb    # Main notebook (6 cells)
├── train-00000-of-00001.csv # English-Darija dataset (~16k rows)
├── tokenizer/              # Saved BPE tokenizer files
│   ├── vocab.json
│   └── merges.txt
├── checkpoint_epoch_*.pth  # Training checkpoints
└── README.md               # This file
```
