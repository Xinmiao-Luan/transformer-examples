# Minimal Transformer Examples in PyTorch

This repository provides **simple, minimal implementations of Transformer architectures in PyTorch**, inspired by the seminal paper:

> *"Attention Is All You Need"*
> Vaswani et al., 2017

The goal of this project is **not** to reproduce a full-scale Transformer model, but to offer **small, readable examples** that illustrate how the core components work.

**These models are designed to be:**

small and easy enough to understand and modify

---

## What’s Included

The repository contains **three types of Transformer architectures**, each implemented from scratch with standard PyTorch modules:

### 1. Encoder-only (e.g. BERT)

* Stacks self-attention + feed-forward layers
* Processes full sequences in parallel
* Suitable for **classification, embedding extraction, etc.**

File: `src/encoder_only.py`

---

### 2. Encoder + Decoder (Original Transformer structure in the paper)

* Classic seq2seq architecture
* Encoder produces memory representations
* Decoder uses **masked self-attention** + **cross-attention**
* Suitable for **machine translation, summarization**

File: `src/encoder_decoder.py`

---

### 3. Decoder-only (e.g. GPT)

* Masked self-attention only
* Autoregressive language modeling
* Suitable for **text generation**

File: `src/decoder_only.py`

---

## Design Philosophy

These implementations are intentionally:

* Minimal (a few hundred lines each)
* Modular and readable
* Using PyTorch’s built-in `nn.MultiheadAttention`

They include:
* Sinusoidal positional encoding
* Feed-forward networks
* Residual connections & layer normalization
* Causal masks for decoders
* Simple embeddings and output heads

But they **avoid** extra complexities such as:

* Optimizers
* Distributed training
* Mixed precision
* Tokenizers
* Dataset loaders

so you can focus on understanding the architecture itself.

---

## Installation

```bash
git clone https://github.com/Xinmiao-Luan/transformer-examples.git
cd transformer-examples
pip install -r requirements.txt
```

---

## Run the Examples

Each script includes a tiny demo that runs a forward pass and prints output shapes.

```bash
python -m src.encoder_only
python -m src.encoder_decoder
python -m src.decoder_only
```

---

## Project Structure

```
transformer-examples/
├─ README.md
├─ requirements.txt
└─ src/
   ├─ common.py
   ├─ encoder_only.py
   ├─ encoder_decoder.py
   └─ decoder_only.py
```

---

## Good Learning Use-Cases

This repo can help you:

* Understand how Transformer blocks are wired together
* Build intuition around **self-attention vs. cross-attention**
* See how causal masks enforce autoregressive behavior
* Experiment with your own architectures
* Create toy tasks (copy task, character LM, etc.)

---

## Limitations

These examples **do not aim for state-of-the-art performance**.
They are intentionally simple and minimal.

They are **not** optimized for:

* speed
* large vocabularies
* long sequences
* GPU training

---

## Acknowledgment

This work was **inspired by the original Transformer architecture** described in:

> Vaswani, A., et al.
> *“Attention Is All You Need.”*
> NeurIPS 2017.

---

## Contributions

If you’d like to:

* fix bugs
* improve clarity
* add small extensions

Pull requests are welcome — **as long as they keep the code simple and educational**.

---

## License

MIT License. Feel free to use and modify for your own learning or teaching.

---

## Final Notes

This repository is meant to be a **lightweight educational resource**.
If you’re new to Transformers, reading and experimenting with small models is a great way to build intuition before working with large-scale implementations.

Happy hacking! ✨
