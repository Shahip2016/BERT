# BERT Implementation

This is a PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) from scratch.

## Features
- Full BERT architecture (Embeddings, Self-Attention, Encoders)
- Pre-training heads (Masked LM, NSP)
- Training loop with AdamW
- Verification script

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify setup:
   ```bash
   python src/verify_setup.py
   ```
3. Train:
   ```bash
   python src/train.py
   ```
