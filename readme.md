# Peptide Immunogenicity Prediction using Topological and Sequential Features

**Python** • **PyTorch** • **RDKit**

A deep learning framework for predicting peptide immunogenicity by combining topological features from molecular dynamics and sequential features from protein language models.

## Features

- Integrates topological data analysis (TDA) and ESM-2 protein language model embeddings
- Cross-attention fusion of structural and sequential features
- End-to-end pipeline from sequence to immunogenicity score

## Quick Start

1. Install dependencies (Python ≥3.8, PyTorch, RDKit, etc.)
2. Run feature extraction:
    ```bash
    python md.py          # Molecular conformations
    python topo.py        # Topological features
    python esm.py         # Sequence embeddings
    ```
3. Train model:
    ```bash
    python train.py
    ```
4. Make prediction:
    ```bash
    python infer.py --sequence "NPCNNPLKARCMK"
    ```

## Happy Researching

If you use this work, please start!
