# MedSearch: Clinical Literature Retrieval System

A multi-stage biomedical search engine combining lexical and semantic retrieval with neural re-ranking. Achieves **28% nDCG@10 improvement** over BM25 baseline on TREC-COVID benchmark.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Architecture
```
Query
  ↓
┌─────────────────────────────────────┐
│  Stage 1: Candidate Retrieval       │
│  ├── BM25 (lexical) → Top 1000      │
│  └── BioBERT + FAISS → Top 1000     │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  Stage 2: Score Fusion (α=0.6)      │
│  → Top 100 candidates               │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  Stage 3: Cross-Encoder Re-ranking  │
│  → Final Top 10 results             │
└─────────────────────────────────────┘
```

## Results

| Configuration | nDCG@10 | MRR@10 | Improvement |
|:--------------|:-------:|:------:|:-----------:|
| BM25 only | 0.544 | 0.802 | — |
| BioBERT Dense only | 0.418 | 0.639 | -23.2% |
| BM25 + BioBERT Fusion | 0.628 | 0.842 | +15.4% |
| **+ Cross-Encoder (Final)** | **0.697** | **0.920** | **+28.2%** |

## Tech Stack

| Component | Technology |
|:----------|:-----------|
| Lexical Retrieval | BM25 (rank-bm25) |
| Dense Retrieval | BioBERT (sentence-transformers) |
| Vector Search | FAISS |
| Re-ranking | Cross-Encoder (ms-marco-MiniLM) |
| Dataset | TREC-COVID (171K PubMed documents) |
| API | FastAPI |
| Demo | Gradio |

## Quick Start

### Installation
```bash
git clone https://github.com/hasinisirigari/medsearch.git
cd medsearch
pip install -r requirements.txt
```

### Run API
```bash
python src/api/main.py
```
Open http://127.0.0.1:8000/docs for interactive API documentation.

### Run Gradio Demo
```bash
python app.py
```
Open http://127.0.0.1:7860 for the web interface.

## Project Structure
```
medsearch/
├── app.py                    # Gradio demo
├── src/api/main.py           # FastAPI service
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_bm25_baseline.ipynb
│   ├── 03_dense_retrieval.ipynb
│   ├── 04_fusion.ipynb
│   └── 05_evaluation.ipynb
├── results/                  # Evaluation metrics & ablation study
└── requirements.txt
```

## Reproducing Results

Large model files are not included in the repo. To reproduce:

1. **Download data**: Run `notebooks/01_data_download.ipynb`
2. **Build BM25 index**: Run `notebooks/02_bm25_baseline.ipynb`
3. **Create embeddings**: Run `notebooks/03_dense_retrieval.ipynb`
4. **Run fusion & evaluation**: Run `notebooks/04_fusion.ipynb`

## Demo

[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/s-hasini/medsearch)

## License

MIT License
