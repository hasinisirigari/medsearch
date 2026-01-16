\# MedSearch: Clinical Literature Retrieval System



A multi-stage biomedical search engine combining lexical and semantic retrieval with neural re-ranking. Achieves \*\*28% nDCG@10 improvement\*\* over BM25 baseline on TREC-COVID benchmark.



\## Architecture

```

Query → BM25 (top 1000) + BioBERT/FAISS (top 1000)

&nbsp;     → Score Fusion (α=0.6) → Top 100

&nbsp;     → Cross-Encoder Re-ranking → Top 10

```



\## Results



| Configuration | nDCG@10 | MRR@10 | Improvement |

|---------------|---------|--------|-------------|

| BM25 only | 0.544 | 0.802 | - |

| BioBERT Dense only | 0.418 | 0.639 | -23.2% |

| BM25 + BioBERT Fusion | 0.628 | 0.842 | +15.4% |

| + Cross-Encoder (Final) | 0.697 | 0.920 | \*\*+28.2%\*\* |



\## Tech Stack



\- \*\*Retrieval\*\*: BM25 (rank-bm25), BioBERT (sentence-transformers), FAISS

\- \*\*Re-ranking\*\*: Cross-Encoder (ms-marco-MiniLM)

\- \*\*Evaluation\*\*: TREC-COVID benchmark (171K PubMed documents)

\- \*\*Deployment\*\*: FastAPI, Gradio



\## Quick Start



\### Installation

```bash

pip install -r requirements.txt

```



\### Run API

```bash

python src/api/main.py

\# Open http://127.0.0.1:8000/docs

```



\### Run Gradio Demo

```bash

python app.py

\# Open http://127.0.0.1:7860

```



\## Project Structure

```

medsearch/

├── app.py                 # Gradio demo

├── src/api/main.py        # FastAPI service

├── notebooks/

│   ├── 01\_data\_download.ipynb

│   ├── 02\_bm25\_baseline.ipynb

│   ├── 03\_dense\_retrieval.ipynb

│   ├── 04\_fusion.ipynb

│   └── 05\_evaluation.ipynb

├── results/               # Evaluation metrics

└── requirements.txt

```



\## Data \& Models



Large files not included in repo. To reproduce:

1\. Run `notebooks/01\_data\_download.ipynb` to download TREC-COVID

2\. Run notebooks 02-04 to build indices and embeddings

3\. Models saved to `models/` folder



\## Demo



🔗 \[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/s-hasini/medsearch)



\## Author



Hasini Sirigar

