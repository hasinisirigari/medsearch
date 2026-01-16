# MedSearch API - Clinical Literature Retrieval System

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import pickle
import faiss
import re
import time
from sentence_transformers import SentenceTransformer, CrossEncoder

# Stopwords for BM25
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'once', 'here', 'there', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
    'will', 'just', 'should', 'now', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'would', 'could', 'ought', 'of', 'it', 'its', 'this', 'that',
    'these', 'those', 'am', 'as', 'what', 'which', 'who', 'whom'
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and not t.isdigit() and t not in STOPWORDS]

# Model storage class
class Models:
    bm25 = None
    faiss_index = None
    dense_model = None
    cross_encoder = None
    corpus_df = None
    idx_to_doc_id = None

models = Models()

def load_all_models():
    print("Loading models...")
    
    # Load corpus
    models.corpus_df = pd.read_parquet("data/corpus.parquet")
    models.corpus_df["combined"] = models.corpus_df["title"] + " " + models.corpus_df["text"]
    models.idx_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(models.corpus_df["doc_id"])}
    print(f"  Corpus: {len(models.corpus_df):,} documents")
    
    # Load BM25
    with open("models/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    models.bm25 = bm25_data["bm25"]
    print("  BM25 index loaded")
    
    # Load FAISS index
    models.faiss_index = faiss.read_index("models/faiss_index.bin")
    print("  FAISS index loaded")
    
    # Load dense model
    models.dense_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    print("  BioBERT model loaded")
    
    # Load cross-encoder
    models.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    print("  Cross-encoder loaded")
    
    print("All models loaded!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    print("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="MedSearch API",
    description="Multi-stage biomedical literature retrieval system",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranking: bool = True

class SearchResult(BaseModel):
    doc_id: str
    score: float
    title: str
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float
    method: str

def normalize_scores(results):
    if not results:
        return results
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [(doc_id, 0.5) for doc_id, _ in results]
    return [(doc_id, (s - min_s) / (max_s - min_s)) for doc_id, s in results]

def search_bm25(query, top_k=1000):
    tokens = tokenize(query)
    scores = models.bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(models.idx_to_doc_id[idx], float(scores[idx])) for idx in top_indices]

def search_dense(query, top_k=1000):
    query_emb = models.dense_model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    scores, indices = models.faiss_index.search(query_emb.astype(np.float32), top_k)
    return [(models.idx_to_doc_id[idx], float(score)) for idx, score in zip(indices[0], scores[0])]

def fuse_results(bm25_res, dense_res, alpha=0.6, top_k=100):
    bm25_norm = normalize_scores(bm25_res)
    dense_norm = normalize_scores(dense_res)
    
    bm25_scores = {doc_id: score for doc_id, score in bm25_norm}
    dense_scores = {doc_id: score for doc_id, score in dense_norm}
    
    all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
    
    fused = []
    for doc_id in all_docs:
        score = alpha * bm25_scores.get(doc_id, 0.0) + (1 - alpha) * dense_scores.get(doc_id, 0.0)
        fused.append((doc_id, score))
    
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]

def rerank(query, candidates, top_k=10):
    doc_ids = [doc_id for doc_id, _ in candidates]
    pairs = []
    for doc_id in doc_ids:
        row = models.corpus_df[models.corpus_df["doc_id"] == doc_id].iloc[0]
        doc_text = (row["title"] + " " + row["text"])[:512]
        pairs.append([query, doc_text])
    
    scores = models.cross_encoder.predict(pairs, show_progress_bar=False)
    results = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": models.bm25 is not None}

@app.get("/stats")
async def get_stats():
    return {
        "corpus_size": len(models.corpus_df) if models.corpus_df is not None else 0,
        "index_size_mb": 526.3
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if models.bm25 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = time.time()
    
    # Stage 1: Retrieve from both
    bm25_res = search_bm25(request.query, top_k=1000)
    dense_res = search_dense(request.query, top_k=1000)
    
    # Stage 2: Fuse
    fused = fuse_results(bm25_res, dense_res, alpha=0.6, top_k=100)
    
    # Stage 3: Rerank (optional)
    if request.use_reranking:
        final = rerank(request.query, fused, top_k=request.top_k)
        method = "BM25 + BioBERT + CrossEncoder"
    else:
        final = fused[:request.top_k]
        method = "BM25 + BioBERT Fusion"
    
    # Format results
    results = []
    for doc_id, score in final:
        row = models.corpus_df[models.corpus_df["doc_id"] == doc_id].iloc[0]
        results.append(SearchResult(
            doc_id=doc_id,
            score=float(score),
            title=row["title"],
            text=row["text"][:500]
        ))
    
    latency = (time.time() - start_time) * 1000
    
    return SearchResponse(results=results, latency_ms=latency, method=method)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)