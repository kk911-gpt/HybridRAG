# hybrid_retriever.py
# Goal: combine FAISS (semantic) + BM25 (keyword) scores into one Top-K result

import pandas as pd
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- Load data ----------
df = pd.read_csv("data/complaints.csv")
complaints = df["complaint"].tolist()
print(f"Loaded {len(complaints)} complaints")

# ---------- Build FAISS index ----------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embedding_model.embed_documents(complaints)
vectors_np = np.array(vectors).astype("float32")
dimension = vectors_np.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(vectors_np)
print("FAISS index ready")

# ---------- Build BM25 index ----------
tokenized = [c.lower().split() for c in complaints]
bm25 = BM25Okapi(tokenized)
print("BM25 index ready")

# ---------- Hybrid search function ----------
def hybrid_search(query, k=3):
    # --- FAISS search ---
    query_vector = embedding_model.embed_query(query)
    query_np = np.array([query_vector]).astype("float32")
    distances, faiss_indices = faiss_index.search(query_np, len(complaints))

    # Convert FAISS distances to scores (smaller distance = higher score)
    faiss_scores = {}
    for rank, idx in enumerate(faiss_indices[0]):
        faiss_scores[idx] = 1 / (1 + distances[0][rank])  # flip so higher = better

    # --- BM25 search ---
    tokenized_query = query.lower().split()
    bm25_raw = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to 0-1 range so they're comparable to FAISS
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {i: bm25_raw[i] / bm25_max for i in range(len(complaints))}

    # --- Combine scores (equal weight: 50% FAISS + 50% BM25) ---
    combined = {}
    for i in range(len(complaints)):
        combined[i] = 0.5 * faiss_scores.get(i, 0) + 0.5 * bm25_scores.get(i, 0)

    # --- Get Top-K ---
    top_k_indices = sorted(combined, key=combined.get, reverse=True)[:k]

    return top_k_indices, combined

# ---------- Test it ----------
query = "I never received my package"
print(f"\nQuery: '{query}'")
print("\n--- Hybrid Top 3 Results ---")

top_indices, scores = hybrid_search(query, k=3)
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {complaints[idx]}  (hybrid score: {scores[idx]:.4f})")