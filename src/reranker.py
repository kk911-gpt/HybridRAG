# reranker.py
# Goal: understand how cross-encoder re-ranking works

from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import chromadb

print("Loading re-ranker model...")
# This is a small cross-encoder model specifically trained for re-ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Re-ranker loaded!")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_collection("complaints")
print(f"Connected! {collection.count()} complaints")

print("Loading BM25...")
df = pd.read_csv("data/complaints_processed.csv")
df = df[["narrative", "product"]].dropna()
df.columns = ["complaint", "category"]
df = df.drop_duplicates(subset="complaint")

balanced_parts = []
for cat in df["category"].unique():
    cat_df = df[df["category"] == cat]
    sampled = cat_df.sample(min(len(cat_df), 3000), random_state=42)
    balanced_parts.append(sampled)
df_balanced = pd.concat(balanced_parts).reset_index(drop=True)
complaints = df_balanced["complaint"].tolist()
categories = df_balanced["category"].tolist()
tokenized = [c.lower().split() for c in complaints]
bm25 = BM25Okapi(tokenized)
print("BM25 ready!")

# ---------- Hybrid search (returns Top 10 now) ----------
def hybrid_search(query, k=10):
    query_vector = embedding_model.embed_query(query)
    chroma_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k * 2
    )

    semantic_scores = {}
    for i, doc in enumerate(chroma_results["documents"][0]):
        distance = chroma_results["distances"][0][i]
        semantic_scores[doc] = 1 / (1 + distance)

    bm25_raw = bm25.get_scores(query.lower().split())
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1

    combined = {}
    for i, complaint in enumerate(complaints):
        bm25_score = bm25_raw[i] / bm25_max
        semantic_score = semantic_scores.get(complaint, 0)
        combined[complaint] = 0.5 * semantic_score + 0.5 * bm25_score

    top_complaints = sorted(combined, key=combined.get, reverse=True)[:k]

    results = []
    for complaint in top_complaints:
        idx = complaints.index(complaint)
        results.append({
            "complaint": complaint,
            "category": categories[idx],
            "hybrid_score": combined[complaint]
        })
    return results

# ---------- Re-rank function ----------
def rerank(query, candidates, top_k=3):
    # Cross-encoder scores each [query, candidate] pair together
    pairs = [[query, c["complaint"]] for c in candidates]
    scores = reranker.predict(pairs)

    # Add re-rank scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort by re-rank score instead of hybrid score
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

# ---------- Test ----------
query = "I was charged twice for the same credit card transaction"

print(f"\nQuery: '{query}'")

# Step 1: Get Top 10 from hybrid search
candidates = hybrid_search(query, k=10)
print("\n--- Top 10 from Hybrid Search ---")
for i, c in enumerate(candidates):
    print(f"{i+1}. [{c['category']}] {c['complaint'][:80]}...")
    print(f"   Hybrid score: {c['hybrid_score']:.4f}")

# Step 2: Re-rank those 10 to get best 3
print("\n--- Top 3 after Re-ranking ---")
reranked = rerank(query, candidates, top_k=3)
for i, c in enumerate(reranked):
    print(f"{i+1}. [{c['category']}] {c['complaint'][:80]}...")
    print(f"   Hybrid score: {c['hybrid_score']:.4f} → Rerank score: {c['rerank_score']:.4f}")

print("\nNotice how re-ranking may change the order compared to hybrid search alone!")