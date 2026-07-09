# hybrid_retriever_v2.py
# Goal: hybrid search using ChromaDB (15k real complaints) + BM25

import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb_store")
collection = client.get_collection("complaints")
print(f"Connected! {collection.count()} complaints in store")

print("Loading complaints for BM25...")
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
print(f"BM25 ready on {len(complaints)} complaints")

def hybrid_search(query, k=5):
    # ChromaDB semantic search
    query_vector = embedding_model.embed_query(query)
    chroma_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k * 3
    )

    semantic_scores = {}
    for i, doc in enumerate(chroma_results["documents"][0]):
        distance = chroma_results["distances"][0][i]
        semantic_scores[doc] = 1 / (1 + distance)

    # BM25 keyword search
    bm25_raw = bm25.get_scores(query.lower().split())
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1

    # Combine scores
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
            "complaint": complaint[:100],
            "category": categories[idx],
            "score": combined[complaint]
        })

    return results

# Test
query = "I have a problem with my credit card payment"
print(f"\nQuery: '{query}'")
print("\n--- Hybrid Top 5 Results ---")
results = hybrid_search(query, k=5)
for i, r in enumerate(results):
    print(f"{i+1}. [{r['category']}] {r['complaint']}...")
    print(f"   Score: {r['score']:.4f}")