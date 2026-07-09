# classifier.py v2
# Goal: full pipeline with ChromaDB + BM25 + Groq + Pydantic

import os
import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ---------- Pydantic schema ----------
class ComplaintClassification(BaseModel):
    category: Literal["credit_card", "retail_banking", "credit_reporting", "mortgages_and_loans", "debt_collection"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining why this category was chosen")

# ---------- Load everything ----------
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
print(f"BM25 ready on {len(complaints)} complaints")

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)
structured_llm = llm.with_structured_output(ComplaintClassification)

# ---------- Hybrid search ----------
def hybrid_search(query, k=5):
    query_vector = embedding_model.embed_query(query)
    chroma_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k * 3
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
            "category": categories[idx]
        })
    return results

# ---------- Classify ----------
def classify_complaint(query):
    top_results = hybrid_search(query, k=3)

    context = ""
    for r in top_results:
        context += f"- Complaint: {r['complaint'][:150]}\n"
        context += f"  Category: {r['category']}\n\n"

    prompt = f"""You are a financial complaint classifier.

Here are similar complaints and their categories:
{context}
Now classify this complaint:
Complaint: {query}

Choose from: credit_card, retail_banking, credit_reporting, mortgages_and_loans, debt_collection"""

    result = structured_llm.invoke(prompt)
    return result

# ---------- Test ----------
test_queries = [
    "I have a problem with my credit card payment",
    "My mortgage payment was wrongly calculated",
    "A debt collector keeps calling me at midnight",
    "Wrong information on my credit report",
    "My bank account was charged without my permission"
]

print("\n" + "="*50)
for query in test_queries:
    result = classify_complaint(query)
    print(f"Complaint:  {query}")
    print(f"Category:   {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning:  {result.reasoning}")
    print("="*50)