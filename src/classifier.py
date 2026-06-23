# classifier.py
# Goal: classify complaints with structured Pydantic output

import os
import pandas as pd
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ---------- Pydantic schema ----------
# This defines exactly what shape the LLM output must take
class ComplaintClassification(BaseModel):
    category: Literal["Shipping", "Billing", "Refund/Return", "Technical"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining why this category was chosen")

# ---------- Load data ----------
df = pd.read_csv("data/complaints.csv")
complaints = df["complaint"].tolist()
categories = df["category"].tolist()

# ---------- Build FAISS index ----------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embedding_model.embed_documents(complaints)
vectors_np = np.array(vectors).astype("float32")
faiss_index = faiss.IndexFlatL2(vectors_np.shape[1])
faiss_index.add(vectors_np)

# ---------- Build BM25 index ----------
tokenized = [c.lower().split() for c in complaints]
bm25 = BM25Okapi(tokenized)

# ---------- Hybrid search ----------
def hybrid_search(query, k=3):
    query_vector = embedding_model.embed_query(query)
    query_np = np.array([query_vector]).astype("float32")
    distances, faiss_indices = faiss_index.search(query_np, len(complaints))

    faiss_scores = {}
    for rank, idx in enumerate(faiss_indices[0]):
        faiss_scores[idx] = 1 / (1 + distances[0][rank])

    bm25_raw = bm25.get_scores(query.lower().split())
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {i: bm25_raw[i] / bm25_max for i in range(len(complaints))}

    combined = {}
    for i in range(len(complaints)):
        combined[i] = 0.5 * faiss_scores.get(i, 0) + 0.5 * bm25_scores.get(i, 0)

    return sorted(combined, key=combined.get, reverse=True)[:k]

# ---------- Classify with structured output ----------
def classify_complaint(query):
    top_indices = hybrid_search(query, k=3)

    context = ""
    for idx in top_indices:
        context += f"- Complaint: {complaints[idx]}\n"
        context += f"  Category: {categories[idx]}\n\n"

    prompt = f"""You are a customer complaint classifier.

Here are some example complaints and their categories:
{context}
Now classify this new complaint:
Complaint: {query}

Choose from: Shipping, Billing, Refund/Return, Technical."""

    # This forces LLM to return output matching ComplaintClassification schema
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
    structured_llm = llm.with_structured_output(ComplaintClassification)
    result = structured_llm.invoke(prompt)

    return result

# ---------- Test ----------
test_queries = [
    "I never received my package",
    "I was billed twice this month",
    "The app crashes every time I open it",
    "I want to return this broken item"
]

print("=" * 50)
for query in test_queries:
    result = classify_complaint(query)
    print(f"Complaint:  {query}")
    print(f"Category:   {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning:  {result.reasoning}")
    print("=" * 50)