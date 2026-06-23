# app.py
# Goal: Streamlit UI for HybridRAG complaint classifier

import os
import sys
import pandas as pd
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st

load_dotenv()

# ---------- Page config ----------
st.set_page_config(
    page_title="HybridRAG Complaint Classifier",
    page_icon="🔍",
    layout="centered"
)

# ---------- Pydantic schema ----------
class ComplaintClassification(BaseModel):
    category: Literal["Shipping", "Billing", "Refund/Return", "Technical"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining why this category was chosen")

# ---------- Load everything once (cached so it doesn't reload on every click) ----------
@st.cache_resource
def load_pipeline():
    df = pd.read_csv("data/complaints.csv")
    complaints = df["complaint"].tolist()
    categories = df["category"].tolist()

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = embedding_model.embed_documents(complaints)
    vectors_np = np.array(vectors).astype("float32")
    faiss_index = faiss.IndexFlatL2(vectors_np.shape[1])
    faiss_index.add(vectors_np)

    tokenized = [c.lower().split() for c in complaints]
    bm25 = BM25Okapi(tokenized)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(ComplaintClassification)

    return complaints, categories, embedding_model, faiss_index, bm25, structured_llm

# ---------- Hybrid search ----------
def hybrid_search(query, complaints, embedding_model, faiss_index, bm25, k=3):
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

    return sorted(combined, key=combined.get, reverse=True)[:k], combined

# ---------- UI ----------
st.title("🔍 HybridRAG Complaint Classifier")
st.markdown("Classifies customer complaints using hybrid retrieval (FAISS + BM25) and structured LLM output.")
st.divider()

# Load pipeline
with st.spinner("Loading models..."):
    complaints, categories, embedding_model, faiss_index, bm25, structured_llm = load_pipeline()

# Input
complaint_input = st.text_area(
    "Enter a customer complaint:",
    placeholder="e.g. I was charged twice for my order last week",
    height=100
)

if st.button("Classify Complaint", type="primary"):
    if not complaint_input.strip():
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Retrieving and classifying..."):

            # Hybrid search
            top_indices, combined_scores = hybrid_search(
                complaint_input, complaints, embedding_model, faiss_index, bm25, k=3
            )

            # LLM classification
            context = ""
            for idx in top_indices:
                context += f"- Complaint: {complaints[idx]}\n  Category: {categories[idx]}\n\n"

            prompt = f"""You are a customer complaint classifier.

Here are some example complaints and their categories:
{context}
Now classify this new complaint:
Complaint: {complaint_input}

Choose from: Shipping, Billing, Refund/Return, Technical."""

            result = structured_llm.invoke(prompt)

        # ---------- Results ----------
        st.divider()

        # Category + confidence
        col1, col2 = st.columns(2)

        with col1:
            color_map = {
                "Shipping": "🚚",
                "Billing": "💳",
                "Refund/Return": "↩️",
                "Technical": "⚙️"
            }
            icon = color_map.get(result.category, "📌")
            st.metric(label="Category", value=f"{icon} {result.category}")

        with col2:
            confidence_color = {
                "high": "🟢 High",
                "medium": "🟡 Medium",
                "low": "🔴 Low"
            }
            st.metric(label="Confidence", value=confidence_color[result.confidence])

        # Reasoning
        st.info(f"**Reasoning:** {result.reasoning}")

        # Retrieved context
        st.divider()
        st.subheader("📚 Retrieved Context (what influenced this decision)")

        for i, idx in enumerate(top_indices):
            with st.expander(f"Match {i+1}: {complaints[idx]}"):
                st.write(f"**Category:** {categories[idx]}")
                st.write(f"**Hybrid Score:** {combined_scores[idx]:.4f}")