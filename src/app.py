# app.py v2
# Full pipeline: ChromaDB + BM25 + Groq + Pydantic + Streamlit

import os
import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="ComplaintIQ",
    page_icon="🔍",
    layout="centered"
)

class ComplaintClassification(BaseModel):
    category: Literal["credit_card", "retail_banking", "credit_reporting", "mortgages_and_loans", "debt_collection"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining why this category was chosen")

@st.cache_resource
def load_pipeline():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path="./chromadb_store")
    collection = client.get_collection("complaints")

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

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(ComplaintClassification)

    return embedding_model, collection, complaints, categories, bm25, structured_llm

def hybrid_search(query, embedding_model, collection, complaints, categories, bm25, k=3):
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
            "category": categories[idx],
            "score": combined[complaint]
        })
    return results

# ---------- UI ----------
st.title("🔍 ComplaintIQ")
st.markdown("Financial complaint classifier powered by Hybrid RAG — 15,000 real CFPB complaints.")
st.divider()

with st.spinner("Loading models and connecting to ChromaDB..."):
    embedding_model, collection, complaints, categories, bm25, structured_llm = load_pipeline()

st.success(f"Ready — {collection.count():,} complaints in knowledge base")

complaint_input = st.text_area(
    "Enter a financial complaint:",
    placeholder="e.g. A debt collector keeps calling me at odd hours",
    height=120
)

if st.button("Classify Complaint", type="primary"):
    if not complaint_input.strip():
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Searching knowledge base and classifying..."):
            top_results = hybrid_search(
                complaint_input,
                embedding_model, collection,
                complaints, categories, bm25, k=3
            )

            context = ""
            for r in top_results:
                context += f"- Complaint: {r['complaint'][:150]}\n  Category: {r['category']}\n\n"

            prompt = f"""You are a financial complaint classifier.

Here are similar complaints and their categories:
{context}
Now classify this complaint:
Complaint: {complaint_input}

Choose from: credit_card, retail_banking, credit_reporting, mortgages_and_loans, debt_collection"""

            result = structured_llm.invoke(prompt)

        st.divider()

        # Results
        col1, col2 = st.columns(2)
        category_icons = {
            "credit_card": "💳",
            "retail_banking": "🏦",
            "credit_reporting": "📊",
            "mortgages_and_loans": "🏠",
            "debt_collection": "📞"
        }
        confidence_display = {
            "high": "🟢 High",
            "medium": "🟡 Medium",
            "low": "🔴 Low"
        }

        with col1:
            icon = category_icons.get(result.category, "📌")
            st.metric("Category", f"{icon} {result.category.replace('_', ' ').title()}")

        with col2:
            st.metric("Confidence", confidence_display[result.confidence])

        st.info(f"**Reasoning:** {result.reasoning}")

        st.divider()
        st.subheader("📚 Retrieved Context")
        for i, r in enumerate(top_results):
            with st.expander(f"Match {i+1} — [{r['category']}] (score: {r['score']:.4f})"):
                st.write(r['complaint'][:300] + "...")