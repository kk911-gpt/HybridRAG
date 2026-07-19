# app.py v3
# Full agentic pipeline with LangGraph + Streamlit

import os
import random
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import streamlit as st

load_dotenv()

# LangSmith tracing — automatically traces all LLM calls
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ComplaintIQ")

st.set_page_config(
    page_title="ComplaintIQ",
    page_icon="🔍",
    layout="centered"
)

# ---------- State ----------
class ComplaintState(TypedDict):
    complaint: str
    category: str
    confidence: str
    reasoning: str
    action_taken: str
    reference_number: str
    final_response: str

# ---------- Pydantic ----------
class ClassificationOutput(BaseModel):
    category: Literal["credit_card", "retail_banking", "credit_reporting", "mortgages_and_loans", "debt_collection"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining the classification")

# ---------- Load pipeline (cached) ----------
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

    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
    structured_llm = llm.with_structured_output(ClassificationOutput)

    return embedding_model, collection, complaints, categories, bm25, llm, structured_llm

# ---------- Hybrid search ----------
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
        results.append({"complaint": complaint, "category": categories[idx], "score": combined[complaint]})
    return results

# ---------- Build LangGraph ----------
def build_graph(embedding_model, collection, complaints, categories, bm25, llm, structured_llm):

    def classifier_agent(state):
        top_results = hybrid_search(state["complaint"], embedding_model, collection, complaints, categories, bm25)
        context = ""
        for r in top_results:
            context += f"- Complaint: {r['complaint'][:150]}\n  Category: {r['category']}\n\n"
        prompt = f"""You are a financial complaint classifier.
Here are similar complaints:
{context}
Classify: {state['complaint']}
Choose from: credit_card, retail_banking, credit_reporting, mortgages_and_loans, debt_collection"""
        result = structured_llm.invoke(prompt)
        return {**state, "category": result.category, "confidence": result.confidence, "reasoning": result.reasoning}

    def action_agent(state):
        category = state["category"]
        ref = f"#{category[:3].upper()}-{random.randint(1000,9999)}"
        action_map = {
            "credit_card": "Escalated to Credit Card disputes team. Temporary credit applied.",
            "retail_banking": "Account flagged for review. Security team notified.",
            "credit_reporting": "Dispute filed with credit bureaus. Investigation initiated.",
            "mortgages_and_loans": "Loan file sent to mortgage review team for audit.",
            "debt_collection": "Complaint escalated to compliance team. Collector notified of FDCPA guidelines."
        }
        action = action_map.get(category, "Complaint logged and assigned to general support.")
        return {**state, "action_taken": action, "reference_number": ref}

    def response_agent(state):
        prompt = f"""You are a professional customer service agent.
Customer complaint: "{state['complaint']}"
Category: {state['category']}
Action taken: {state['action_taken']}
Reference: {state['reference_number']}
Write a professional empathetic response in 2-3 sentences mentioning the reference number."""
        response = llm.invoke(prompt)
        return {**state, "final_response": response.content}

    graph = StateGraph(ComplaintState)
    graph.add_node("classifier", classifier_agent)
    graph.add_node("action", action_agent)
    graph.add_node("response", response_agent)
    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "action")
    graph.add_edge("action", "response")
    graph.add_edge("response", END)
    return graph.compile()

# ---------- UI ----------
st.title("🔍 ComplaintIQ")
st.markdown("Agentic financial complaint classifier — 3 AI agents working in sequence.")
st.divider()

with st.spinner("Loading models..."):
    embedding_model, collection, complaints, categories, bm25, llm, structured_llm = load_pipeline()
    pipeline = build_graph(embedding_model, collection, complaints, categories, bm25, llm, structured_llm)

st.success(f"Ready — {collection.count():,} complaints in knowledge base")

complaint_input = st.text_area(
    "Enter a financial complaint:",
    placeholder="e.g. A debt collector keeps calling me at midnight",
    height=120
)

if st.button("Run Agent Pipeline", type="primary"):
    if not complaint_input.strip():
        st.warning("Please enter a complaint.")
    else:
        with st.spinner("Running 3 agents..."):
            initial_state = ComplaintState(
                complaint=complaint_input,
                category="", confidence="", reasoning="",
                action_taken="", reference_number="", final_response=""
            )
            result = pipeline.invoke(initial_state)

        st.divider()

        # Agent 1 results
        st.subheader("🤖 Agent 1 — Classifier")
        col1, col2 = st.columns(2)
        category_icons = {
            "credit_card": "💳", "retail_banking": "🏦",
            "credit_reporting": "📊", "mortgages_and_loans": "🏠",
            "debt_collection": "📞"
        }
        confidence_display = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}
        with col1:
            icon = category_icons.get(result['category'], "📌")
            st.metric("Category", f"{icon} {result['category'].replace('_',' ').title()}")
        with col2:
            st.metric("Confidence", confidence_display[result['confidence']])
        st.info(f"**Reasoning:** {result['reasoning']}")

        st.divider()

        # Agent 2 results
        st.subheader("⚡ Agent 2 — Action")
        st.success(f"**Action taken:** {result['action_taken']}")
        st.code(f"Reference Number: {result['reference_number']}")

        st.divider()

        # Agent 3 results
        st.subheader("💬 Agent 3 — Customer Response")
        st.write(result['final_response'])

        st.divider()

        # Retrieved context
        st.subheader("📚 Retrieved Context")
        top_results = hybrid_search(complaint_input, embedding_model, collection, complaints, categories, bm25)
        for i, r in enumerate(top_results):
            with st.expander(f"Match {i+1} — [{r['category']}] (score: {r['score']:.4f})"):
                st.write(r['complaint'][:300] + "...")