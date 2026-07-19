# agent_pipeline.py
# Goal: multi-agent pipeline using LangGraph

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

load_dotenv()

# LangSmith tracing — automatically traces all LLM calls
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ComplaintIQ")

# ============================================================
# STEP 1: Define the State
# This is the tray that carries data between all agents
# ============================================================
class ComplaintState(TypedDict):
    complaint: str                    # original complaint
    category: str                     # filled by Classifier Agent
    confidence: str                   # filled by Classifier Agent
    reasoning: str                    # filled by Classifier Agent
    action_taken: str                 # filled by Action Agent
    reference_number: str             # filled by Action Agent
    final_response: str               # filled by Response Agent

# ============================================================
# STEP 2: Pydantic schema for structured LLM output
# ============================================================
class ClassificationOutput(BaseModel):
    category: Literal["credit_card", "retail_banking", "credit_reporting", "mortgages_and_loans", "debt_collection"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="One sentence explaining the classification")

# ============================================================
# STEP 3: Load shared resources (used by Classifier Agent)
# ============================================================
print("Loading resources...")
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
print("Resources loaded!")

# ============================================================
# STEP 4: Hybrid search (same as before)
# ============================================================
def hybrid_search(query, k=3):
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
        results.append({"complaint": complaint, "category": categories[idx]})
    return results

# ============================================================
# STEP 5: Agent 1 — Classifier Agent
# ============================================================
def classifier_agent(state: ComplaintState) -> ComplaintState:
    print("\n[Agent 1: Classifier] Running...")
    query = state["complaint"]

    top_results = hybrid_search(query, k=3)
    context = ""
    for r in top_results:
        context += f"- Complaint: {r['complaint'][:150]}\n  Category: {r['category']}\n\n"

    prompt = f"""You are a financial complaint classifier.

Here are similar complaints and their categories:
{context}
Now classify this complaint:
Complaint: {query}

Choose from: credit_card, retail_banking, credit_reporting, mortgages_and_loans, debt_collection"""

    result = structured_llm.invoke(prompt)
    print(f"[Agent 1: Classifier] Category: {result.category} ({result.confidence})")

    return {
        **state,
        "category": result.category,
        "confidence": result.confidence,
        "reasoning": result.reasoning
    }

# ============================================================
# STEP 6: Agent 2 — Action Agent (mock tool calls)
# ============================================================
def action_agent(state: ComplaintState) -> ComplaintState:
    print("\n[Agent 2: Action] Running...")
    category = state["category"]
    ref = f"#{category[:3].upper()}-{random.randint(1000, 9999)}"

    action_map = {
        "credit_card": "Escalated to Credit Card disputes team. Temporary credit applied.",
        "retail_banking": "Account flagged for review. Security team notified.",
        "credit_reporting": "Dispute filed with credit bureaus. Investigation initiated.",
        "mortgages_and_loans": "Loan file sent to mortgage review team for audit.",
        "debt_collection": "Complaint escalated to compliance team. Collector notified of FDCPA guidelines."
    }

    action = action_map.get(category, "Complaint logged and assigned to general support.")
    print(f"[Agent 2: Action] {action} | Ref: {ref}")

    return {
        **state,
        "action_taken": action,
        "reference_number": ref
    }

# ============================================================
# STEP 7: Agent 3 — Response Agent
# ============================================================
def response_agent(state: ComplaintState) -> ComplaintState:
    print("\n[Agent 3: Response] Running...")

    prompt = f"""You are a professional customer service agent for a financial institution.

A customer submitted this complaint:
"{state['complaint']}"

It was classified as: {state['category']}
Action taken: {state['action_taken']}
Reference number: {state['reference_number']}

Write a professional, empathetic response to the customer in 2-3 sentences.
Mention the reference number and what was done."""

    response = llm.invoke(prompt)
    print(f"[Agent 3: Response] Generated customer reply")

    return {
        **state,
        "final_response": response.content
    }

# ============================================================
# STEP 8: Build the LangGraph
# ============================================================
def build_graph():
    graph = StateGraph(ComplaintState)

    # Add nodes
    graph.add_node("classifier", classifier_agent)
    graph.add_node("action", action_agent)
    graph.add_node("response", response_agent)

    # Add edges (flow)
    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "action")
    graph.add_edge("action", "response")
    graph.add_edge("response", END)

    return graph.compile()

# ============================================================
# STEP 9: Test it
# ============================================================
pipeline = build_graph()

test_complaint = "A debt collector keeps calling me at midnight demanding money I don't owe"

print("\n" + "="*60)
print(f"COMPLAINT: {test_complaint}")
print("="*60)

initial_state = ComplaintState(
    complaint=test_complaint,
    category="",
    confidence="",
    reasoning="",
    action_taken="",
    reference_number="",
    final_response=""
)

result = pipeline.invoke(initial_state)

print("\n" + "="*60)
print("FINAL RESULT:")
print(f"Category:   {result['category']} ({result['confidence']})")
print(f"Reasoning:  {result['reasoning']}")
print(f"Action:     {result['action_taken']}")
print(f"Reference:  {result['reference_number']}")
print(f"\nCustomer Response:\n{result['final_response']}")
print("="*60)