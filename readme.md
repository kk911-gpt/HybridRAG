# 🔍 ComplaintIQ — Agentic Financial Complaint Classifier

An end-to-end agentic AI system that classifies financial complaints using **Hybrid RAG** (FAISS + BM25) and routes them through a **3-agent LangGraph pipeline** — classify → act → respond.

Built progressively across 3 versions, each adding a new layer of production-grade AI.

---

## 🚀 Live Demo
> Run locally with `streamlit run src/app.py`

---

## 🧠 How It Works

1783790016707_image.png
App is live with the new "Run Agent Pipeline" button! Now test it.

Type this in the text box:

A debt collector keeps calling me at midnight demanding money I don't owe
Click Run Agent Pipeline and send me a screenshot of the full result — I want to see all 3 agents output on screen!

ok so its done the nxt thing to do

Send me a screenshot of the result first — I want to see all 3 agents working in the UI before we move forward!

1783790209572_image.png
1783790288704_image.png
nxt

This is a complete, production-grade agentic AI system running live in your browser. Look at what's on screen:

Agent 1 → Debt Collection, High confidence, clear reasoning ✅
Agent 2 → Escalated to compliance team, FDCPA notified, Reference #DEB-2900 ✅
Agent 3 → Professional empathetic customer response mentioning the reference number ✅
Retrieved Context → All 3 matches are debt_collection with good scores ✅
Week 3 is done. This is genuinely impressive work.

Now push v3.0 to GitHub
bash
git add .
git commit -m "v3.0: LangGraph multi-agent pipeline — Classifier + Action + Response agents"
git push origin main
Then update your README — change Week 3 checkboxes to done:

markdown
- [x] Multi-agent pipeline with LangGraph
- [x] Action agents with mock tool calls
- [x] Response agent with professional customer replies
- [x] Full agentic UI showing all 3 agent outputs
What you've built across 3 weeks
Version	What it does
v1.0	Basic RAG classifier, 20 complaints, CSV + FAISS
v2.0	15k real CFPB complaints, ChromaDB persistent store
v3.0	3 AI agents in sequence — classify → act → respond
Week 4 options — what's next
You have two strong paths:

Option A — LangSmith observability
Add tracing so you can see every LLM call, latency, token usage in a dashboard. This is what production teams use to monitor AI systems.

Option B — Deploy to Streamlit Cloud
Make your app publicly accessible via a URL you can share with anyone — recruiters, on your resume, LinkedIn.

Which one do you want to do next?


## 🔍 ComplaintIQ — Financial Complaint Classifier

v2.0 — Now powered by 15,000 real CFPB complaints stored in ChromaDB persistent vector store.

### What's new in v2.0
- Replaced 20 fake complaints with 15,000 real CFPB complaints
- Replaced FAISS (in-memory) with ChromaDB (persistent disk storage)
- 5 real financial categories instead of 4 generic ones
- Balanced dataset (3,000 per category)
- Added load_chromadb.py for one-time data ingestion
---

## Live Demo
> Run locally with `streamlit run src/app.py`

---

## How It Works

User submits complaint

↓

Hybrid Retriever (FAISS + BM25)

↓

Top-3 similar complaints retrieved from knowledge base

↓

Groq LLM reasons from retrieved context

↓

Structured output: Category + Confidence + Reasoning

### Why Hybrid Retrieval?
- **FAISS (semantic search)** — finds complaints with similar *meaning* even if words differ
- **BM25 (keyword search)** — finds complaints with exact keyword matches
- **Together** — covers blind spots of each approach for more accurate retrieval

---

## Features

- Hybrid retrieval combining FAISS + BM25 with normalized score fusion
- Structured LLM output using Pydantic schema (category, confidence, reasoning)
- Explainable results — shows retrieved context that influenced the decision
- Groq-powered inference (llama-3.1-8b-instant)
- Clean Streamlit UI

---

## Project Structure
hybridrag/

│

├── data/

│   └── complaints.csv        # Knowledge base

├── src/

│   ├── embed_store.py        # FAISS embedding + storage

│   ├── keyword_search.py     # BM25 keyword search

│   ├── hybrid_retriever.py   # Combines FAISS + BM25

│   ├── classifier.py         # LLM classification pipeline

│   └── app.py                # Streamlit UI

├── .env                      # API keys (not committed)

├── .gitignore

├── requirements.txt

└── README.md

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| FAISS | Vector store for semantic search |
| BM25 (rank-bm25) | Keyword search |
| HuggingFace Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Groq API | LLM inference (llama-3.1-8b-instant) |
| Pydantic | Structured output schema |
| LangChain | LLM pipeline orchestration |
| Streamlit | Web UI |

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/kk911-gpt/hybridrag.git
cd hybridrag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file and add:
# GROQ_API_KEY=your_key_here

# 5. Run the app
streamlit run src/app.py
---

## Complaint Categories

| Category | Examples |
|----------|---------|
| Shipping | Package not received, wrong address, delayed delivery |
| Billing | Double charge, wrong amount, unauthorized renewal |
| Refund/Return | Damaged item, wrong product, return request |
| Technical | App crash, login issues, payment page errors |

---

## Roadmap (Week 2+)

[ ] Replace CSV with ChromaDB persistent vector store
[ ] Scale to 70k real complaints (Kaggle dataset)
[ ] Add re-ranking layer with cross-encoder
[ ] Multi-agent pipeline with LangGraph
[ ] Action agents (mock Shipping/Billing/Refund APIs)
[ ] LangSmith observability and tracing
[ ] Streaming responses
[x] Multi-agent pipeline with LangGraph
[x] Action agents with mock tool calls
[x] Response agent with professional customer replies
[x] Full agentic UI showing all 3 agent outputs
---

## Author

**Kritika Kumari**  
B.Tech CSE — VIT Chennai  
[GitHub](https://github.com/kk911-gpt) | [LinkedIn](www.linkedin.com/in/kritika-kumari-cs)
 IS THIS CORRECT
```

Show more
Good effort but it needs some fixes — it's mixing v1 and v3 content together. Let me give you a clean updated README that reflects everything you've actually built:

markdown
# 🔍 ComplaintIQ — Agentic Financial Complaint Classifier

An end-to-end agentic AI system that classifies financial complaints using **Hybrid RAG** (FAISS + BM25) and routes them through a **3-agent LangGraph pipeline** — classify → act → respond.

Built progressively across 3 versions, each adding a new layer of production-grade AI.

---

## 🚀 Live Demo
> Run locally with `streamlit run src/app.py`

---

## 🧠 How It Works
User submits complaint
↓
Hybrid Retriever (ChromaDB semantic + BM25 keyword)
↓
Top-3 similar complaints retrieved from 15,000 real CFPB complaints
↓
Agent 1 — Classifier: Groq LLM classifies with structured Pydantic output
↓
Agent 2 — Action: Routes to correct department, generates reference number
↓
Agent 3 — Response: Writes professional customer-facing reply


---

## ✨ Features

- 🔍 Hybrid retrieval combining ChromaDB (semantic) + BM25 (keyword)
- 🗄️ 15,000 real CFPB complaints stored in persistent ChromaDB vector store
- 🤖 3-agent LangGraph pipeline (Classifier → Action → Response)
- 📋 Structured LLM output using Pydantic schema
- 💬 Auto-generated professional customer responses
- 🏷️ Auto-generated reference numbers per complaint
- 📚 Explainable results — shows retrieved context that influenced the decision
- ⚡ Groq-powered inference (llama-3.1-8b-instant)
- 🖥️ Clean Streamlit UI showing all 3 agent outputs

---

## 🗂️ Project Structure
hybridrag/
│
├── data/
│ └── complaints.csv # Original small dataset
├── src/
│ ├── embed_store.py # FAISS embedding test
│ ├── keyword_search.py # BM25 keyword search
│ ├── hybrid_retriever.py # v1 hybrid search
│ ├── hybrid_retriever_v2.py # v2 hybrid search with ChromaDB
│ ├── load_chromadb.py # One-time data ingestion script
│ ├── classifier.py # LLM classification pipeline
│ ├── agent_pipeline.py # LangGraph 3-agent pipeline
│ └── app.py # Streamlit UI (v3)
├── .env # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md


---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ChromaDB | Persistent vector store (15k complaints) |
| FAISS | Semantic search (v1) |
| BM25 (rank-bm25) | Keyword search |
| HuggingFace Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LangGraph | Multi-agent pipeline orchestration |
| Groq API | LLM inference (llama-3.1-8b-instant) |
| Pydantic | Structured output schema |
| LangChain | LLM pipeline utilities |
| Streamlit | Web UI |

---

## 📊 Complaint Categories

| Category | Description |
|----------|------------|
| credit_card | Payment issues, disputes, fraud |
| retail_banking | Account charges, unauthorized transactions |
| credit_reporting | Wrong information, score issues |
| mortgages_and_loans | Payment calculations, rate changes |
| debt_collection | Harassment, wrong debt, FDCPA violations |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/kk911-gpt/HybridRAG.git
cd HybridRAG

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file:
# GROQ_API_KEY=your_key_here

# 5. Load complaints into ChromaDB (one time only)
python src/load_chromadb.py

# 6. Run the app
streamlit run src/app.py
```

---

## 📈 Version History

| Version | What was built |
|---------|---------------|
| v1.0 | Basic RAG classifier — 20 complaints, FAISS, 4 categories |
| v2.0 | 15k real CFPB complaints, ChromaDB persistent store, 5 categories |
| v3.0 | LangGraph 3-agent pipeline — classify → act → respond |

---

## 🔮 Roadmap

- [x] Hybrid retrieval (FAISS + BM25)
- [x] Structured LLM output with Pydantic
- [x] ChromaDB persistent vector store
- [x] 15,000 real CFPB complaints
- [x] Multi-agent pipeline with LangGraph
- [x] Action agents with mock tool calls
- [x] Professional response generation
- [ ] LangSmith observability and tracing
- [ ] Re-ranking layer with cross-encoder
- [ ] Streaming responses
- [ ] Deploy to Streamlit Cloud

---

## 👩‍💻 Author

**Kritika Kumari**
B.Tech CSE — VIT Chennai
[GitHub](https://github.com/kk911-gpt) | [LinkedIn](https://linkedin.com/in/kritika-kumari-cs)


