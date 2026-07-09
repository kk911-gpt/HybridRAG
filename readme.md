
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
```

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

- [ ] Replace CSV with ChromaDB persistent vector store
- [ ] Scale to 70k real complaints (Kaggle dataset)
- [ ] Add re-ranking layer with cross-encoder
- [ ] Multi-agent pipeline with LangGraph
- [ ] Action agents (mock Shipping/Billing/Refund APIs)
- [ ] LangSmith observability and tracing
- [ ] Streaming responses

---

## Author

**Kritika Kumari**  
B.Tech CSE — VIT Chennai  
[GitHub](https://github.com/kk911-gpt) | [LinkedIn](www.linkedin.com/in/kritika-kumari-cs)
