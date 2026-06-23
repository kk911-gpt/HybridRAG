# embed_store.py
# Goal: embed all complaints from CSV and store in FAISS

import pandas as pd
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load your complaints dataset
df = pd.read_csv("data/complaints.csv")
complaints = df["complaint"].tolist()
print(f"Loaded {len(complaints)} complaints")

# Step 2: Load embedding model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Embed all complaints
print("Embedding all complaints...")
vectors = embedding_model.embed_documents(complaints)
vectors_np = np.array(vectors).astype("float32")
print(f"Embedded {len(vectors)} complaints, each vector size: {vectors_np.shape[1]}")

# Step 4: Store in FAISS
dimension = vectors_np.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)  # L2 = measures distance between vectors
index.add(vectors_np)
print(f"Stored in FAISS. Total vectors in index: {index.ntotal}")

# Step 5: Test a search
print("\n--- Testing similarity search ---")
query = "I never received my package"
query_vector = embedding_model.embed_query(query)
query_np = np.array([query_vector]).astype("float32")

# Search for top 3 most similar complaints
k = 3
distances, indices = index.search(query_np, k)

print(f"Query: '{query}'")
print(f"\nTop {k} most similar complaints:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {complaints[idx]}  (distance: {distances[0][i]:.4f})")