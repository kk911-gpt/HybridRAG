# load_chromadb.py
# Goal: load real CFPB complaints into ChromaDB (balanced sample)

import pandas as pd
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- Step 1: Load and clean ----------
print("Loading dataset...")
df = pd.read_csv("data/complaints_processed.csv")
df = df[["narrative", "product"]].dropna()
df.columns = ["complaint", "category"]
df = df.drop_duplicates(subset="complaint")
print(f"Full dataset: {len(df)} complaints")

# ---------- Step 2: Balance the dataset ----------
SAMPLES_PER_CATEGORY = 3000
balanced_parts = []
for cat in df["category"].unique():
    cat_df = df[df["category"] == cat]
    sampled = cat_df.sample(min(len(cat_df), SAMPLES_PER_CATEGORY), random_state=42)
    balanced_parts.append(sampled)

df_balanced = pd.concat(balanced_parts).reset_index(drop=True)
print(f"Balanced dataset: {len(df_balanced)} complaints")
print(df_balanced["category"].value_counts())
# ---------- Step 3: Load embedding model ----------
print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Step 4: Set up ChromaDB ----------
print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb_store")

try:
    client.delete_collection("complaints")
    print("Cleared old collection")
except:
    pass

collection = client.create_collection("complaints")
print("Created fresh collection")

# ---------- Step 5: Load in batches ----------
complaints = df_balanced["complaint"].tolist()
categories = df_balanced["category"].tolist()

BATCH_SIZE = 500
total = len(complaints)
print(f"\nLoading {total} complaints in batches of {BATCH_SIZE}...")

for i in range(0, total, BATCH_SIZE):
    batch_complaints = complaints[i:i+BATCH_SIZE]
    batch_categories = categories[i:i+BATCH_SIZE]
    batch_vectors = embedding_model.embed_documents(batch_complaints)

    collection.add(
        ids=[str(i+j) for j in range(len(batch_complaints))],
        embeddings=batch_vectors,
        documents=batch_complaints,
        metadatas=[{"category": cat} for cat in batch_categories]
    )

    print(f"Batch {i//BATCH_SIZE + 1}/{(total//BATCH_SIZE)+1} done — {min(i+BATCH_SIZE, total)}/{total} loaded")

print(f"\nDone! Total in ChromaDB: {collection.count()}")
print("Saved permanently to ./chromadb_store")