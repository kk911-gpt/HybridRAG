# keyword_search.py
# Goal: search complaints using keywords (BM25), not meaning

import pandas as pd
from rank_bm25 import BM25Okapi

# Step 1: Load complaints
df = pd.read_csv("data/complaints.csv")
complaints = df["complaint"].tolist()
print(f"Loaded {len(complaints)} complaints")

# Step 2: Tokenize (split each complaint into individual words)
tokenized = [c.lower().split() for c in complaints]

# Step 3: Build BM25 index
bm25 = BM25Okapi(tokenized)
print("BM25 index built")

# Step 4: Search
query = "I never received my package"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)

# Step 5: Get top 3 results
import numpy as np
top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]

print(f"\nQuery: '{query}'")
print(f"\nTop {top_k} keyword matches:")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {complaints[idx]}  (score: {scores[idx]:.4f})")