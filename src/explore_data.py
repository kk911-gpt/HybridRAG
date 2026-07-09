# explore_data.py
# Goal: understand the structure of our new dataset before using it

import pandas as pd

df = pd.read_csv("data/customer_support_tickets.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nCategory counts:")
print(df.iloc[:, -1].value_counts().head(10))
print("\nMissing values:")
print(df.isnull().sum())