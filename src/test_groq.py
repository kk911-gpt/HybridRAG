# test_groq.py
# Goal: confirm .env loads and Groq API key works

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads .env file

api_key = os.getenv("GROQ_API_KEY")
print("Key loaded:", api_key[:8] + "..." if api_key else "NOT FOUND")

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
response = llm.invoke("Say hello in one short sentence")
print("Response:", response.content)