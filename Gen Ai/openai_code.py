from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 0: Load API key
# -----------------------------
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check .env file")

print("API Key Loaded:", api_key[:5] + "*****")

client = OpenAI(api_key=api_key)

# -----------------------------
# Step 1: Documents (Banking)
# -----------------------------
docs = [
    "Transaction failed due to insufficient balance",
    "Payment declined due to security reasons",
    "Account blocked due to suspicious activity"
]

# -----------------------------
# Step 2: Create embeddings
# -----------------------------
response = client.embeddings.create(
    model="text-embedding-3-small", #dim 1536
    input=docs
)

doc_embeddings = [item.embedding for item in response.data]

print(" Document embeddings created")

# -----------------------------
# Step 3: User query
# -----------------------------
query = input("\nEnter your question: ")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_embedding = response.data[0].embedding

# -----------------------------
# Step 4: Similarity search
# -----------------------------
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

best_index = scores.argmax()
best_doc = docs[best_index]

print("\n Best matched context:", best_doc)

# -----------------------------
# Step 5: LLM Answer
# -----------------------------
prompt = f"""
You are a banking assistant.

Use only the given context.

Context: {best_doc}

Question: {query}

Answer:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

answer = response.choices[0].message.content

print("\n Final Answer:\n", answer)