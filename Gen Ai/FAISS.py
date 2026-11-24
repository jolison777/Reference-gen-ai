from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: documents
docs = [
    "Transaction failed due to insufficient balance",
    "Payment declined due to security reasons",
    "Account blocked due to suspicious activity"
]

# Step 2: load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: create embeddings
doc_embeddings = model.encode(docs)

print(doc_embeddings)

# Step 4: create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print(index)

# Step 5: add vectors
index.add(np.array(doc_embeddings))

# Step 6: query
query = "Why did my payment fail?"
query_vector = model.encode([query])

# Step 7: search top 2 results
k = 2
distances, indices = index.search(query_vector, k)

# Step 8: print results
for i in indices[0]:
    print(docs[i])