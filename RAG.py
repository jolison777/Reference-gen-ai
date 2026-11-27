# =====================================================
# END TO END RAG PIPELINE WITHOUT LANGCHAIN
# =====================================================

# pip install sentence-transformers
# pip install faiss-cpu
# pip install transformers
# pip install torch

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# =====================================================
# STEP 1 : DOCUMENT
# =====================================================

document = """
Snowflake is a cloud data warehouse.

Snowpipe enables continuous data ingestion from cloud storage.

Streams capture insert, update, and delete changes.

Tasks automate SQL execution on schedules.

DBT is used for data transformation.

FAISS is a vector database used for similarity search.

Embeddings convert text into vectors for semantic search.

Python is widely used for AI and machine learning.

RAG stands for Retrieval Augmented Generation.

RAG combines retrieval and LLM generation.
"""

print("Document Loaded")

# =====================================================
# STEP 2 : CHUNKING
# =====================================================

def chunk_text(text, chunk_size=200, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - overlap

    return chunks


chunks = chunk_text(document)

print("Chunks Created :", len(chunks))

# =====================================================
# STEP 3 : EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embeddings Created")

# =====================================================
# STEP 4 : FAISS VECTOR DATABASE
# =====================================================

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print("Vectors Stored In FAISS")

# =====================================================
# STEP 5 : LOAD OPEN SOURCE LLM
# =====================================================

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

print("LLM Loaded")

# =====================================================
# STEP 6 : QUERY LOOP
# =====================================================

while True:

    query = input("\nAsk Question (type exit to stop): ")

    if query.lower() == "exit":
        break

    # ==============================================
    # QUERY EMBEDDING
    # ==============================================

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # ==============================================
    # TOP K SEARCH
    # ==============================================

    top_k = 3

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    context = "\n".join(retrieved_chunks)

    print("\nRetrieved Context:")
    print(context)

    # ==============================================
    # PROMPT
    # ==============================================

    prompt = f"""
Use only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    # ==============================================
    # GENERATE RESPONSE
    # ==============================================

    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    print("\nFinal Answer:\n")

    print(response[0]["generated_text"])