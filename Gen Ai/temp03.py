from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# documents
docs = [
    "Transaction failed due to insufficient balance.",
    "Payment declined due to security reasons.",
    "Account blocked due to suspicious activity."
]

# embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(docs)

# query
query = "Why did my payment fail?"
query_embedding = embed_model.encode([query])

# similarity search
scores = cosine_similarity(query_embedding, doc_embeddings)[0]
best_doc = docs[scores.argmax()]

print("Retrieved Context:", best_doc)

# LLM generation
generator = pipeline("text-generation", model="gpt2")

def generate_answer(temp):
    prompt = f"Context: {best_doc}\nAnswer:"
    
    result = generator(
        prompt,
        max_new_tokens=40,
        temperature=temp,
        
    )
    
    print(f"\nTemperature = {temp}")
    print(result[0]['generated_text'])

#generate_answer(0.2)
generate_answer(0.2)