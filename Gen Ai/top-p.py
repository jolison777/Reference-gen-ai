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

def generate_answer(temp, top_p_val):
    prompt = f"Context: {best_doc}\nAnswer:"
    
    result = generator(
        prompt,
        max_new_tokens=40,
        temperature=temp,
        top_p=top_p_val,        #  added
        do_sample=True          # required for sampling
    )
    
    print(f"\nTemperature = {temp}, Top-P = {top_p_val}")
    print(result[0]['generated_text'])

# different combinations
generate_answer(0.2, 0.6)   # low randomness, strict filtering
generate_answer(0.5, 0.8)   # balanced
generate_answer(0.9, 0.95)  # high creativity