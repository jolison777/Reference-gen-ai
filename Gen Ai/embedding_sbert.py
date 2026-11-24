from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "bank loan approval",
    "credit risk analysis",
    "weather is sunny"
]

embeddings = model.encode(sentences)

query = "Its too hot outside"

query_embedding = model.encode(query)

print("Embedding Shape", embeddings.shape)

similarity = cosine_similarity([query_embedding], embeddings)

print("Similarity", similarity)

best_index = similarity.argmax()

print("Best match:", sentences[best_index])