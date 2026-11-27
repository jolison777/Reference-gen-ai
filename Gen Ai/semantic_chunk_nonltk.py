import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Step 1 : Load Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# Step 2 : Input Text
# -----------------------------------

text = """
Bank account was blocked due to suspicious activity.
Customer needs to verify identity using KYC documents.
Failure to verify will result in permanent suspension.
Customer can contact support for assistance.

Loan request was rejected due to low credit score.
Customer can reapply after improving credit history.

Credit card limit was reduced due to delayed payments.
Customer received notification through mobile banking.
"""


# -----------------------------------
# Step 3 : Configuration
# -----------------------------------

similarity_threshold = 0.50

max_chunk_size = 200       # characters
overlap_sentences = 1      # one complete sentence


# -----------------------------------
# Step 4 : Split into Sentences
# -----------------------------------

sentences = re.split(
    r'(?<=[.!?])\s+',
    text.strip()
)


# -----------------------------------
# Step 5 : Create Embeddings
# -----------------------------------

embeddings = model.encode(sentences)


# -----------------------------------
# Step 6 : Semantic Chunking
# -----------------------------------

chunks = []

current_chunk = [sentences[0]]


for i in range(1, len(sentences)):

    similarity = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    next_sentence = sentences[i]

    # Current chunk + next sentence
    temp_chunk = " ".join(
        current_chunk + [next_sentence]
    )

    # Same meaning AND within size
    if (
        similarity >= similarity_threshold
        and len(temp_chunk) <= max_chunk_size
    ):

        current_chunk.append(next_sentence)

    else:

        # Save current chunk
        chunks.append(" ".join(current_chunk))

        # Take COMPLETE sentence for overlap
        overlap = current_chunk[-overlap_sentences:]

        # Start new chunk
        current_chunk = overlap + [next_sentence]


# Add last chunk
chunks.append(" ".join(current_chunk))


# -----------------------------------
# Step 7 : Print Chunks
# -----------------------------------

for i, chunk in enumerate(chunks):

    print("\n" + "=" * 50)
    print(f"Chunk {i + 1}:")
    print(chunk)
