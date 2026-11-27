import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Step 1 : Load Embedding Model
# -----------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Step 2 : Semantic Chunking
# with Chunk Size Control
# -----------------------------------
def semantic_chunking(
        text,
        similarity_threshold=0.7,
        max_sentences_per_chunk=3
):

    # split paragraph into sentences
    sentences = sent_tokenize(text)

    # generate embeddings
    embeddings = model.encode(sentences)

    chunks = []

    # first sentence starts first chunk
    current_chunk = [sentences[0]]

    # loop through remaining sentences
    for i in range(1, len(sentences)):

        # calculate similarity
        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        print(f"\nSimilarity between:")
        print(f"'{sentences[i-1]}'")
        print("AND")
        print(f"'{sentences[i]}'")
        print(f"= {sim}")

        # -----------------------------------
        # same semantic meaning
        # AND chunk size not exceeded
        # -----------------------------------
        if (
            sim >= similarity_threshold
            and len(current_chunk) < max_sentences_per_chunk
        ):

            current_chunk.append(sentences[i])

        else:
            # save old chunk
            chunks.append(current_chunk)

            # create new chunk
            current_chunk = [sentences[i]]

    # append last chunk
    chunks.append(current_chunk)

    return chunks

# -----------------------------------
# Step 3 : Apply Overlap
# -----------------------------------
def apply_overlap(chunks, overlap_sentences=1):

    final_chunks = []

    for i in range(len(chunks)):

        chunk = chunks[i]

        # add overlap from previous chunk
        if i > 0:

            prev_chunk = chunks[i - 1]

            overlap = prev_chunk[-overlap_sentences:]

            chunk = overlap + chunk

        final_chunks.append(" ".join(chunk))

    return final_chunks

# -----------------------------------
# Step 4 : Input Text
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
# Step 5 : Run Semantic Chunking
# -----------------------------------
semantic_chunks = semantic_chunking(
    text,
    similarity_threshold=0.7,
    max_sentences_per_chunk=3
)

# -----------------------------------
# Step 6 : Apply Overlap
# -----------------------------------
final_chunks = apply_overlap(
    semantic_chunks,
    overlap_sentences=1
)

# -----------------------------------
# Step 7 : Print Final Chunks
# -----------------------------------
for i, chunk in enumerate(final_chunks):

    print("\n" + "=" * 50)

    print(f"\nChunk {i + 1}:\n")

    print(chunk)
