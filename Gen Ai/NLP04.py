import PyPDF2
import nltk
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

files=[
    r"D:\GENAI\HR_Policy_Document.pdf",
    r"D:\GENAI\Employee_Conduct_Policy.pdf",
    r"D:\GENAI\Workplace_Safety_Policy.pdf",
    r"D:\GENAI\IT_Security_Policy.pdf"
   
]

documents = []

for file in files:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    documents.append(text)

print("Documents Loaded:", len(documents))

sentences = []

for doc in documents:
    sents = sent_tokenize(doc)
    for s in sents:
        sentences.append(s)

print("Total Sentences:", len(sentences))


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    clean_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stop_words:
            stem = stemmer.stem(word)
            clean_tokens.append(stem)
    return " ".join(clean_tokens)

processed_sentences = []

for s in sentences:
    processed_sentences.append(preprocess(s))

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_sentences)

query = input("Ask Question: ")

processed_query = preprocess(query)

query_vector = vectorizer.transform([processed_query])

similarities = cosine_similarity(query_vector, tfidf_matrix)

scores = similarities[0]


# -------------------------------
# TOP-K + CONFIDENCE THRESHOLD
# -------------------------------
top_k = 3
threshold = 0.2   # you can tune this (0.2–0.5)

scores_series = pd.Series(scores)

top_indices = scores_series.nlargest(top_k).index

print("\nTop Matching Sentences:\n")

found = False

for idx in top_indices:

    if scores[idx] >= threshold:   # APPLYING THRESHOLD

        print("Answer:", sentences[idx])
        print("Similarity Score:", scores[idx])
        print("----------------------------------")
        found = True


# If no result passes threshold
if not found:
    print("No confident answer found. Please refine your question.")

