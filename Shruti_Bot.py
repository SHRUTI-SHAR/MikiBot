import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Load sentence-transformer model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and parse FAQ file
def load_faqs_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    qa_pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)", content, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

# Create embeddings and FAISS index
def build_faiss_index(faqs):
    questions = [item['question'] for item in faqs]
    embeddings = embedding_model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, faqs

# Search and get answer
def chatbot_answer(query, index, faqs):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    matched_index = I[0][0]
    return faqs[matched_index]["answer"]

# Load FAQs
faqs = load_faqs_from_file("faqs.txt")
index, faqs = build_faiss_index(faqs)

print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break
    response = chatbot_answer(user_input, index, faqs)
    print("Bot:", response)
