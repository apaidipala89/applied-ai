# In this demo we will do a small chunking of a paragraph
# and embed each chunk using real embedding locally
# and then we will use cosine similarity to find the most similar chunk
# to a given query.

import re
import numpy as np
from sentence_transformers import SentenceTransformer

PARAGRAPH = """
Retrieval-Augmented Generation (RAG) combines information retrieval with large language models.
First, documents such as PDFs or web pages are split into overlapping chunks to preserve context.
Each chunk is embedded into a vector space, where semantically similar texts are close together.
At query time, the user question is also embedded and the system retrieves the top-k most similar chunks.
The LLM then generates an answer grounded in those retrieved passages, reducing hallucinations.
"""

# STEP 1: Chunk the paragraph into smaller pieces
def chunk_text(text: str, chunk_size: int = 60, overlap: int = 20):
    words = re.findall(r"\w+(?:'\w+)?", text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

chunks = chunk_text(PARAGRAPH, chunk_size=60, overlap=10)
print(f"Made {len(chunks)} chunks.")

# STEP 2: Embed the chunks using a local embedding model
# For this demo, we will use all-MiniLM-L6-v2 model from sentence-transformers
# It's a small model that can run on CPU and gives good quality embeddings.
# Some other popular sentence embedding models:
# - "sentence-transformers/all-MiniLM-L12-v2"
# - "sentence-transformers/paraphrase-MiniLM-L6-v2"
# - "sentence-transformers/paraphrase-MPNet-base-v2"
# - "sentence-transformers/all-mpnet-base-v2"
# - "sentence-transformers/distiluse-base-multilingual-cased-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
print(f"Embeddings shape: {chunk_embeddings.shape} for {len(chunks)} chunks.")

# STEP 3: Give a query and convert the query to embedding 
# and then retrieve the most similar chunk using cosine similarity
def retrieve(query: str, top_k: int = 3):
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    similarities = chunk_embeddings @ query_embedding
    top_k_indices = np.argsort(-similarities)[:top_k]
    return [(float(similarities[i]), chunks[i]) for i in top_k_indices]
    
# STEP 4: Test the retrieval with a sample query
query = "How does RAG reduce hallucinations in LLMs?"
results = retrieve(query, top_k=2)

print(f"Query: {query} \n")
print("Top matches:")
for score, text in results:
    preview = (text[:140] + "...") if len(text) > 140 else text
    print(f"{score:.4f} :: {preview}")
