import os
import io
import math
from typing import List, Tuple

from dotenv import load_dotenv
import gradio as gr
import numpy as np
from pypdf import PdfReader

from openai import OpenAI, RateLimitError, APIError

try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

# ------ Setup ------
load_dotenv()
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OpenAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found. Please set it in the .env file. ⚠️")
client = OpenAI(api_key=OpenAI_API_KEY)

# Model names
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ------ PDF -> Text ------
def read_pdfs(files) -> str:
    """Read uploaded PDFs and return a single text string."""
    texts = []
    for f in files:
        try:
            if hasattr(f, 'read') and callable(f.read):
                content = f.read()
            elif hasattr(f, 'file') and hasattr(f.file, 'read') and callable(f.file.read):
                content = f.file.read()
            elif hasattr(f, 'name') and isinstance(f.name, str) and os.path.isfile(f.name):
                with open(f.name, 'rb') as file_obj:
                    content = file_obj.read()
            else:
                content = (getattr(f, 'data', None) or getattr(f, 'content', None) or str(f)).encode('utf-8')
            pdf = PdfReader(io.BytesIO(content))
            pages = []
            for page in pdf.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            texts.append("\n".join(pages))
        except Exception as e:
            print(f"⚠️ Error reading {getattr(f, 'name', str(f))}: {e}")
            texts.append("")
    return "\n\n".join(texts)

# ------ Text -> Chunks ------
# We will chunk the text into smaller pieces(~500 tokens) with some overlap(~50 tokens) so we keep the context.
import tiktoken
_enc_cache = {}

def get_encoder(model: str = "gpt-4o-mini"):
    # using cl100k_base for all models as they are all based on gpt-4, if gpt-4o-mini not found
    if model not in _enc_cache:
        try:
            _enc_cache[model] = tiktoken.encoding_for_model(model)
        except Exception:
            _enc_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _enc_cache[model]

def chunk_by_tokens(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    enc = get_encoder()
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        window = tokens[i:i + max_tokens]
        chunks.append(enc.decode(window))
    return [c.strip() for c in chunks if c.strip()] # remove empty chunks

# ------ Text Chunks -> OpenAI Embeddings ------
def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API. Return (N, D) array."""
    if not texts:
        return np.zeros((0, 1536), dtype='float32')  # 1536 is the embedding dimension for text-embedding-3-small
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vecs.extend([np.array(data.embedding, dtype='float32') for data in resp.data])
    arr = np.vstack(vecs)
    # Normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

# ------ Index Chunks with FAISS ------
def build_index(chunks: List[str]):
    """Return (index, chunks). If FAISS is available return FAISS index else return normalized matrix"""
    if not chunks:
        return None, []
    vectors = embed_texts(chunks)
    if FAISS_OK:
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity with normalized vectors
        index.add(vectors)
        return (index, chunks)
    else:
        return (vectors, chunks)

# ----- Retrieval ------
def retrieve(query: str, index_or_matrix, chunks: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    """Given a query and index or matrix, return top_k (score, chunk) tuples."""
    if not query.strip() or not chunks:
        return []
    query_vec = embed_texts([query])[0]  # (D,)
    if FAISS_OK and isinstance(index_or_matrix, faiss.IndexFlatIP):
        D, I = index_or_matrix.search(query_vec.reshape(1, -1), top_k)
        results = [(float(D[0][j]), chunks[i]) for j, i in enumerate(I[0]) if D[0][i] > 0]
    else:
        # index_or_matrix is the normalized matrix
        sims = index_or_matrix @ query_vec  # (N,)
        top_indices = np.argsort(-sims)[:top_k]
        results = [(float(sims[i]), chunks[i]) for i in top_indices if sims[i] > 0]
    return results

# ------ Generate Answer ------
def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate answer using OpenAI chat completion API given the query and context chunks."""
    if not query.strip():
        return "Please enter a valid query."
    system_prompt = (
        "You are a helpful AI assistant. Use the provided context to answer the question. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    context_text = "\n\n".join(context_chunks)
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}:"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_completion_tokens=500,
            temperature=0.2,
            top_p=1,
            n=1,
            stop=None
        )
        answer = resp.choices[0].message.content.strip()
        return answer
    except RateLimitError:
        return "⚠️ Rate limit exceeded. Please try again later."
    except APIError as e:
        return f"⚠️ API Error: {e}"
    except Exception as e:
        return f"⚠️ Error: {e}"
    
# ------ Gradio UI ------
with gr.Blocks(title="RAG Mini (OpenAI Embeddings)") as demo:
    gr.Markdown("# Retrieval-Augmented Generation (RAG) Mini Demo")
    gr.Markdown("Built with OpenAI Embeddings + (FAISS or NumPy) -> Build index -> Gradio UI")
    gr.Markdown(
        "Upload PDF documents, ask questions about their content, and get answers grounded in the documents."
    )
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDFs", file_types=['.pdf'], file_count="multiple")
            load_button = gr.Button("Load and Process PDFs", variant="primary")
            chunk_size_slider = gr.Slider(100, 1000, value=500, step=50, label="Chunk Size (tokens)")
            overlap_slider = gr.Slider(0, 200, value=50, step=10, label="Chunk Overlap (tokens)")
            status_output = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            query_input = gr.Textbox(label="Enter your question here", lines=2)
            ask_button = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
            context_output = gr.Textbox(label="Retrieved Context Chunks", lines=10, interactive=False)
    
    # App State
    state_chunks = gr.State([])  # List of text chunks
    state_index = gr.State(None)  # FAISS index or NumPy matrix
    index_data = gr.State((None, []))  # (index_or_matrix, chunks)

    def _build(files):
        if not OpenAI_API_KEY:
            return gr.update(value="⚠️ OPENAI_API_KEY not set. Please set it in the .env file."), (None, [])
        if not files:
            return gr.update(value="⚠️ Please upload at least one PDF file."), (None, [])
        text = read_pdfs(files)
        chunks = chunk_by_tokens(text, max_tokens=chunk_size_slider.value, overlap=overlap_slider.value)
        if not chunks:
            return gr.update(value="⚠️ No text extracted from PDFs."), (None, [])
        index_or_matrix, chunks = build_index(chunks)
        if index_or_matrix is None:
            return gr.update(value="⚠️ Failed to build index."), (None, [])
        using = "FAISS" if FAISS_OK and not isinstance(index_or_matrix, np.ndarray) else "NumPy"
        return gr.update(value=f"✅ Processed {len(chunks)} chunks. Using {using} for indexing."), (index_or_matrix, chunks)
    
    def _ask(query, k, chunks, index_data):
        if not OpenAI_API_KEY:
            return "⚠️ OPENAI_API_KEY not set. Please set it in the .env file.", ""
        if not chunks or index_data is None:
            return "⚠️ Please load and process PDFs first.", ""
        hits = retrieve(query, index_data, chunks, top_k=int(k))
        ctx = [text for _, text in hits]
        ans = generate_answer(query, ctx)
        highlights = [(context, str(round(score, 3))) for score, context in hits]
        return ans, '\n\n'.join(highlights)
    
    load_button.click(_build, inputs=[pdf_input], outputs=[status_output, state_chunks, state_index], show_progress=True)
    ask_button.click(_ask, inputs=[query_input, gr.Slider(1, 10, value=5, step=1, label="Top K Chunks"), state_chunks, state_index], outputs=[answer_output, context_output], show_progress=True)

if __name__ == "__main__":
    demo.launch()
    
    