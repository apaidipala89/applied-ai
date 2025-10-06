import gradio as gr
import numpy as np
from ragmini.config import OPENAI_API_KEY, CHUNK_TOKENS, CHUNK_OVERLAP, USE_FAISS
from ragmini.io.pdf_reader import read_pdfs
from ragmini.text.chunking import chunk_by_tokens
from ragmini.index.vector_store import build_index
from ragmini.retriever.retriever import retrieve_similar_chunks as retrieve
from ragmini.llm.answer import generate_answer
    
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
            chunk_size_slider = gr.Slider(100, 1000, value=CHUNK_TOKENS, step=50, label="Chunk Size (tokens)")
            overlap_slider = gr.Slider(0, 200, value=CHUNK_OVERLAP, step=10, label="Chunk Overlap (tokens)")
            top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Top K Chunks")  # moved to UI
            status_output = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            query_input = gr.Textbox(label="Enter your question here", lines=2)
            ask_button = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer", lines=10, interactive=False)
            context_output = gr.Textbox(label="Retrieved Context Chunks", lines=10, interactive=False)
    
    # App State
    state_chunks = gr.State([])      # holds list[str]
    state_index = gr.State(None)     # holds FAISS index or np.ndarray

    def _build(files, max_tokens, overlap):
        if not OPENAI_API_KEY:
            return gr.update(value="⚠️ OPENAI_API_KEY not set. Set it in .env."), [], None
        if not files:
            return gr.update(value="⚠️ Please upload at least one PDF file."), [], None
        text = read_pdfs(files)
        if not text.strip():
            return gr.update(value="⚠️ No extractable text in PDFs."), [], None
        chunks = chunk_by_tokens(text, max_tokens=max_tokens, overlap=overlap)
        if not chunks:
            return gr.update(value="⚠️ Chunking produced 0 chunks."), [], None
        try:
            index_or_matrix, chunks = build_index(chunks)
        except Exception as e:
            return gr.update(value=f"⚠️ Failed to build index: {e}"), [], None
        if index_or_matrix is None:
            return gr.update(value="⚠️ Failed to build index."), [], None
        using = "FAISS" if USE_FAISS and not isinstance(index_or_matrix, np.ndarray) else "NumPy"
        return gr.update(value=f"✅ Processed {len(chunks)} chunks. Using {using} index."), chunks, index_or_matrix

    def _ask(query, k, chunks, index_or_matrix):
        if not OPENAI_API_KEY:
            return "⚠️ OPENAI_API_KEY not set.", ""
        if not index_or_matrix or not chunks:
            return "⚠️ Load and process PDFs first.", ""
        if not query or not query.strip():
            return "⚠️ Enter a question.", ""
        hits = retrieve(query, index_or_matrix, chunks, top_k=int(k))
        ctx = [text for _, text in hits]
        ans = generate_answer(query, ctx)
        highlights = [(context, str(round(score, 3))) for score, context in hits]
        return ans, '\n\n'.join(highlights)

    # Updated inputs/outputs to match function signatures (3 outputs)
    load_button.click(
        _build,
        inputs=[pdf_input, chunk_size_slider, overlap_slider],
        outputs=[status_output, state_chunks, state_index],
        show_progress=True
    )

    ask_button.click(
        _ask,
        inputs=[query_input, top_k_slider, state_chunks, state_index],
        outputs=[answer_output, context_output],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()

