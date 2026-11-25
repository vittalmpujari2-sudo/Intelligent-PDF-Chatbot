"""
app.py - The Streamlit user interface. Run with:  streamlit run app.py

Flow: upload PDF -> extract -> chunk -> embed -> FAISS -> chat with Groq.
"""

import config  # keep this first: it sets environment tweaks before other imports

import hashlib
import html

import streamlit as st

from pdf_processor import extract_pages, split_into_chunks
from rag_pipeline import answer_question
from vector_database import VectorDatabase

st.set_page_config(page_title="Intelligent PDF Chatbot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 980px;}

    .main-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25);
    }
    .main-header h1 {color: #fff; font-size: 1.7rem; margin: 0; padding: 0;}
    .main-header p {color: #e0e7ff; margin: 0.35rem 0 0 0; font-size: 0.95rem;}

    .status-box {
        background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.45);
        border-radius: 10px; padding: 0.7rem 0.9rem; font-size: 0.9rem; margin-top: 0.6rem;
    }
    .file-box {
        background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 10px; padding: 0.6rem 0.9rem; font-size: 0.9rem; margin-top: 0.6rem;
        word-break: break-all;
    }
    .source-label {font-size: 0.8rem; opacity: 0.75; margin-top: 0.6rem;}
    .source-chip {
        display: inline-block; background: rgba(99, 102, 241, 0.14); color: #6366f1;
        border: 1px solid rgba(99, 102, 241, 0.4); padding: 0.15rem 0.7rem;
        border-radius: 999px; font-size: 0.8rem; margin: 0.15rem 0.35rem 0.15rem 0;
    }
    .stButton > button {width: 100%; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- caching
@st.cache_resource(show_spinner=False)
def build_database(file_hash: str, _file_bytes: bytes, filename: str):
    """
    Process a PDF once. The result is cached by file hash, so uploading or
    re-running with the same PDF never repeats the (slow) embedding step.
    """
    pages = extract_pages(_file_bytes)
    chunks = split_into_chunks(pages, filename)
    db = VectorDatabase()
    db.build(chunks)
    return db, len(pages), len(chunks)


# ---------------------------------------------------------------- helpers
def clear_chat():
    st.session_state.messages = []


def render_sources(sources):
    """Show source file + page chips and an expander with the retrieved text."""
    if not sources:
        return
    unique = sorted({(s["source"], s["page"]) for s in sources}, key=lambda x: (x[0], x[1]))
    chips = "".join(
        f'<span class="source-chip">📄 {html.escape(name)} — Page {page}</span>' for name, page in unique
    )
    st.markdown(f'<div class="source-label">Sources</div>{chips}', unsafe_allow_html=True)
    with st.expander("View retrieved text"):
        for s in sources:
            st.markdown(f"**{s['source']} — Page {s['page']}** (similarity {s['score']:.2f})")
            st.caption(s["text"])


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


# ---------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------- header
st.markdown(
    """
    <div class="main-header">
        <h1>📄 Intelligent PDF Question Answering Chatbot</h1>
        <p>Upload a PDF and ask questions — powered by RAG, FAISS and the Groq API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
db = None
with st.sidebar:
    st.header("📁 Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded is not None:
        st.markdown(f'<div class="file-box">📄 <b>{html.escape(uploaded.name)}</b></div>', unsafe_allow_html=True)
        file_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        try:
            with st.spinner("Processing PDF... (first run also downloads the small embedding model)"):
                db, n_pages, n_chunks = build_database(file_hash, file_bytes, uploaded.name)
            st.markdown(
                f'<div class="status-box">✅ <b>Ready</b><br>{n_pages} pages with text • {n_chunks} chunks indexed</div>',
                unsafe_allow_html=True,
            )
        except ValueError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"Something went wrong while processing the PDF: {err}")
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    st.button("🗑️ Clear Chat", on_click=clear_chat)

    st.divider()
    st.caption(f"🤖 LLM: `{config.GROQ_MODEL}`")
    st.caption(f"🔎 Embeddings: `{config.EMBEDDING_MODEL_NAME}`")
    st.caption(f"📚 Top-{config.TOP_K} chunks retrieved per question")
    if not config.is_api_key_configured():
        st.error("GROQ_API_KEY is not set. Edit the .env file, then restart the app.")

# ---------------------------------------------------------------- chat
for msg in st.session_state.messages:
    render_message(msg)

if db is None and not st.session_state.messages:
    st.info("👈 Upload a PDF in the sidebar, then ask your questions here.")

question = st.chat_input(
    "Ask a question about your PDF..." if db else "Upload a PDF first",
    disabled=db is None,
)

if question:
    user_msg = {"role": "user", "content": question}
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and thinking..."):
            result = answer_question(question, db)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(user_msg)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
