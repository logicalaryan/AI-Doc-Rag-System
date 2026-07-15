"""
ui/streamlit_app.py — Chat-style Streamlit frontend.

Run with:
    streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .source-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-right: 6px;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=3)
    st.divider()
    st.markdown("### 📂 Ingest Documents")
    st.info("Drop PDF, TXT, or MD files into the `data/` folder, then click **Ingest**.")
    if st.button("🔄 Ingest data/ folder", use_container_width=True):
        with st.spinner("Ingesting documents…"):
            try:
                from app.ingest import ingest
                ingest()
                st.session_state["vectorstore"] = None  # force reload
                st.success("Ingestion complete!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "Built with **LangChain**, **ChromaDB**, **HuggingFace** embeddings, "
        "and **Google Gemini**."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">📄 RAG Document Q&amp;A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your documents. '
    "Answers are grounded in your content — no hallucinations.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def get_vectorstore():
    if st.session_state.vectorstore is None:
        try:
            from app.retriever import load_vectorstore
            st.session_state.vectorstore = load_vectorstore()
        except Exception as e:
            return None, str(e)
    return st.session_state.vectorstore, None


# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            source_html = "".join(
                f'<span class="source-chip">📎 {s["source"]}'
                + (f' p.{s["page"]}' if s.get("page") else "")
                + "</span>"
                for s in msg["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your documents…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            vs, err = get_vectorstore()
            if err or vs is None:
                answer = (
                    "⚠️ Vectorstore not ready. "
                    "Please ingest documents first using the sidebar button."
                )
                sources = []
            else:
                try:
                    from app.chain import ask
                    result = ask(prompt, vectorstore=vs, k=top_k)
                    answer = result.answer
                    sources = result.sources
                except EnvironmentError:
                    answer = (
                        "⚠️ **GOOGLE_API_KEY not set.** "
                        "Copy `.env.example` → `.env` and add your key, "
                        "then restart the app."
                    )
                    sources = []
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        answer = (
                            "⚠️ **Rate Limit Exceeded:** You're using the Google Gemini free tier and have hit the request limit. "
                            "Please wait about a minute before asking another question."
                        )
                    else:
                        answer = f"⚠️ Error: {e}"
                    sources = []

        st.markdown(answer)
        if sources:
            source_html = "".join(
                f'<span class="source-chip">📎 {s["source"]}'
                + (f' p.{s["page"]}' if s.get("page") else "")
                + "</span>"
                for s in sources
            )
            st.markdown(source_html, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
