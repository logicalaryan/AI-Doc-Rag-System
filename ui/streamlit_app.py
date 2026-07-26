"""
ui/streamlit_app.py — Chat-style Streamlit frontend connected to FastAPI backend.

Run locally with:
    streamlit run ui/streamlit_app.py
"""

import os
import sys
from pathlib import Path
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------
DEFAULT_BACKEND_URL = "https://backend-web-service-a0to.onrender.com"
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")

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
    .status-online { color: #22c55e; font-weight: 600; }
    .status-offline { color: #ef4444; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper functions for backend requests
# ---------------------------------------------------------------------------
def check_backend_health():
    """Query /health endpoint to check server & vectorstore status."""
    if not BACKEND_URL:
        return False, False, "BACKEND_URL not set"
    try:
        # Timeout 15s to allow for Render free tier cold-starts
        res = requests.get(f"{BACKEND_URL}/health", timeout=15)
        if res.status_code == 200:
            data = res.json()
            return True, data.get("vectorstore_ready", False), "Online"
        return False, False, f"HTTP {res.status_code}"
    except Exception as e:
        return False, False, f"Offline / Cold Starting"


def query_backend_ask(question: str, k: int):
    """Send question to backend /ask REST API endpoint."""
    try:
        res = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question, "k": k},
            timeout=60,
        )
        if res.status_code == 200:
            data = res.json()
            return data.get("answer", ""), data.get("sources", []), None
        
        err_detail = ""
        try:
            err_detail = res.json().get("detail", res.text)
        except Exception:
            err_detail = res.text

        if res.status_code == 503:
            return (
                "⚠️ Vectorstore not ready. Please ingest documents first using the sidebar button.",
                [],
                None,
            )
        
        return None, [], f"Backend Error ({res.status_code}): {err_detail}"
    except requests.exceptions.Timeout:
        return None, [], "Request timed out waiting for backend LLM response."
    except Exception as e:
        return None, [], f"Failed to reach backend API: {e}"


def trigger_backend_ingest():
    """Send ingest request to backend /ingest REST API endpoint."""
    try:
        res = requests.post(
            f"{BACKEND_URL}/ingest",
            json={"data_dir": "./data", "chunk_size": 1000, "chunk_overlap": 200},
            timeout=10,
        )
        if res.status_code in (200, 202):
            data = res.json()
            return True, data.get("message", "Ingestion started.")
        return False, f"Backend returned status {res.status_code}: {res.text}"
    except Exception as e:
        return False, f"Failed to trigger ingestion: {e}"

# ---------------------------------------------------------------------------
# Sidebar — settings & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=8, value=3)
    
    st.divider()
    st.markdown("### 🌐 Backend Connection")
    is_online, vs_ready, status_msg = check_backend_health()
    if is_online:
        st.markdown(f"**Status:** <span class='status-online'>🟢 {status_msg}</span>", unsafe_allow_html=True)
        if vs_ready:
            st.caption("✅ Vectorstore Ready")
        else:
            st.caption("⚠️ Vectorstore Empty")
    else:
        st.markdown(f"**Status:** <span class='status-offline'>🔴 {status_msg}</span>", unsafe_allow_html=True)
    st.caption(f"URL: `{BACKEND_URL}`")

    st.divider()
    st.markdown("### 📂 Upload & Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files to the backend",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("⬆️ Upload to Backend", use_container_width=True):
            upload_errors = []
            with st.spinner(f"Uploading {len(uploaded_files)} file(s)…"):
                for uf in uploaded_files:
                    try:
                        res = requests.post(
                            f"{BACKEND_URL}/upload",
                            files={"file": (uf.name, uf.getvalue(), uf.type)},
                            timeout=30,
                        )
                        if res.status_code == 200:
                            st.success(f"✅ {uf.name} uploaded")
                        else:
                            upload_errors.append(f"{uf.name}: {res.text}")
                    except Exception as e:
                        upload_errors.append(f"{uf.name}: {e}")
            if upload_errors:
                for err in upload_errors:
                    st.error(err)
            else:
                st.info("All files uploaded! Now click **Ingest** to index them.")

    st.divider()
    st.markdown("### 🔄 Ingest Uploaded Documents")
    st.caption("Run this after uploading to build the vectorstore.")
    if st.button("🔄 Trigger Backend Ingestion", use_container_width=True):
        with st.spinner("Requesting ingestion on backend…"):
            ok, msg = trigger_backend_ingest()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "Frontend connected via REST API to FastAPI backend running on Render."
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

    # Generate assistant response via backend REST API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources, err = query_backend_ask(prompt, k=top_k)
            if err:
                if "429" in err or "quota" in err.lower():
                    answer = (
                        "⚠️ **Rate Limit Exceeded:** The backend Google Gemini API free tier request limit was reached. "
                        "Please wait about a minute before trying again."
                    )
                else:
                    answer = f"⚠️ {err}"
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
