"""
Main Streamlit application file for EduTutor chatbot.
Handles UI, RAG workflow, LLM streaming responses, and web fallback.
"""

import os
import streamlit as st

from config.config import SAMPLE_PDF_PATH, TOP_K, system_prompt
from utils.ingest import ingest_file
from utils.retrieval import create_index_from_docs, retrieve
from models.llm import stream_groq_chat
from utils.websearch import web_search_fallback

st.set_page_config(page_title="EduTutor — Tutor Chatbot", layout="wide")
st.title("📚 HN Classes (AI Tutor)")

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls
st.sidebar.header("Settings ⚙️")
mode = st.sidebar.radio("Response style", ["Concise", "Detailed"])

assets_folder = "./assets"
use_sample = st.sidebar.checkbox("Use HN Classes Study Notes", value=True)

# Sidebar button to clear conversation
if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Load sample PDFs
docs = []
if use_sample and os.path.exists(assets_folder):
    asset_files = [f for f in os.listdir(assets_folder) if f.endswith(".pdf")]
    if asset_files:
        st.sidebar.success(f"📚 Loaded {len(asset_files)} study notes")
        for file in asset_files:
            file_path = os.path.join(assets_folder, file)
            extracted_docs = ingest_file(file_path)
            if extracted_docs:
                docs.extend(extracted_docs)


# Upload new documents
uploaded = st.file_uploader("📄 Upload document", type=["pdf", "txt"])
if uploaded:
    save_dir = "./uploaded_files"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded.name)

    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Uploaded: {uploaded.name}")

    uploaded_docs = ingest_file(save_path)
    if uploaded_docs:
        docs.extend(uploaded_docs)

# Create vector index only if docs exist
ingested_index = None
if docs:
    ingested_index = create_index_from_docs(docs)

# Show chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Bottom-fixed question box
question = st.chat_input("Ask your question ✍️")
if question:
    # Show user bubble immediately
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append(("user", question))

    # Create memory context
    memory_context = ""
    for role, msg in st.session_state.chat_history[-6:]:
        memory_context += f"{role}: {msg}\n"

    # RAG retrieval
    doc_context = ""
    sources = []
    if ingested_index:
        hits = retrieve(question, ingested_index, top_k=TOP_K)
        for h in hits:
            md = h["metadata"]
            snippet = md.get("text", "")[:1000]
            doc_id = md.get("id", "0")
            doc_context += f"[DOC:{doc_id}] {snippet}\n\n"
            sources.append(doc_id)

    # LLM prompt
    user_prompt = (
        f"Conversation Memory:\n{memory_context}\n\n"
        f"Context from Documents:\n{doc_context}\n\n"
        f"Question: {question}\n\n"
        f"Mode: {mode}\n"
        "Reply clearly and support answers with sources if relevant."
    )
    system_prompt = system_prompt
    # AI Response bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        try:
            for chunk in stream_groq_chat(system_prompt, user_prompt):
                answer += chunk
                placeholder.markdown(answer)
        except Exception as e:
            placeholder.error(f"🚨 Error: {e}")
            answer = "Sorry, something went wrong."

    # Save answer to memory
    st.session_state.chat_history.append(("assistant", answer))
