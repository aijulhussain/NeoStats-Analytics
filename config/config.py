"""
Stores all environment configuration values for LLM, embeddings, and vector storage.
Allows centralized control for tuning and deployment.
"""

import os
from dotenv import load_dotenv  

load_dotenv()



EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

# --- Vectorstore & chunking ---
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))   # approx words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Web search
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
BING_SEARCH_API_URL = os.getenv("BING_SEARCH_API_URL", "https://api.bing.microsoft.com/v7.0/search")

SAMPLE_PDF_PATH = os.getenv("SAMPLE_PDF_PATH", "/mnt/data/NeoStats AI Engineer Case Study.pdf")


user_prompt = """
        Mode: Concise
        Question:
        {question}

        Context (most relevant document excerpts):
        {doc_excerpts}

        If an answer is not fully supported by the documents, use web results and label them [WEB:1], [WEB:2].
        Provide a one-paragraph answer and 1–2 bullet point suggestions for study.

    """

system_prompt = """
        You are EduTutorGPT, an educational assistant. Use a friendly, clear tone. When answering:
        - always begin your answer with: Great! Here is the explanation.
        - Provide accurate, concise information based on the provided document context and web search results.
        - If the question is a math problem, show step-by-step mathematical expresion and final answer.
        - Write formula and mathematical expressions in real format.
        - If uncertain, say "I couldn't find a precise answer in the provided documents" and use the web search tool to find sources.
        - Follow the user-requested response style (Concise/Detailed).
    """