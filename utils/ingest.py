"""
Extracts text from PDF/documents and splits into chunks.
Prepares documents for indexing into the vector store.
"""

import os
from typing import List, Tuple
from config.config import MAX_CHUNK_SIZE, CHUNK_OVERLAP
from langchain_community.document_loaders import PyPDFLoader


try:  
    _HAS_LANGCHAIN = True
except Exception:
    _HAS_LANGCHAIN = False


def load_pdf_text(path: str) -> str:
    """
    Returns full text of a PDF at path.
    Uses LangChain's PyPDFLoader if available; otherwise uses PyPDF2 as fallback.
    """
    text = ""
    if _HAS_LANGCHAIN:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                text += page.page_content + "\n"
            return text
        except Exception:
            pass

    


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple word-based chunking. Returns list of text chunks.
    """
    if not text:
        return []

    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_size]
        chunks.append(" ".join(chunk))
        i += max_size - overlap
    return chunks


def ingest_file(path: str) -> List[Tuple[str, str]]:
    """
    Ingest a file and return list of (chunk_id, chunk_text).
    """
    text = load_pdf_text(path)
    chunks = chunk_text(text)
    basename = os.path.basename(path)
    return [(f"{basename}_chunk_{i}", chunk) for i, chunk in enumerate(chunks)]
