"""
Manages FAISS vector index creation and document retrieval.
Finds top-K relevant chunks based on semantic similarity.
"""

from typing import List, Tuple
from models.embeddings import Embeddings
from utils.vectorstore import FaissStore
from config.config import TOP_K

emb = Embeddings()

def create_index_from_docs(docs: List[Tuple[str, str]]):
    """
    docs: list of (doc_id, text)
    Creates FaissStore and adds docs.
    Returns FaissStore instance.
    """
    if not docs:
        raise ValueError("No docs provided to create index.")

    texts = [d[1] for d in docs]
    vectors = emb.embed(texts)
    dim = len(vectors[0])
    fs = FaissStore(dim=dim)
    metadatas = [{"id": docs[i][0], "text": texts[i]} for i in range(len(docs))]
    fs.add(vectors, metadatas)
    return fs

def retrieve(query: str, fs: FaissStore, top_k: int = TOP_K):
    """
    Returns retrieval results from Faiss Vector Store as list of metadata dicts with score
    """
    qvec = emb.embed([query])[0]
    hits = fs.search(qvec, top_k=top_k)
    return hits
