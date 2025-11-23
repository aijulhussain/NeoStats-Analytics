
import os

from langchain_community.vectorstores import FAISS
import numpy as np
import pickle
from typing import List, Dict
from config.config import VECTORSTORE_DIR

class FaissStore:
    """
    Simple FAISS wrapper storing index + metadata persistently.
    Metadata is a list of dicts saved as a pickle alongside the index.
    """

    def __init__(self, dim: int, path: str = VECTORSTORE_DIR):
        os.makedirs(path, exist_ok=True)
        self.index_path = os.path.join(path, "faiss.index")
        self.meta_path = os.path.join(path, "faiss_meta.pkl")
        self.dim = dim

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = FAISS.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadatas = pickle.load(f)
               
                if self.index.d != dim:
                    print("FAISS index dim mismatch — recreating index.")
                    self._create_index()
            except Exception:
                self._create_index()
        else:
            self._create_index()

    def _create_index(self):
        self.index = FAISS.IndexFlatL2(self.dim)
        self.metadatas = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict]):
        """
        Add vectors (list of lists) and matching metadata dicts.
        """
        arr = np.array(vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {arr.shape[1]}")

        self.index.add(arr)
        self.metadatas.extend(metadatas)
        self._save()

    def search(self, query_vector: List[float], top_k: int = 5):
        """
        Returns list of result dicts: {"score": float, "metadata": metadata_dict}
        """
        q = np.array(query_vector, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({"score": float(dist), "metadata": self.metadatas[idx]})
        return results

    def _save(self):
        FAISS.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
