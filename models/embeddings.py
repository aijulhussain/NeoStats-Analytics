"""
Generates sentence embeddings using HuggingFace model.
Used for document vectorization in RAG retrieval.
"""

from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from config.config import EMBEDDING_MODEL

class Embeddings:
    """
    Hugging Face based embedding generator.
    Uses mean pooling over last_hidden_state and returns Python lists.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        # put model into eval mode
        self.model.eval()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Returns: list of vector lists (float)
        Note: small batches are processed sequentially to avoid OOM.
        """
        vectors = []
        # process in small batches (1 by default) — we can easily batch if needed
        for t in texts:
            tokens = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokens)

            # Mean pooling across sequence length (exclude padded tokens if you want — simple mean for now)
            # outputs.last_hidden_state shape: (batch=1, seq_len, hidden_dim)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            vectors.append(emb.tolist())
        return vectors
