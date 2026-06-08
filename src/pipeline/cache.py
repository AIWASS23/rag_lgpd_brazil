"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).
"""

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional)."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        # Inicializa cliente para embeddings (mesmo provider do RAG)
        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
            self._use_st = False
        elif "OPENAI_API_KEY" in os.environ:
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
            self._use_st = False
        else:
            # HuggingFace ou fallback: embedding local via sentence-transformers
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer(model_name)
            self._client = None
            self._embed_model = model_name
            self._use_st = True

    def _embed(self, text: str) -> np.ndarray:
        if self._use_st:
            return np.array(self._st_model.encode(text))
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        q_emb = self._embed(query)
        store = np.array(self._embeddings)                     # (N, dim)
        norms = np.linalg.norm(store, axis=1) * np.linalg.norm(q_emb)
        sims = np.where(norms > 0, store @ q_emb / norms, 0.0)

        idx = int(np.argmax(sims))
        if sims[idx] >= self.threshold:
            return self._answers[idx]
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
