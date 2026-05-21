from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class VectorStore:
    def __init__(self, embeddings: np.ndarray | List[list[float]], chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if isinstance(embeddings, list):
            if embeddings:
                self.embeddings = np.atleast_2d(np.array(embeddings, dtype=np.float32))
            else:
                self.embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            if embeddings.size == 0:
                self.embeddings = np.empty((0, 0), dtype=np.float32)
            else:
                self.embeddings = np.atleast_2d(embeddings)
        self.chunks = chunks
        self.metadatas = metadatas

    def extend(
        self,
        embeddings: np.ndarray | List[list[float]],
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not chunks:
            return

        new_embeddings = (
            np.array(embeddings, dtype=np.float32)
            if isinstance(embeddings, list)
            else embeddings
        )
        new_embeddings = np.atleast_2d(new_embeddings)

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            current = np.atleast_2d(self.embeddings)
            if current.shape[1] != new_embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            self.embeddings = np.vstack([current, new_embeddings])

        self.chunks.extend(chunks)
        self.metadatas.extend(metadatas)

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(str(dir_path / "embeddings.npy"), self.embeddings)
        with (dir_path / "data.json").open("w", encoding="utf-8") as handle:
            json.dump({"chunks": self.chunks, "metadatas": self.metadatas}, handle)

    @classmethod
    def load(cls, dir_path: Path) -> VectorStore:
        embeddings = np.load(str(dir_path / "embeddings.npy"))
        with (dir_path / "data.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(embeddings, data["chunks"], data["metadatas"])

    def search(
        self, query_embedding: List[float], query_text: str, top_k: int, fetch_k: int = 15
    ) -> List[Dict[str, Any]]:
        if self.embeddings.size == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        similarities = _cosine_similarity(self.embeddings, query)
        
        fetch_k = min(fetch_k, len(similarities))
        top_indices = similarities.argsort()[-fetch_k:][::-1]

        query_words = set(w.lower() for w in query_text.split() if len(w) > 2)
        
        candidates = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            base_score = float(similarities[idx])
            
            chunk_words = set(w.lower() for w in chunk.split() if len(w) > 2)
            overlap = len(query_words.intersection(chunk_words))
            lex_score = overlap / max(len(query_words), 1)
            
            # Hybrid boost: vector + lexical
            final_score = base_score + (lex_score * 0.15)
            
            candidates.append(
                {
                    "chunk": chunk,
                    "metadata": self.metadatas[idx],
                    "score": final_score,
                }
            )
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


def _cosine_similarity(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    emb_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    denom = (emb_norms * query_norm) + 1e-8
    return (embeddings @ query) / denom
