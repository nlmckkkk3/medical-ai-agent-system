"""RAG retrieval tool — supports both embedding-based and keyword-based retrieval.

If sentence-transformers is unavailable (e.g. behind GFW), falls back to
lightweight keyword matching that requires no model downloads.
"""

import os
import re
import json
from pathlib import Path
from collections import Counter

from src.config import CHROMA_DB_PATH


def _tokenize_chinese(text: str) -> list[str]:
    """Simple character bigram tokenizer for Chinese text. No external deps."""
    text = re.sub(r"[^一-鿿\w]", " ", text.lower())
    chars = [c for c in text if c.strip()]
    unigrams = chars
    bigrams = ["".join(chars[i:i+2]) for i in range(len(chars)-1)]
    return unigrams + bigrams


def _jaccard_similarity(query_tokens, doc_tokens):
    """Jaccard similarity between two token lists."""
    if not query_tokens or not doc_tokens:
        return 0.0
    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    intersection = q_set & d_set
    union = q_set | d_set
    return len(intersection) / len(union) if union else 0.0


class KeywordStore:
    """Lightweight keyword-based document store. Replacement for ChromaDB when
    embeddings are unavailable."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir) / "keyword_store"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[dict]] = {}

    def _collection_path(self, name: str) -> Path:
        return self.storage_dir / f"{name}.json"

    def add_documents(self, collection_name: str, documents: list[str],
                      metadatas: list[dict] = None, ids: list[str] = None):
        path = self._collection_path(collection_name)
        existing = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if ids is None:
            start = len(existing)
            ids = [f"doc_{start + i}" for i in range(len(documents))]
        for i, doc in enumerate(documents):
            existing.append({
                "id": ids[i] if ids else f"doc_{len(existing)}",
                "content": doc,
                "metadata": metadatas[i] if metadatas else {},
                "tokens": _tokenize_chinese(doc),
            })
        self._cache[collection_name] = existing
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def search(self, collection_name: str, query: str, top_k: int = 3) -> list[dict]:
        path = self._collection_path(collection_name)
        if collection_name not in self._cache:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._cache[collection_name] = json.load(f)
            else:
                self._cache[collection_name] = []

        docs = self._cache.get(collection_name, [])
        if not docs:
            return []

        query_tokens = _tokenize_chinese(query)
        scored = [
            {
                "content": d["content"],
                "metadata": d.get("metadata", {}),
                "score": _jaccard_similarity(query_tokens, d.get("tokens", [])),
            }
            for d in docs
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [s for s in scored[:top_k] if s["score"] > 0]


def _load_embedding_model():
    """Load BGE embedding model. Tries ModelScope local cache first, then
    direct download via modelscope SDK, finally huggingface as last resort."""
    from src.config import EMBEDDING_MODEL

    # [1] Try modelscope SDK (works in China, no VPN needed)
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(EMBEDDING_MODEL, cache_dir="./data/models")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_dir)
    except Exception:
        pass

    # [2] Try direct local path (ModelScope may have already cached it)
    try:
        local_path = f"./data/models/{EMBEDDING_MODEL}"
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(local_path)
    except Exception:
        pass

    # [3] Try huggingface directly (requires VPN in China)
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


class RAGTool:
    """Unified RAG tool. Uses keyword mode by default (zero network deps).
    Set RAG_MODE=embedding in .env to opt into ChromaDB + BGE embeddings."""

    def __init__(self):
        self._chroma = None
        self._embedder = None
        self._keyword_store = None
        self._use_keyword = True

        if os.getenv("RAG_MODE", "") == "embedding":
            try:
                import chromadb
                from chromadb.config import Settings

                embedder = _load_embedding_model()
                if embedder is not None:
                    self._chroma = chromadb.PersistentClient(
                        path=CHROMA_DB_PATH,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    self._embedder = embedder
                    self._use_keyword = False
            except Exception:
                pass

        if self._use_keyword:
            self._keyword_store = KeywordStore(CHROMA_DB_PATH)

    def get_or_create_collection(self, name: str):
        if self._use_keyword:
            return self._keyword_store
        try:
            return self._chroma.get_collection(name)
        except Exception:
            return self._chroma.create_collection(name)

    def add_documents(self, collection_name: str, documents: list[str],
                      metadatas: list[dict] = None, ids: list[str] = None):
        if self._use_keyword:
            self._keyword_store.add_documents(
                collection_name, documents, metadatas, ids)
            return

        collection = self.get_or_create_collection(collection_name)
        embeddings = self._embedder.encode(documents).tolist()
        if ids is None:
            ids = [f"doc_{collection.count() + i}" for i in range(len(documents))]
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids,
        )

    def search(self, collection_name: str, query: str, top_k: int = 3) -> list[dict]:
        if self._use_keyword:
            raw = self._keyword_store.search(collection_name, query, top_k)
            return [
                {"content": r["content"], "metadata": r["metadata"], "distance": 1.0 - r["score"]}
                for r in raw
            ]

        collection = self.get_or_create_collection(collection_name)
        query_embedding = self._embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def search_formatted(self, collection_name: str, query: str, top_k: int = 3) -> str:
        results = self.search(collection_name, query, top_k)
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results):
            meta_str = ", ".join(f"{k}: {v}" for k, v in r["metadata"].items() if v)
            lines.append(f"[来源 {i+1}] {meta_str}\n{r['content']}")
        return "\n\n".join(lines)
