"""ChromaDB-based paper retrieval for the chatbot.

Queries the same ChromaDB collection that the pipeline writes to.
No local index needed — connects to ChromaDB server on the cloud.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class PaperRetriever:
    """Retrieves relevant papers from ChromaDB."""

    def __init__(
        self,
        host: str = "",
        port: int = 8000,
        collection_name: str = "arxiv_papers",
    ):
        """Initialize retriever.

        Args:
            host: ChromaDB server hostname (from env CHROMADB_HOST).
            port: ChromaDB server port (from env CHROMADB_PORT).
            collection_name: ChromaDB collection name.
        """
        import chromadb

        host = host or os.getenv("CHROMADB_HOST", "localhost")
        port = int(os.getenv("CHROMADB_PORT", str(port)))
        persist_dir = os.getenv("CHROMADB_PERSIST_DIR", "")

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.HttpClient(host=host, port=port)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB retriever: {self._collection.count()} documents in '{collection_name}'"
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant paper chunks.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of result dicts with text, title, arxiv_id, score, etc.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            output.append({
                "text": doc,
                "title": meta.get("title", ""),
                "arxiv_id": meta.get("arxiv_id", ""),
                "topics": meta.get("topics", "").split("|"),
                "abs_url": meta.get("abs_url", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "quality_i": meta.get("quality_i", 0),
                "published": meta.get("published", ""),
                "authors": meta.get("authors", ""),
                "score": 1.0 - float(dist),
            })

        return output
