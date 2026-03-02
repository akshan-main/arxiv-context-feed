"""ChromaDB store for paper text and metadata.

Used by both the pipeline (write) and the chatbot (read).
ChromaDB handles embedding automatically using its built-in
ONNX MiniLM model — no separate embedding step needed.

The pipeline extracts text from PDFs, chunks it, and stores
chunks in ChromaDB. The chatbot queries ChromaDB directly.
No local file storage — everything lives in ChromaDB.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Chunk settings
CHUNK_SIZE_CHARS = 512
CHUNK_OVERLAP_CHARS = 64


class ChromaDBStore:
    """Stores and retrieves paper text chunks via ChromaDB.

    Connects to a ChromaDB server (for cloud deployment) or uses
    persistent local storage (for dev). ChromaDB handles embeddings
    internally using its built-in ONNX model.
    """

    def __init__(
        self,
        host: str = "",
        port: int = 8000,
        persist_dir: str = "",
        collection_name: str = "arxiv_papers",
    ):
        """Initialize ChromaDB connection.

        Args:
            host: ChromaDB server hostname. If set, uses client-server mode.
            port: ChromaDB server port.
            persist_dir: Local persistent directory. Used if host is not set.
            collection_name: Name of the ChromaDB collection.
        """
        import chromadb

        host = host or os.getenv("CHROMADB_HOST", "")
        port = int(os.getenv("CHROMADB_PORT", str(port)))
        persist_dir = persist_dir or os.getenv("CHROMADB_PERSIST_DIR", "")

        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
            logger.info(f"ChromaDB: connected to {host}:{port}")
        elif persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB: persistent storage at {persist_dir}")
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB: in-memory mode (data will not persist)")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}': "
            f"{self._collection.count()} existing documents"
        )

    def store_paper(
        self,
        arxiv_id: str,
        version: int,
        title: str,
        topics: list[str],
        quality_i: int,
        rationale: str,
        pdf_bytes: bytes,
        published: str = "",
        authors: str = "",
    ) -> int:
        """Extract text from PDF, chunk, and store in ChromaDB.

        Args:
            arxiv_id: arXiv paper ID.
            version: Paper version.
            title: Paper title.
            topics: List of matched topic keys.
            quality_i: Quality score (0-100).
            rationale: Judge rationale.
            pdf_bytes: Raw PDF bytes.
            published: Publication date (ISO format, e.g. "2024-01-15").
            authors: Pipe-separated author names.

        Returns:
            Number of chunks stored.
        """
        # Extract text from PDF
        text_parts = self._extract_text(pdf_bytes, arxiv_id)
        if not text_parts:
            logger.warning(f"No text extracted from PDF for {arxiv_id}")
            return 0

        full_text = "\n\n".join(text_parts)

        # Chunk text
        chunks = _chunk_text(full_text)

        # Extract and add figure captions as separate chunks
        captions = _extract_figure_captions(full_text)
        for caption in captions:
            chunks.append(f"[Figure] {caption}")

        if not chunks:
            return 0

        # Build IDs and metadata
        doc_prefix = f"{arxiv_id}v{version}"
        ids = [f"{doc_prefix}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "title": title,
                "topics": "|".join(topics),
                "quality_i": quality_i,
                "chunk_index": i,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf",
                "rationale": (rationale or "")[:500],
                "published": published,
                "authors": (authors or "")[:200],
            }
            for i in range(len(chunks))
        ]

        # Upsert (idempotent — safe to re-run)
        # ChromaDB has batch limits, so process in batches
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            f"Stored {len(chunks)} chunks for {arxiv_id}v{version} in ChromaDB "
            f"({len(captions)} figure captions)"
        )
        return len(chunks)

    def paper_exists(self, arxiv_id: str, version: int) -> bool:
        """Check if paper already exists in ChromaDB."""
        try:
            results = self._collection.get(
                ids=[f"{arxiv_id}v{version}_chunk_0"],
                include=[],
            )
            return bool(results["ids"])
        except Exception:
            return False

    def query(
        self, question: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Query ChromaDB for relevant paper chunks.

        Args:
            question: Search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts with text, title, arxiv_id, score, etc.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            output.append({
                "text": doc,
                "title": meta.get("title", ""),
                "arxiv_id": meta.get("arxiv_id", ""),
                "topics": meta.get("topics", "").split("|"),
                "abs_url": meta.get("abs_url", ""),
                "pdf_url": meta.get("pdf_url", ""),
                "quality_i": meta.get("quality_i", 0),
                "rationale": meta.get("rationale", ""),
                "score": 1.0 - float(dist),  # cosine distance → similarity
            })

        return output

    def prune_old_papers(self, max_age_days: int = 270) -> int:
        """Delete paper chunks older than max_age_days (default 9 months).

        Uses the 'published' metadata field to determine age.
        Papers without a published date are kept (safe default).

        Args:
            max_age_days: Maximum age in days. Default 270 (9 months).

        Returns:
            Number of chunks deleted.
        """
        from datetime import datetime, timedelta

        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
        logger.info(f"Pruning papers published before {cutoff} ({max_age_days} days)")

        # Get all documents with their metadata
        total = self._collection.count()
        if total == 0:
            return 0

        # Fetch in batches to find old chunks
        batch_size = 1000
        ids_to_delete: list[str] = []

        for offset in range(0, total, batch_size):
            results = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )

            for doc_id, meta in zip(results["ids"], results["metadatas"], strict=False):
                published = meta.get("published", "")
                # Only prune if published date exists and is older than cutoff
                if published and published < cutoff:
                    ids_to_delete.append(doc_id)

        if not ids_to_delete:
            logger.info("No old papers to prune")
            return 0

        # Delete in batches (ChromaDB limit)
        for start in range(0, len(ids_to_delete), batch_size):
            end = start + batch_size
            self._collection.delete(ids=ids_to_delete[start:end])

        logger.info(f"Pruned {len(ids_to_delete)} chunks (papers before {cutoff})")
        return len(ids_to_delete)

    def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB PersistentClient auto-persists; no explicit close needed

    def _extract_text(self, pdf_bytes: bytes, arxiv_id: str) -> list[str]:
        """Extract text from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text.strip())
            return parts
        except Exception as e:
            logger.warning(f"PDF text extraction failed for {arxiv_id}: {e}")
            return []


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks by word boundaries."""
    words = text.split()
    words_per_chunk = max(chunk_size // 6, 10)  # ~6 chars per word avg
    overlap_words = max(overlap // 6, 2)

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += words_per_chunk - overlap_words

    return chunks


def _extract_figure_captions(text: str) -> list[str]:
    """Extract figure captions from paper text."""
    pattern = r"(?:Figure|Fig\.?)\s*\d+[.:]\s*(.+?)(?:\n\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    captions = []
    for match in matches:
        caption = match.strip()[:500]
        if caption and len(caption) > 20:  # skip very short matches
            captions.append(caption)
    return captions
