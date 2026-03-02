"""Contextual AI Reranker v2 for retrieved paper chunks.

Uses ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b —
state-of-the-art instruction-following reranker, purpose-built
for relevance scoring.

~2-3GB RAM (1B model). Runs on Oracle Cloud alongside
the main Qwen2.5-14B generation model.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None


def _load_model():
    """Lazy-load Contextual AI reranker on first use."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Contextual AI reranker: {MODEL_ID}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        _model.eval()
        logger.info("Contextual AI reranker loaded")
    return _model, _tokenizer


class Reranker:
    """Reranks retrieved chunks using Contextual AI Reranker v2."""

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """Rerank retrieved chunks by relevance to query.

        Uses Contextual AI's instruction-following reranker to score
        query-document pairs.

        Args:
            query: User's question.
            results: Retrieved chunks from ChromaDB.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of results (best first), trimmed to top_k.
        """
        if not results or len(results) <= 1:
            return results[:top_k]

        model, tokenizer = _load_model()

        # Build prompts in Contextual AI reranker format
        instruction = "Is this document relevant to the query about AI/ML research?"
        prompts = []
        for r in results:
            title = r.get("title", "")
            text = r.get("text", "")[:500]
            doc = f"{title}. {text}" if title else text
            prompts.append(
                f"Check whether a given document contains information "
                f"helpful to answer the query.\n"
                f"<Document> {doc}\n"
                f"<Query> {query} {instruction} ??"
            )

        # Tokenize and score
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        # Extract relevance scores from last token logits
        scores = out.logits[:, -1, 0].float().tolist()

        # Sort by score (descending)
        scored = list(zip(scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def close(self) -> None:
        pass
