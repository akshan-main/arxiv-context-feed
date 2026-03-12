"""Test ingesting 5 arXiv papers with expanded metadata fields (35 limit).

Downloads PDFs and ingests to Contextual AI datastore with all individual
metadata fields (no more source packing).
"""

import json
import logging
import os
import time

import httpx
from dotenv import load_dotenv

from contextual_arxiv_feed.arxiv.api import ArxivAPI
from contextual_arxiv_feed.contextual.contextual_client import ContextualClient
from contextual_arxiv_feed.contextual.metadata import build_paper_metadata

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 5 top-scoring relevant papers from our evaluation
PAPER_IDS = [
    "2602.22469",  # Beyond Dominant Patches (q=81, VLM hallucination)
    "2603.03310",  # Entropic-Time Inference (q=81, LLM decoding)
    "2603.03336",  # Prompt-Dependent Ranking (q=80, LLM evaluation)
    "2603.04128",  # Crab+ Audio-Visual LLM (q=80, multimodal)
    "2603.03536",  # SafeCRS (q=80, LLM safety alignment)
]

# Scores from the judge run
PAPER_SCORES = {
    "2602.22469": {"q": 81, "c": 90, "topics": "llm-inference"},
    "2603.03310": {"q": 81, "c": 78, "topics": "llm-inference"},
    "2603.03336": {"q": 80, "c": 85, "topics": "generative-ai-foundations"},
    "2603.04128": {"q": 80, "c": 85, "topics": "generative-ai-foundations|fine-tuning"},
    "2603.03536": {"q": 80, "c": 85, "topics": "fine-tuning"},
}


def download_pdf(arxiv_id: str, version: int = 1) -> bytes | None:
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf"
    logger.info(f"Downloading PDF: {url}")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=60.0)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            logger.error(f"PDF too small ({len(resp.content)} bytes), likely error page")
            return None
        logger.info(f"  Downloaded {len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"  Failed to download: {e}")
        return None


def main():
    api_key = os.getenv("CONTEXTUAL_API_KEY", "")
    datastore_id = os.getenv("CONTEXTUAL_DATASTORE_ID", "")

    if not api_key or not datastore_id:
        print("ERROR: CONTEXTUAL_API_KEY and CONTEXTUAL_DATASTORE_ID must be set")
        return

    # Fetch metadata from arXiv API
    api = ArxivAPI()
    logger.info(f"Fetching metadata for {len(PAPER_IDS)} papers...")
    id_list = [f"{pid}v1" for pid in PAPER_IDS]
    papers_meta = api.fetch_by_ids(id_list)
    api.close()

    meta_map = {m.arxiv_id: m for m in papers_meta}
    logger.info(f"Got metadata for {len(meta_map)} papers")

    # Initialize Contextual client
    client = ContextualClient(api_key, datastore_id)

    success = 0
    failed = 0

    for arxiv_id in PAPER_IDS:
        meta = meta_map.get(arxiv_id)
        if not meta:
            logger.warning(f"No metadata for {arxiv_id}, skipping")
            failed += 1
            continue

        scores = PAPER_SCORES[arxiv_id]

        # Check if already exists
        if client.document_exists(arxiv_id, 1):
            logger.info(f"Already exists: {arxiv_id}, skipping")
            success += 1
            continue

        # Download PDF
        pdf_bytes = download_pdf(arxiv_id)
        if not pdf_bytes:
            failed += 1
            continue

        # Build metadata with all individual fields
        custom_metadata = build_paper_metadata(
            arxiv_id=arxiv_id,
            version=1,
            title=meta.title,
            categories=meta.categories,
            primary_category=meta.primary_category,
            authors="|".join(str(a) for a in meta.authors[:5]),
            publication_date=str(meta.published)[:10],
            updated_date=str(meta.updated)[:10] if meta.updated else "",
            doi=meta.doi,
            journal_ref=meta.journal_ref,
            comments=meta.comments,
            topics=scores["topics"],
            quality_verdict="accept",
            quality_i=scores["q"],
            novelty_i=0,
            relevance_i=0,
            technical_depth_i=0,
            confidence_i=scores["c"],
        )

        logger.info(f"Metadata fields: {len(custom_metadata)}")
        logger.info(f"Fields: {list(custom_metadata.keys())}")

        # Ingest
        result = client.ingest_pdf(arxiv_id, 1, pdf_bytes, custom_metadata)
        if result.success:
            logger.info(f"SUCCESS: {arxiv_id} -> doc_id={result.document_id}")
            success += 1
        else:
            logger.error(f"FAILED: {arxiv_id} -> {result.error}")
            failed += 1

        # Small delay between ingestions
        time.sleep(2)

    client.close()

    print(f"\n{'='*60}")
    print(f"INGESTION RESULTS")
    print(f"{'='*60}")
    print(f"  Success: {success}/{len(PAPER_IDS)}")
    print(f"  Failed:  {failed}/{len(PAPER_IDS)}")


if __name__ == "__main__":
    main()
