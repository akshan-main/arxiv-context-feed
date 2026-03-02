"""Seed a new Contextual AI datastore with 5 random RAG papers from arXiv.

Usage:
    CONTEXTUAL_API_KEY=... python scripts/seed_rag_datastore.py

Creates a brand-new datastore, searches arXiv for RAG papers,
downloads the PDFs, and uploads them.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
CONTEXTUAL_BASE = "https://api.contextual.ai/v1"
NUM_PAPERS = 5


# ── arXiv helpers ────────────────────────────────────────────────


def search_arxiv_rag_papers(num_results: int = 10) -> list[dict]:
    """Search arXiv for RAG papers and return metadata."""
    params = {
        "search_query": 'all:"retrieval augmented generation"',
        "start": 500,  # Skip the newest papers so OpenAlex has had time to index
        "max_results": num_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = httpx.get(ARXIV_API, params=params, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
        arxiv_id_url = entry.findtext("atom:id", "", ns)
        arxiv_id = arxiv_id_url.split("/abs/")[-1]

        # Find PDF and abstract links
        pdf_url = ""
        abs_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            if link.get("rel") == "alternate":
                abs_url = link.get("href", "")

        # Primary category
        primary_cat_el = entry.find("arxiv:primary_category", ns)
        primary_category = primary_cat_el.get("term", "") if primary_cat_el is not None else ""

        # All categories (dedupe, preserve order)
        categories = [primary_category] if primary_category else []
        categories += [c.get("term", "") for c in entry.findall("atom:category", ns)]
        categories = list(dict.fromkeys(categories))

        # Authors with affiliations
        authors = []
        affiliations = []
        for a in entry.findall("atom:author", ns):
            name = a.findtext("atom:name", "", ns)
            if name:
                authors.append(name)
            aff = a.findtext("arxiv:affiliation", "", ns)
            if aff:
                affiliations.append(aff)
        affiliations = list(dict.fromkeys(affiliations))  # dedupe

        # Dates
        published = entry.findtext("atom:published", "", ns).strip()
        updated = entry.findtext("atom:updated", "", ns).strip()

        # Optional fields (may not exist on every paper)
        comment = entry.findtext("arxiv:comment", "", ns)
        doi = entry.findtext("arxiv:doi", "", ns)
        journal_ref = entry.findtext("arxiv:journal_ref", "", ns)

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "url": abs_url or arxiv_id_url,
                "primary_category": primary_category,
                "categories": "|".join(categories),
                "authors": ", ".join(authors),
                "affiliations": ", ".join(affiliations) if affiliations else "",
                "published": published,
                "updated": updated,
                "comment": comment or "",
                "doi": doi or "",
                "journal_ref": journal_ref or "",
            }
        )
    return papers


# ── OpenAlex helpers ─────────────────────────────────────────────


OPENALEX_API = "https://api.openalex.org"


def get_openalex_citations(arxiv_id: str, doi: str = "") -> dict:
    """Look up a paper on OpenAlex and return citation metadata.

    Tries arXiv ID first (strips version suffix), falls back to DOI.
    Returns dict with citation_count, reference_count, venue, or empty dict.
    """
    # Strip version from arxiv_id (e.g. "2602.17529v1" -> "2602.17529")
    base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

    # Every arXiv paper has a DOI in format 10.48550/arXiv.{id}
    arxiv_doi = f"10.48550/arXiv.{base_id}"
    identifiers = [f"doi:{arxiv_doi}"]
    if doi:
        identifiers.append(f"doi:{doi}")
    for identifier in identifiers:
        url = f"{OPENALEX_API}/works/{identifier}"
        try:
            resp = httpx.get(url, timeout=15.0, follow_redirects=True)
            if resp.status_code != 200:
                continue
            data = resp.json()

            # Extract venue from primary_location
            venue = ""
            primary = data.get("primary_location") or {}
            source = primary.get("source") or {}
            venue = source.get("display_name", "")

            return {
                "citation_count": data.get("cited_by_count", 0),
                "reference_count": len(data.get("referenced_works", [])),
                "venue": venue,
                "openalex_id": data.get("id", ""),
            }
        except httpx.HTTPError:
            continue

    return {}


def download_pdf(pdf_url: str) -> bytes:
    """Download a PDF from arXiv."""
    logger.info(f"  Downloading {pdf_url}")
    resp = httpx.get(pdf_url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError("Response is not a valid PDF")
    return resp.content


# ── Contextual AI helpers ────────────────────────────────────────


def create_datastore(api_key: str, name: str) -> str:
    """Create a new Contextual AI datastore. Returns the datastore ID."""
    resp = httpx.post(
        f"{CONTEXTUAL_BASE}/datastores",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"name": name},
        timeout=30.0,
    )
    resp.raise_for_status()
    datastore_id = resp.json()["id"]
    logger.info(f"Created datastore: {name} ({datastore_id})")
    return datastore_id


def upload_pdf(api_key: str, datastore_id: str, filename: str, pdf_bytes: bytes, metadata: dict) -> str:
    """Upload a PDF to a Contextual AI datastore. Returns document ID."""
    ingestion_config = {"figure_caption_mode": "ignore"}
    resp = httpx.post(
        f"{CONTEXTUAL_BASE}/datastores/{datastore_id}/documents",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (filename, pdf_bytes, "application/pdf")},
        data={
            "metadata": json.dumps({"custom_metadata": metadata}),
            "configuration": json.dumps(ingestion_config),
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    doc_id = resp.json().get("id", resp.json().get("document_id", ""))
    return doc_id


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    api_key = os.environ.get("CONTEXTUAL_API_KEY")
    if not api_key:
        logger.error("Set CONTEXTUAL_API_KEY environment variable")
        sys.exit(1)

    # 1. Search arXiv for RAG papers
    logger.info(f"Searching arXiv for {NUM_PAPERS} RAG papers...")
    papers = search_arxiv_rag_papers(num_results=NUM_PAPERS)

    if len(papers) < NUM_PAPERS:
        logger.warning(f"Only found {len(papers)} papers (wanted {NUM_PAPERS})")
    papers = papers[:NUM_PAPERS]

    logger.info(f"Found {len(papers)} papers:")
    for i, p in enumerate(papers, 1):
        logger.info(f"  {i}. {p['title'][:80]}")

    # 2. Create a new datastore
    datastore_id = create_datastore(api_key, "arXiv RAG Papers (seed)")

    # 3. Download PDFs and upload to datastore
    uploaded = 0
    for i, paper in enumerate(papers, 1):
        logger.info(f"\n[{i}/{len(papers)}] {paper['title'][:80]}")

        if not paper["pdf_url"]:
            logger.warning("  No PDF URL, skipping")
            continue

        try:
            # Download
            pdf_bytes = download_pdf(paper["pdf_url"])
            logger.info(f"  Downloaded {len(pdf_bytes) / (1024*1024):.1f} MB")

            # Look up citations on OpenAlex
            logger.info("  Querying OpenAlex for citations...")
            citations = get_openalex_citations(paper["arxiv_id"], paper.get("doi", ""))
            if citations:
                logger.info(
                    f"  OpenAlex: {citations['citation_count']} citations, "
                    f"{citations['reference_count']} references"
                    + (f", venue: {citations['venue']}" if citations.get("venue") else "")
                )
            else:
                logger.info("  OpenAlex: paper not found (may be too new)")

            # Build metadata — workspace has a 15 unique field name budget
            # shared across all datastores. Reddit already uses 13 slots,
            # leaving room for ~2 new unique names. We reuse "url" and "title"
            # from Reddit and add "source", "arxiv_id", "categories",
            # "authors", "published" as new fields (7 total per doc).
            metadata = {
                "source": "arxiv",
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"][:200],
                "authors": paper["authors"][:300],
                "categories": paper["categories"],
                "url": paper["url"],
                "published": paper["published"],
            }
            # Drop empty values
            metadata = {k: v for k, v in metadata.items() if v}
            doc_id = upload_pdf(api_key, datastore_id, f"{paper['arxiv_id']}.pdf", pdf_bytes, metadata)
            logger.info(f"  Uploaded -> document {doc_id}")
            uploaded += 1

            # Be polite to arXiv
            if i < len(papers):
                time.sleep(3)

        except Exception as e:
            logger.error(f"  Failed: {e}")

    logger.info(f"\nDone! Uploaded {uploaded}/{len(papers)} papers to datastore {datastore_id}")


if __name__ == "__main__":
    main()
