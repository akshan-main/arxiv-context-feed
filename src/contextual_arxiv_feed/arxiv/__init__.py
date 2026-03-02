"""arXiv integration layer for feeds, API, and PDF downloads."""

from contextual_arxiv_feed.arxiv.api import ArxivAPI, ArxivMetadata
from contextual_arxiv_feed.arxiv.feeds import ArxivFeedParser, FeedEntry
from contextual_arxiv_feed.arxiv.pdf import PDFDownloader, PDFDownloadResult, compress_pdf_bytes
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle

__all__ = [
    "ArxivAPI",
    "ArxivMetadata",
    "ArxivFeedParser",
    "FeedEntry",
    "PDFDownloader",
    "PDFDownloadResult",
    "ArxivThrottle",
    "compress_pdf_bytes",
]
