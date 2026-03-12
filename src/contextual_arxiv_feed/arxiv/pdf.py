"""PDF downloading from arXiv with streaming and validation.

Downloads PDFs with:
- Streaming to handle large files
- Technical abort for corrupted/non-PDF responses
- MAX_DOWNLOAD_MB limit (technical safety, NOT quality gating)
- PDF magic byte validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import httpx

from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle, arxiv_retry, check_response_status

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"

DEFAULT_MAX_DOWNLOAD_MB = 100


class DownloadStatus(Enum):
    """Status of PDF download attempt."""

    SUCCESS = auto()
    NOT_PDF = auto()  # Response was not a PDF
    TOO_LARGE = auto()  # Exceeded size limit (technical abort)
    HTTP_ERROR = auto()  # HTTP request failed
    TIMEOUT = auto()  # Request timed out
    CORRUPTED = auto()  # PDF appears corrupted


@dataclass
class PDFDownloadResult:
    """Result of a PDF download attempt."""

    status: DownloadStatus
    pdf_bytes: bytes | None = None
    size_bytes: int = 0
    error_message: str = ""

    @property
    def success(self) -> bool:
        """Whether download was successful."""
        return self.status == DownloadStatus.SUCCESS

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)


class PDFDownloader:
    """Downloads PDFs from arXiv with validation.

    Note: MAX_DOWNLOAD_MB is a technical safety limit to prevent
    infinite downloads or broken responses. It is NOT a quality gate.
    Papers are never rejected for being "too long" - only for
    technical download failures.
    """

    def __init__(
        self,
        throttle: ArxivThrottle | None = None,
        max_download_mb: int = DEFAULT_MAX_DOWNLOAD_MB,
    ):
        """Initialize downloader.

        Args:
            throttle: Rate limiter. Creates default if None.
            max_download_mb: Technical abort limit in MB. Default 100.
        """
        self._throttle = throttle or ArxivThrottle()
        self._max_bytes = max_download_mb * 1024 * 1024
        self._client = httpx.Client(timeout=120.0, follow_redirects=True)

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> PDFDownloader:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @arxiv_retry
    def download(self, pdf_url: str) -> PDFDownloadResult:
        """Download PDF from URL.

        Args:
            pdf_url: URL to PDF (typically https://arxiv.org/pdf/XXXX.XXXXX.pdf).

        Returns:
            PDFDownloadResult with status and bytes if successful.
        """
        logger.info(f"Downloading PDF: {pdf_url}")

        self._throttle.sync_wait()

        try:
            with self._client.stream("GET", pdf_url) as response:
                check_response_status(response.status_code, pdf_url)

                if response.status_code != 200:
                    return PDFDownloadResult(
                        status=DownloadStatus.HTTP_ERROR,
                        error_message=f"HTTP {response.status_code}",
                    )

                content_type = response.headers.get("content-type", "")
                if "application/pdf" not in content_type.lower():
                    # Some servers don't set content-type correctly, so we'll
                    # also check magic bytes later
                    logger.debug(f"Content-Type is {content_type}, will check magic bytes")

                chunks = []
                total_size = 0

                for chunk in response.iter_bytes(chunk_size=8192):
                    total_size += len(chunk)

                    # Technical abort: size limit exceeded
                    if total_size > self._max_bytes:
                        logger.warning(
                            f"PDF exceeds size limit ({total_size} > {self._max_bytes} bytes). "
                            "This is a technical abort, not a quality judgment."
                        )
                        return PDFDownloadResult(
                            status=DownloadStatus.TOO_LARGE,
                            size_bytes=total_size,
                            error_message=f"Exceeded {self._max_bytes // (1024*1024)}MB limit",
                        )

                    chunks.append(chunk)

                pdf_bytes = b"".join(chunks)

                if not pdf_bytes.startswith(PDF_MAGIC):
                    logger.error(f"Downloaded content is not a PDF: {pdf_url}")
                    return PDFDownloadResult(
                        status=DownloadStatus.NOT_PDF,
                        size_bytes=total_size,
                        error_message="Content does not start with PDF magic bytes",
                    )

                # Basic PDF structure check (should contain %%EOF)
                if b"%%EOF" not in pdf_bytes[-1024:]:
                    logger.warning(f"PDF may be corrupted (no %%EOF): {pdf_url}")
                    # We still accept it - might be valid, just unusual

                logger.info(f"Downloaded PDF: {total_size / (1024*1024):.2f} MB")
                return PDFDownloadResult(
                    status=DownloadStatus.SUCCESS,
                    pdf_bytes=pdf_bytes,
                    size_bytes=total_size,
                )

        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading PDF: {e}")
            return PDFDownloadResult(
                status=DownloadStatus.TIMEOUT,
                error_message=str(e),
            )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF: {e}")
            return PDFDownloadResult(
                status=DownloadStatus.HTTP_ERROR,
                error_message=str(e),
            )

    def download_by_arxiv_id(
        self, arxiv_id: str, version: int = 1
    ) -> PDFDownloadResult:
        """Download PDF by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2401.12345").
            version: Paper version. Default 1.

        Returns:
            PDFDownloadResult.
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf"
        return self.download(pdf_url)


def compress_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """Compress PDF bytes to reduce file size.

    Uses pypdf to compress content streams and remove duplicate objects.
    Lossless for text content — safe for ingestion.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Compressed PDF bytes. Returns original if compression fails or increases size.
    """
    try:
        import io

        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()

        for page in reader.pages:
            page.compress_content_streams()
            writer.add_page(page)

        writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)

        output = io.BytesIO()
        writer.write(output)
        compressed = output.getvalue()

        # Only use compressed version if it's actually smaller
        if len(compressed) < len(pdf_bytes):
            return compressed
        return pdf_bytes

    except ImportError:
        logger.debug("pypdf not installed, skipping PDF compression")
        return pdf_bytes
    except Exception as e:
        logger.warning(f"PDF compression failed, using original: {e}")
        return pdf_bytes
