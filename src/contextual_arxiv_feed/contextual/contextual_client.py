"""Contextual AI Datastore client for document ingestion and management.

Handles:
- PDF document ingestion with custom metadata
- Manifest document ingestion (JSON text)
- Document listing for idempotency checks
- Metadata updates for citation refresh
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from contextual_arxiv_feed.contextual.naming import (
    build_document_name,
    build_manifest_name,
    build_prefix,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of document ingestion."""

    success: bool
    document_id: str = ""
    document_name: str = ""
    error: str = ""


@dataclass
class DocumentInfo:
    """Information about an existing document."""

    document_id: str
    document_name: str
    metadata: dict[str, Any]


class ContextualClient:
    """Client for Contextual AI Datastore API.

    Provides methods for:
    - Ingesting PDFs with custom metadata
    - Ingesting manifest documents (JSON)
    - Listing documents by prefix (for idempotency)
    - Updating document metadata
    """

    def __init__(
        self,
        api_key: str,
        datastore_id: str,
        base_url: str = "https://api.contextual.ai",
    ):
        """Initialize client.

        Args:
            api_key: Contextual API key.
            datastore_id: Target datastore ID.
            base_url: API base URL.
        """
        self._api_key = api_key
        self._datastore_id = datastore_id
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=300.0)  # Long timeout for uploads

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> ContextualClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def _headers(self) -> dict[str, str]:
        """Common headers for API requests."""
        return {
            "Authorization": f"Bearer {self._api_key}",
        }

    def _documents_url(self, document_id: str = "") -> str:
        """Build documents endpoint URL."""
        base = f"{self._base_url}/v1/datastores/{self._datastore_id}/documents"
        if document_id:
            return f"{base}/{document_id}"
        return base

    def _datastore_url(self) -> str:
        """Build datastore endpoint URL."""
        return f"{self._base_url}/v1/datastores/{self._datastore_id}"

    def configure_text_only_ingestion(self) -> bool:
        """Configure datastore for text-only ingestion (Basic tier, $3/1K pages).

        Sets ingestion_tier to null and figure_caption_mode to ignore
        at the datastore level. Call once during setup.

        Returns:
            True if configuration succeeded.
        """
        logger.info("Configuring datastore for text-only ingestion (Basic tier)")

        try:
            response = self._client.put(
                self._datastore_url(),
                headers={**self._headers, "Content-Type": "application/json"},
                json={
                    "configuration": {
                        "parsing": {
                            "figure_caption_mode": "ignore",
                            "ingestion_tier": None,
                        }
                    }
                },
            )

            if response.status_code in (200, 204):
                logger.info("Datastore configured for text-only ingestion")
                return True
            else:
                logger.error(
                    f"Failed to configure datastore: HTTP {response.status_code}: "
                    f"{response.text}"
                )
                return False

        except httpx.HTTPError as e:
            logger.error(f"HTTP error configuring datastore: {e}")
            return False

    def ingest_pdf(
        self,
        arxiv_id: str,
        version: int,
        pdf_bytes: bytes,
        custom_metadata: dict[str, Any],
    ) -> IngestResult:
        """Ingest a PDF document.

        Args:
            arxiv_id: arXiv paper ID.
            version: Paper version.
            pdf_bytes: PDF content.
            custom_metadata: Flat dict with primitive values only.

        Returns:
            IngestResult indicating success or failure.
        """
        document_name = build_document_name(arxiv_id, version)
        logger.info(f"Ingesting PDF: {document_name}")

        # Validate metadata is flat and primitive-only
        self._validate_metadata(custom_metadata)

        try:
            # Prepare multipart form data
            files = {
                "file": (f"{arxiv_id}v{version}.pdf", pdf_bytes, "application/pdf"),
            }
            # Text-only ingestion: Basic tier ($3/1K pages).
            # figure_caption_mode=ignore skips VLM figure processing.
            # Datastore must also have ingestion_tier=null (call configure_text_only_ingestion once).
            ingestion_config = {
                "figure_caption_mode": "ignore",
            }
            data = {
                "document_name": document_name,
                "metadata": json.dumps(custom_metadata),
                "configuration": json.dumps(ingestion_config),
            }

            response = self._client.post(
                self._documents_url(),
                headers=self._headers,
                files=files,
                data=data,
            )

            if response.status_code in (200, 201):
                result = response.json()
                return IngestResult(
                    success=True,
                    document_id=result.get("document_id", ""),
                    document_name=document_name,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Failed to ingest PDF {document_name}: {error_msg}")
                return IngestResult(success=False, error=error_msg)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error ingesting PDF {document_name}: {e}")
            return IngestResult(success=False, error=str(e))

    def ingest_manifest(
        self,
        arxiv_id: str,
        version: int,
        manifest_content: dict[str, Any],
        custom_metadata: dict[str, Any],
    ) -> IngestResult:
        """Ingest a manifest document (JSON text).

        The manifest contains all metadata that doesn't fit in custom_metadata:
        - Full arXiv API response
        - Full judge output
        - Citation data
        - Run metadata

        Args:
            arxiv_id: arXiv paper ID.
            version: Paper version.
            manifest_content: Full manifest data.
            custom_metadata: Flat dict with primitive values only.

        Returns:
            IngestResult indicating success or failure.
        """
        document_name = build_manifest_name(arxiv_id, version)
        logger.info(f"Ingesting manifest: {document_name}")

        # Validate metadata is flat and primitive-only
        self._validate_metadata(custom_metadata)

        try:
            # Convert manifest to JSON text
            manifest_json = json.dumps(manifest_content, indent=2, default=str)

            # Prepare multipart form data
            files = {
                "file": (
                    f"{arxiv_id}v{version}_manifest.json",
                    manifest_json.encode("utf-8"),
                    "application/json",
                ),
            }
            ingestion_config = {
                "figure_caption_mode": "ignore",
            }
            data = {
                "document_name": document_name,
                "metadata": json.dumps(custom_metadata),
                "configuration": json.dumps(ingestion_config),
            }

            response = self._client.post(
                self._documents_url(),
                headers=self._headers,
                files=files,
                data=data,
            )

            if response.status_code in (200, 201):
                result = response.json()
                return IngestResult(
                    success=True,
                    document_id=result.get("document_id", ""),
                    document_name=document_name,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Failed to ingest manifest {document_name}: {error_msg}")
                return IngestResult(success=False, error=error_msg)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error ingesting manifest {document_name}: {e}")
            return IngestResult(success=False, error=str(e))

    def list_documents(self, prefix: str = "", limit: int = 1000) -> list[str]:
        """List document names by prefix.

        Used for idempotency checks.

        Args:
            prefix: Filter documents by name prefix.
            limit: Maximum documents to return.

        Returns:
            List of document names.
        """
        logger.debug(f"Listing documents with prefix: {prefix}")

        try:
            params: dict[str, str | int] = {"limit": limit}
            if prefix:
                params["prefix"] = prefix

            response = self._client.get(
                self._documents_url(),
                headers=self._headers,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                return [doc.get("document_name", "") for doc in documents]
            else:
                logger.error(f"Failed to list documents: HTTP {response.status_code}")
                return []

        except httpx.HTTPError as e:
            logger.error(f"HTTP error listing documents: {e}")
            return []

    def document_exists(self, arxiv_id: str, version: int) -> bool:
        """Check if a document exists (idempotency check).

        Args:
            arxiv_id: arXiv paper ID.
            version: Paper version.

        Returns:
            True if document exists.
        """
        prefix = build_prefix(arxiv_id, version)
        documents = self.list_documents(prefix=prefix, limit=1)
        target = build_document_name(arxiv_id, version)
        return target in documents

    def get_document(self, document_name: str) -> DocumentInfo | None:
        """Get document information by name.

        Args:
            document_name: Document name.

        Returns:
            DocumentInfo or None if not found.
        """
        try:
            # List with exact prefix
            response = self._client.get(
                self._documents_url(),
                headers=self._headers,
                params={"prefix": document_name, "limit": 1},
            )

            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                for doc in documents:
                    if doc.get("document_name") == document_name:
                        return DocumentInfo(
                            document_id=doc.get("document_id", ""),
                            document_name=document_name,
                            metadata=doc.get("metadata", {}),
                        )
            return None

        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting document: {e}")
            return None

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID.

        Args:
            document_id: Document ID to delete.

        Returns:
            True if deletion succeeded.
        """
        logger.info(f"Deleting document: {document_id}")

        try:
            response = self._client.delete(
                self._documents_url(document_id),
                headers=self._headers,
            )

            if response.status_code in (200, 204):
                return True
            else:
                logger.error(f"Failed to delete document: HTTP {response.status_code}")
                return False

        except httpx.HTTPError as e:
            logger.error(f"HTTP error deleting document: {e}")
            return False

    def delete_document_by_name(self, document_name: str) -> bool:
        """Delete a document by name.

        Args:
            document_name: Document name to delete.

        Returns:
            True if deletion succeeded.
        """
        doc = self.get_document(document_name)
        if not doc:
            logger.warning(f"Document not found for deletion: {document_name}")
            return False
        return self.delete_document(doc.document_id)

    def update_metadata(
        self, document_name: str, metadata_updates: dict[str, Any]
    ) -> bool:
        """Update document metadata.

        Used for citation refresh without re-ingesting PDFs.

        Args:
            document_name: Document name.
            metadata_updates: Metadata fields to update.

        Returns:
            True if update succeeded.
        """
        logger.info(f"Updating metadata for: {document_name}")

        # Validate updates are flat and primitive-only
        self._validate_metadata(metadata_updates)

        try:
            # First get the document to find its ID
            doc = self.get_document(document_name)
            if not doc:
                logger.error(f"Document not found: {document_name}")
                return False

            # Merge with existing metadata
            merged = {**doc.metadata, **metadata_updates}

            response = self._client.patch(
                self._documents_url(doc.document_id),
                headers={**self._headers, "Content-Type": "application/json"},
                json={"metadata": merged},
            )

            if response.status_code in (200, 204):
                return True
            else:
                logger.error(
                    f"Failed to update metadata: HTTP {response.status_code}"
                )
                return False

        except httpx.HTTPError as e:
            logger.error(f"HTTP error updating metadata: {e}")
            return False

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate metadata is flat with primitive values only.

        Raises ValueError if validation fails.
        """
        for key, value in metadata.items():
            if isinstance(value, dict):
                raise ValueError(
                    f"Metadata must be flat. Key '{key}' contains a nested dict."
                )
            if isinstance(value, list):
                raise ValueError(
                    f"Metadata must be flat. Key '{key}' contains a list."
                )
            if isinstance(value, float):
                raise ValueError(
                    f"Float values not allowed in metadata. Key '{key}' has float value {value}. "
                    "Convert to int."
                )
            if not isinstance(value, (str, int, bool, type(None))):
                raise ValueError(
                    f"Metadata values must be str, int, or bool. "
                    f"Key '{key}' has type {type(value).__name__}."
                )


def create_client(
    api_key: str, datastore_id: str, base_url: str = "https://api.contextual.ai"
) -> ContextualClient:
    """Factory function to create Contextual client.

    Args:
        api_key: Contextual API key.
        datastore_id: Target datastore ID.
        base_url: API base URL.

    Returns:
        Configured ContextualClient.
    """
    return ContextualClient(api_key, datastore_id, base_url)
