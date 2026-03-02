"""llama.cpp server management for local LLM inference.

Manages a llama-cpp-python server process for serving local models.
On Oracle Cloud, the server runs as a systemd service instead.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_N_CTX = 4096
DEFAULT_N_THREADS = 4


@dataclass
class ServerConfig:
    """Configuration for llama.cpp server."""

    model_path: str = ""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    n_ctx: int = DEFAULT_N_CTX
    n_threads: int = DEFAULT_N_THREADS
    n_gpu_layers: int = 0  # CPU-only for Oracle Cloud ARM

    @property
    def base_url(self) -> str:
        """Get the server base URL."""
        return f"http://{self.host}:{self.port}"


def check_server_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check if a llama.cpp server is healthy.

    Args:
        base_url: Server base URL (e.g., http://127.0.0.1:8080).
        timeout: Request timeout in seconds.

    Returns:
        True if server is healthy and responding.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/health")
            return response.status_code == 200
    except (httpx.HTTPError, httpx.ConnectError):
        return False


class LlamaCppServer:
    """Manages a llama-cpp-python server process.

    For local development and testing. On Oracle Cloud, use the
    systemd service instead (deploy/llama-server.service).
    """

    def __init__(self, config: ServerConfig):
        """Initialize server manager.

        Args:
            config: Server configuration.
        """
        self._config = config
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        """Get server base URL."""
        return self._config.base_url

    def is_running(self) -> bool:
        """Check if the server is running and healthy."""
        return check_server_health(self._config.base_url)

    def start(self, wait_timeout: float = 60.0) -> None:
        """Start the llama.cpp server.

        Args:
            wait_timeout: Max seconds to wait for server to be ready.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If server fails to start.
        """
        if self.is_running():
            logger.info("Server already running")
            return

        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", str(model_path),
            "--host", self._config.host,
            "--port", str(self._config.port),
            "--n_ctx", str(self._config.n_ctx),
            "--n_threads", str(self._config.n_threads),
            "--n_gpu_layers", str(self._config.n_gpu_layers),
        ]

        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        start_time = time.monotonic()
        while time.monotonic() - start_time < wait_timeout:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(f"Server process exited unexpectedly: {stderr[:500]}")
            if self.is_running():
                logger.info(f"Server ready at {self._config.base_url}")
                return
            time.sleep(1.0)

        # Timeout — kill and raise
        self.stop()
        raise RuntimeError(f"Server failed to start within {wait_timeout}s")

    def stop(self) -> None:
        """Stop the server process."""
        if self._process is not None:
            logger.info("Stopping llama.cpp server")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def __enter__(self) -> LlamaCppServer:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
