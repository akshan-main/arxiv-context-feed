"""Tests for LLM server management."""

from unittest.mock import patch

from contextual_arxiv_feed.llm.server import ServerConfig, check_server_health


class TestServerConfig:
    """Test ServerConfig defaults."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.n_ctx == 4096
        assert config.n_threads == 4
        assert config.n_gpu_layers == 0

    def test_base_url(self):
        config = ServerConfig(host="0.0.0.0", port=9090)
        assert config.base_url == "http://0.0.0.0:9090"

    def test_custom_config(self):
        config = ServerConfig(
            model_path="/models/test.gguf",
            host="localhost",
            port=8081,
            n_ctx=8192,
            n_threads=8,
        )
        assert config.model_path == "/models/test.gguf"
        assert config.n_ctx == 8192
        assert config.n_threads == 8


class TestCheckServerHealth:
    """Test server health check."""

    def test_unreachable_server(self):
        # No server running on this port
        assert check_server_health("http://127.0.0.1:19999", timeout=1.0) is False

    @patch("contextual_arxiv_feed.llm.server.httpx.Client")
    def test_healthy_server(self, mock_client_cls):
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = mock_client.get.return_value
        mock_response.status_code = 200

        assert check_server_health("http://127.0.0.1:8080") is True
