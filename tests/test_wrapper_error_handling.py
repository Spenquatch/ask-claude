"""Tests for error handling and edge cases in ClaudeCodeWrapper."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ask_claude.wrapper import (
    ClaudeCodeConfig,
    ClaudeCodeConfigurationError,
    ClaudeCodeTimeoutError,
    ClaudeCodeWrapper,
)


@pytest.fixture(autouse=True)
def mock_validate_binary() -> Iterator[None]:
    """Automatically mock binary validation for all tests"""
    with patch.object(ClaudeCodeWrapper, "_validate_binary"):
        yield


class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases in the wrapper."""

    def test_command_timeout(self) -> None:
        """Test handling of command timeout."""
        config = ClaudeCodeConfig(timeout=0.1)
        wrapper = ClaudeCodeWrapper(config)

        # Mock subprocess to raise TimeoutExpired
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 0.1)

            with pytest.raises(ClaudeCodeTimeoutError) as exc_info:
                wrapper.run("test query")

            assert "Claude Code execution timed out after 0.1s" in str(exc_info.value)

    def test_subprocess_error_with_returncode(self) -> None:
        """Test handling of subprocess errors with return codes."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        # Mock subprocess to return error
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout=b"",
                stderr=b"Error: Command failed",
            )

            response = wrapper.run("test query")
            assert not response.success  # Should fail due to non-zero return code
            assert response.returncode == 1
            assert response.stderr == b"Error: Command failed"  # type: ignore

    def test_json_decode_error(self) -> None:
        """Test handling of invalid JSON output - falls back to text."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        # Mock subprocess to return invalid JSON
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=b"Invalid JSON {",
                stderr=b"",
            )

            response = wrapper.run("test query")
            # When JSON parsing fails, it falls back to text format
            assert response.success
            assert response.content == b"Invalid JSON {"  # type: ignore

    def test_config_validation_basic(self) -> None:
        """Test basic configuration validation."""
        # Test valid config creation
        config = ClaudeCodeConfig(timeout=10.0, max_retries=2)
        assert config.timeout == 10.0
        assert config.max_retries == 2

    def test_file_not_found_error(self) -> None:
        """Test handling when Claude binary is not found."""
        config = ClaudeCodeConfig(claude_binary="/nonexistent/claude")

        # Override the mock to actually check binary
        with patch.object(
            ClaudeCodeWrapper,
            "_validate_binary",
            side_effect=ClaudeCodeConfigurationError("Binary not found"),
        ):
            with pytest.raises(ClaudeCodeConfigurationError):
                ClaudeCodeWrapper(config)

    def test_keyboard_interrupt_handling(self) -> None:
        """Test handling of keyboard interrupts."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            with pytest.raises(KeyboardInterrupt):
                wrapper.run("test query")

    def test_unicode_in_responses(self) -> None:
        """Test handling of unicode in responses."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        # Mock subprocess to return unicode content
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps({"content": "Hello 世界 🌍"}).encode("utf-8"),
                stderr=b"",
            )

            response = wrapper.run("test query")
            assert response.success
            assert "世界" in response.content
            assert "🌍" in response.content

    def test_empty_response_handling(self) -> None:
        """Test handling of empty responses."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        # Mock subprocess to return empty JSON
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps({}).encode(),
                stderr=b"",
            )

            response = wrapper.run("test query")
            assert response.content == ""


class TestAdditionalErrorHandling:
    """Additional error handling tests for coverage"""

    def test_session_ask_error_handling(self) -> None:
        """Test session ask with error response"""
        with patch("subprocess.Popen") as mock_popen:
            # First call succeeds (session creation)
            mock_process1 = Mock()
            mock_process1.communicate.return_value = (
                b'{"session_id": "test-123"}',
                b"",
            )
            mock_process1.returncode = 0

            # Second call fails (ask)
            mock_process2 = Mock()
            mock_process2.communicate.return_value = (
                b'{"error": "Something went wrong"}',
                b"",
            )
            mock_process2.returncode = 1

            mock_popen.side_effect = [mock_process1, mock_process2]

            wrapper = ClaudeCodeWrapper()
            session = wrapper.create_session()

            response = session.ask("Test query")
            assert response.is_error is True

    def test_clear_cache_functionality(self) -> None:
        """Test cache clearing"""
        wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(cache_responses=True))

        # Test that clear_cache method exists and can be called
        wrapper.clear_cache()

        # Since we can't access private attributes directly,
        # just verify the method runs without error
        assert True  # Method completed successfully

    def test_config_environment_variables(self) -> None:
        """Test config with environment variables"""
        config = ClaudeCodeConfig(
            environment_vars={"CUSTOM_VAR": "value", "API_KEY": "secret"}
        )

        assert config.environment_vars["CUSTOM_VAR"] == "value"
        assert config.environment_vars["API_KEY"] == "secret"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
