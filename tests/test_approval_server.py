"""Tests for the MCP approval server."""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ask_claude.approval.server import (
    SimpleMCP,
    load_strategy_config,
    log_to_file,
    permissions__approve,
)


class TestSimpleMCP:
    """Test the SimpleMCP fallback implementation."""

    def test_init(self) -> None:
        """Test SimpleMCP initialization."""
        mcp = SimpleMCP("test-server")
        assert mcp.name == "test-server"
        assert mcp.tools == {}

    def test_tool_decorator(self) -> None:
        """Test the tool decorator."""
        mcp = SimpleMCP("test-server")

        @mcp.tool()
        def test_function() -> str:
            return "test"

        assert "test_function" in mcp.tools
        assert mcp.tools["test_function"] == test_function

    @pytest.mark.asyncio
    async def test_run_tools_list(self) -> None:
        """Test handling tools/list request."""
        mcp = SimpleMCP("test-server")

        @mcp.tool()
        def test_tool() -> str:
            return "test"

        request = {"id": 1, "method": "tools/list"}

        with patch("sys.stdin.readline", return_value=json.dumps(request) + "\n"):
            with patch("sys.stdout.write") as mock_write:
                # Run one iteration
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=[json.dumps(request) + "\n", ""]
                    )
                    await mcp.run()

                # Check response
                response = json.loads(mock_write.call_args_list[0][0][0].strip())
                assert response["id"] == 1
                assert response["result"]["tools"] == [{"name": "test_tool"}]

    @pytest.mark.asyncio
    async def test_run_tools_call(self) -> None:
        """Test handling tools/call request."""
        mcp = SimpleMCP("test-server")

        @mcp.tool()
        async def test_tool(arg: str) -> dict:
            return {"result": f"processed {arg}"}

        request = {
            "id": 2,
            "method": "tools/call",
            "params": {"name": "test_tool", "arguments": {"arg": "test_value"}},
        }

        with patch("sys.stdin.readline", return_value=json.dumps(request) + "\n"):
            with patch("sys.stdout.write") as mock_write:
                # Run one iteration
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=[json.dumps(request) + "\n", ""]
                    )
                    await mcp.run()

                # Check response
                response = json.loads(mock_write.call_args_list[0][0][0].strip())
                assert response["id"] == 2
                assert response["result"] == {"result": "processed test_value"}

    @pytest.mark.asyncio
    async def test_run_unknown_method(self) -> None:
        """Test handling unknown method."""
        mcp = SimpleMCP("test-server")
        request = {"id": 3, "method": "unknown/method"}

        with patch("sys.stdin.readline", return_value=json.dumps(request) + "\n"):
            with patch("sys.stdout.write") as mock_write:
                # Run one iteration
                with patch("asyncio.get_event_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        side_effect=[json.dumps(request) + "\n", ""]
                    )
                    await mcp.run()

                # Check error response
                response = json.loads(mock_write.call_args_list[0][0][0].strip())
                assert response["id"] == 3
                assert response["error"]["code"] == -32601
                assert response["error"]["message"] == "Method not found"


class TestLoadStrategyConfig:
    """Test strategy configuration loading."""

    def test_load_from_env_var(self) -> None:
        """Test loading config from environment variable."""
        config = {"type": "all", "extra": "data"}
        with patch.dict(os.environ, {"APPROVAL_STRATEGY_CONFIG": json.dumps(config)}):
            result = load_strategy_config()
            assert result == config

    def test_load_from_file(self) -> None:
        """Test loading config from file."""
        config = {"type": "allowlist", "allowlist": ["tool1", "tool2"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            try:
                with patch.dict(os.environ, {"APPROVAL_CONFIG_FILE": f.name}):
                    result = load_strategy_config()
                    assert result == config
            finally:
                os.unlink(f.name)

    def test_default_config(self) -> None:
        """Test default configuration when no env or file."""
        with patch.dict(os.environ, {}, clear=True):
            result = load_strategy_config()
            assert result == {"type": "allowlist", "allowlist": []}


class TestLogToFile:
    """Test file logging functionality."""

    def test_log_to_default_path(self) -> None:
        """Test logging to default path."""
        with patch("builtins.open", mock_open()) as mock_file:
            log_to_file("Test message")
            mock_file.assert_called_once_with("approval_log.txt", "a")
            handle = mock_file()
            written_content = handle.write.call_args[0][0]
            assert "Test message" in written_content

    def test_log_to_custom_path(self) -> None:
        """Test logging to custom path from env."""
        with patch.dict(os.environ, {"APPROVAL_LOG_PATH": "/custom/log.txt"}):
            with patch("builtins.open", mock_open()) as mock_file:
                log_to_file("Custom log")
                mock_file.assert_called_once_with("/custom/log.txt", "a")


class TestPermissionsApprove:
    """Test the permissions approval function."""

    @pytest.mark.asyncio
    async def test_approve_allowed_tool(self) -> None:
        """Test approving an allowed tool."""
        # Mock the strategy
        with patch(
            "ask_claude.approval.server.strategy",
            Mock(
                should_approve=Mock(return_value=True),
                get_denial_reason=Mock(return_value=""),
            ),
        ):
            with patch("ask_claude.approval.server.log_to_file"):
                result = await permissions__approve(
                    "allowed_tool", {"param": "value"}, "test reason"
                )

                assert result == {
                    "behavior": "allow",
                    "updatedInput": {"param": "value"},
                }

    @pytest.mark.asyncio
    async def test_deny_disallowed_tool(self) -> None:
        """Test denying a disallowed tool."""
        # Mock the strategy
        with patch(
            "ask_claude.approval.server.strategy",
            Mock(
                should_approve=Mock(return_value=False),
                get_denial_reason=Mock(return_value="Tool not in allowlist"),
            ),
        ):
            with patch("ask_claude.approval.server.log_to_file"):
                result = await permissions__approve(
                    "denied_tool", {"param": "value"}, "test reason"
                )

                assert result == {
                    "behavior": "deny",
                    "message": "Tool not in allowlist",
                }

    @pytest.mark.asyncio
    async def test_logging_decision(self) -> None:
        """Test that decisions are logged."""
        with patch(
            "ask_claude.approval.server.strategy",
            Mock(should_approve=Mock(return_value=True)),
        ):
            with patch("ask_claude.approval.server.log_to_file") as mock_log:
                await permissions__approve("test_tool", {"key": "value"})

                mock_log.assert_called_once()
                log_message = mock_log.call_args[0][0]
                assert "Tool: test_tool" in log_message
                assert "Approved: True" in log_message
                assert '"key": "value"' in log_message


class TestModuleLevel:
    """Test module-level code and initialization."""

    def test_mcp_import_fallback(self) -> None:
        """Test that SimpleMCP is used when FastMCP is not available."""
        # This is already tested by the module import, but we can verify
        from ask_claude.approval.server import HAS_FASTMCP, mcp

        if not HAS_FASTMCP:
            assert isinstance(mcp, SimpleMCP)

    def test_strategy_initialization(self) -> None:
        """Test that strategy is initialized on import."""
        from ask_claude.approval.server import strategy

        # Should have a strategy instance (default is allowlist with empty list)
        assert hasattr(strategy, "should_approve")
        assert hasattr(strategy, "get_denial_reason")
