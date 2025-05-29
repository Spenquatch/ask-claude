import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ask_claude.cli import ClaudeCLI, create_parser, main  # noqa: E402
from ask_claude.wrapper import (  # noqa: E402
    ClaudeCodeConfig,
    ClaudeCodeConfigurationError,
    ClaudeCodeResponse,
    ClaudeCodeTimeoutError,
    ClaudeCodeWrapper,
)


@pytest.fixture(autouse=True)
def mock_validate_binary() -> Iterator[None]:
    """Automatically mock binary validation for all tests"""
    with patch.object(ClaudeCodeWrapper, "_validate_binary"):
        yield


class TestCLIParser:
    """Test command-line parser"""

    def test_create_parser(self) -> None:
        """Test parser creation"""
        parser = create_parser()

        # Test parser has all subcommands
        args = parser.parse_args(["ask", "Hello"])
        assert args.command == "ask"
        assert args.query == "Hello"

        args = parser.parse_args(["stream", "Test"])
        assert args.command == "stream"

        args = parser.parse_args(["session", "--interactive"])
        assert args.command == "session"
        assert args.interactive

        args = parser.parse_args(["health"])
        assert args.command == "health"

        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"

    def test_ask_command_args(self) -> None:
        """Test ask command arguments"""
        parser = create_parser()

        args = parser.parse_args(
            [
                "ask",
                "Test prompt",
                "--format",
                "json",
                "--timeout",
                "60",
                "--show-metadata",
                "--session-id",
                "test-session",
                "--continue",
            ]
        )

        assert args.query == "Test prompt"
        assert args.format == "json"
        assert args.timeout == 60
        assert args.show_metadata
        assert args.session_id == "test-session"
        assert getattr(args, "continue")

    def test_session_command_args(self) -> None:
        """Test session command arguments"""
        parser = create_parser()

        # Test basic session args
        args = parser.parse_args(["session", "--interactive"])
        assert args.command == "session"
        assert args.interactive

        # Test session with max turns and approval
        args = parser.parse_args(
            [
                "session",
                "--max-turns",
                "10",
                "--approval-strategy",
                "allowlist",
                "--approval-allowlist",
                "tool1",
                "tool2",
            ]
        )
        assert args.max_turns == 10
        assert args.approval_strategy == "allowlist"
        assert args.approval_allowlist == ["tool1", "tool2"]

    def test_stream_command_args(self) -> None:
        """Test stream command arguments"""
        parser = create_parser()

        # Test basic stream args
        args = parser.parse_args(["stream", "Test query"])
        assert args.command == "stream"
        assert args.query == "Test query"

        # Test stream with all options
        args = parser.parse_args(
            [
                "stream",
                "Test streaming query",
                "--timeout",
                "120",
                "--show-stats",
                "--approval-strategy",
                "patterns",
                "--approval-allow-patterns",
                ".*read.*",
                ".*list.*",
            ]
        )
        assert args.query == "Test streaming query"
        assert args.timeout == 120
        assert args.show_stats
        assert args.approval_strategy == "patterns"
        assert args.approval_allow_patterns == [".*read.*", ".*list.*"]


class TestClaudeCLI:
    """Test ClaudeCLI class methods"""

    @pytest.fixture
    def cli(self) -> ClaudeCLI:
        """Create CLI instance"""
        return ClaudeCLI()

    def test_cli_initialization(self, cli: ClaudeCLI) -> None:
        """Test CLI initialization"""
        assert cli.wrapper is None
        assert cli.config is None  # Config is loaded on demand

        # Test config loading
        config = cli.load_config()
        assert isinstance(config, ClaudeCodeConfig)

    @patch("sys.stderr", new_callable=StringIO)
    def test_print_response_metadata(
        self, mock_stderr: StringIO, cli: ClaudeCLI
    ) -> None:
        """Test response metadata printing"""
        response = ClaudeCodeResponse(
            content="Test response", returncode=0, execution_time=1.5, retries=0
        )

        cli._print_response_metadata(response)
        output = mock_stderr.getvalue()
        assert "Metadata" in output
        assert "Execution Time: 1.500s" in output
        assert "Is Error: False" in output

    @patch("ask_claude.cli.ClaudeCodeWrapper")
    def test_cmd_ask(self, mock_wrapper_class: Mock, cli: ClaudeCLI) -> None:
        """Test ask command handler"""
        # Setup mock
        mock_wrapper = Mock()
        mock_response = ClaudeCodeResponse(
            content="Test response", returncode=0, execution_time=1.0, retries=0
        )
        mock_wrapper.run.return_value = mock_response
        mock_wrapper_class.return_value = mock_wrapper

        # Initialize the wrapper
        cli.wrapper = mock_wrapper

        with patch("sys.stdout", new_callable=StringIO):
            # Test execution - cmd_ask takes query string, not args object
            result = cli.cmd_ask("Test prompt")
            assert result == 0

    def test_cmd_ask_json(self, cli: ClaudeCLI) -> None:
        """Test ask command with JSON output"""
        mock_response = Mock()
        mock_response.content = '{"result": "success"}'
        mock_response.is_error = False

        mock_wrapper = Mock()
        mock_wrapper.run.return_value = mock_response
        cli.wrapper = mock_wrapper

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cli.cmd_ask(
                "Return JSON", output_format="json", show_metadata=False
            )
            assert result == 0
            # Check JSON was printed
            printed_json = mock_stdout.getvalue()
            assert json.loads(printed_json) == {"result": "success"}

    def test_cmd_stream(self, cli: ClaudeCLI) -> None:
        """Test stream command handler"""
        # Mock streaming events with correct structure
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "chunk1"},
                        {"type": "text", "text": "chunk2"},
                        {"type": "text", "text": "chunk3"},
                    ],
                    "stop_reason": "end_turn",
                },
            },
        ]

        mock_wrapper = Mock()
        mock_wrapper.run_streaming.return_value = iter(events)
        cli.wrapper = mock_wrapper

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cli.cmd_stream("Stream test")
            assert result == 0
            # Check chunks were printed
            output = mock_stdout.getvalue()
            assert "chunk1" in output
            assert "chunk2" in output
            assert "chunk3" in output

    def test_cmd_session_interactive(self, cli: ClaudeCLI) -> None:
        """Test interactive session"""
        mock_response = Mock()
        mock_response.content = "Interactive response"

        mock_session = Mock()
        mock_session.run.return_value = mock_response

        # Create a proper context manager mock
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)

        mock_wrapper = Mock()
        mock_wrapper.session.return_value = mock_context
        cli.wrapper = mock_wrapper

        # Mock initialize_wrapper to return True and not overwrite the wrapper
        with patch.object(cli, "initialize_wrapper", return_value=True):
            # Mock user input
            with patch("builtins.input", side_effect=["Hello", "exit"]):
                with patch("sys.stdout", new_callable=StringIO):
                    result = cli.cmd_session(interactive=True)
                    assert result == 0

    def test_cmd_health(self, cli: ClaudeCLI) -> None:
        """Test health check command"""
        mock_response = Mock()
        mock_response.content = "OK"
        mock_response.is_error = False

        mock_wrapper = Mock()
        mock_wrapper.run.return_value = mock_response
        mock_wrapper.get_metrics.return_value = {
            "total_requests": 10,
            "successful_requests": 9,
            "failed_requests": 1,
            "success_rate": 0.9,
            "average_retries_per_request": 0.5,
            "cache_hit_rate": 0.2,
        }
        cli.wrapper = mock_wrapper

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = cli.cmd_health()
            assert result == 0
            output = mock_stdout.getvalue()
            assert "Health Check" in output
            assert "Working" in output

    def test_cmd_benchmark(self, cli: ClaudeCLI) -> None:
        """Test benchmark command"""
        mock_response = Mock()
        mock_response.content = "Response"
        mock_response.is_error = False
        mock_response.execution_time = 0.1

        mock_wrapper = Mock()
        mock_wrapper.run.return_value = mock_response
        mock_wrapper.get_metrics.return_value = {"total_requests": 8}

        # Set wrapper before calling cmd_benchmark
        cli.wrapper = mock_wrapper

        # Patch initialize_wrapper to prevent it from creating a real wrapper
        with patch.object(cli, "initialize_wrapper", return_value=True):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = cli.cmd_benchmark(iterations=2)
                assert result == 0
                # Should run 4 queries x 2 iterations = 8 calls
                assert mock_wrapper.run.call_count == 8
                output = mock_stdout.getvalue()
                assert "performance benchmark" in output


class TestMainFunction:
    """Test main entry point"""

    @patch("sys.argv", ["cli_tool.py", "ask", "Hello"])
    @patch("ask_claude.cli.ClaudeCLI")
    def test_main_ask(self, mock_cli_class: Mock) -> None:
        """Test main with ask command"""
        mock_cli = Mock()
        mock_cli.cmd_ask.return_value = 0
        mock_cli_class.return_value = mock_cli

        result = main()
        assert result == 0
        assert mock_cli.cmd_ask.called

    @patch("sys.argv", ["cli_tool.py", "stream", "Hello"])
    @patch("ask_claude.cli.ClaudeCLI")
    def test_main_stream(self, mock_cli_class: Mock) -> None:
        """Test main with stream command"""
        mock_cli = Mock()
        mock_cli.cmd_stream.return_value = 0
        mock_cli_class.return_value = mock_cli

        result = main()
        assert result == 0
        assert mock_cli.cmd_stream.called

    @patch("sys.argv", ["cli_tool.py", "invalid"])
    def test_main_invalid_command(self) -> None:
        """Test main with invalid command"""
        with patch("sys.stderr", new_callable=StringIO):
            with pytest.raises(SystemExit):
                main()

    @patch("sys.argv", ["cli_tool.py", "ask", "Test"])
    @patch("ask_claude.cli.ClaudeCLI")
    def test_main_error_handling(self, mock_cli_class: Mock) -> None:
        """Test main error handling"""
        mock_cli = Mock()
        mock_cli.cmd_ask.side_effect = Exception("Test error")
        mock_cli_class.return_value = mock_cli

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = main()
            assert result == 1
            assert "Test error" in mock_stderr.getvalue()


class TestCLIIntegration:
    """CLI integration tests"""

    def test_config_file_loading(self) -> None:
        """Test loading configuration from file"""
        config_data = {"timeout": 120, "max_retries": 5, "retry_delay": 2.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            cli = ClaudeCLI()
            config = cli.load_config(Path(config_file))

            assert config.timeout == 120
            assert config.max_retries == 5
            assert config.retry_delay == 2.0
        finally:
            os.unlink(config_file)

    def test_error_output_formatting(self) -> None:
        """Test error output formatting"""
        cli = ClaudeCLI()

        mock_wrapper = Mock()
        mock_wrapper.run.side_effect = ClaudeCodeTimeoutError(30.0)
        cli.wrapper = mock_wrapper

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cli.cmd_ask("Test query")
            assert result == 1
            assert "Timeout Error" in mock_stderr.getvalue()


class TestToolDisplayInfo:
    """Test tool display information functionality"""

    def test_exact_tool_match_bash(self) -> None:
        """Test exact match for Bash tool"""
        cli = ClaudeCLI()

        emoji, action, fields = cli._get_tool_display_info(
            "Bash", {"command": "ls -la"}
        )

        assert emoji == "🖥️"
        assert action == "run Bash command"
        assert fields == {"command": "Command", "description": "Purpose"}

    def test_exact_tool_match_read(self) -> None:
        """Test exact match for Read tool"""
        cli = ClaudeCLI()

        emoji, action, fields = cli._get_tool_display_info(
            "Read", {"file_path": "/path/to/file"}
        )

        assert emoji == "📄"
        assert action == "read file"
        assert fields == {"file_path": "File"}

    def test_sequential_thinking_pattern_match(self) -> None:
        """Test pattern match for sequential-thinking MCP tool"""
        cli = ClaudeCLI()

        emoji, action, fields = cli._get_tool_display_info(
            "mcp__sequential-thinking__sequentialthinking",
            {"thought": "Testing", "thoughtNumber": 1},
        )

        assert emoji == "🤔"
        assert action == "think"
        assert fields == {
            "thought": "Thought",
            "thoughtNumber": "Step",
            "totalThoughts": "Total",
        }

    def test_default_fallback_for_unknown_tool(self) -> None:
        """Test default fallback for unknown tool"""
        cli = ClaudeCLI()

        emoji, action, fields = cli._get_tool_display_info(
            "UnknownTool", {"param": "value"}
        )

        assert emoji == "🔧"
        assert action == "use tool"
        assert fields == {
            "description": "Purpose",
            "query": "Query",
            "command": "Command",
        }


class TestCLIErrorHandling:
    """Test CLI error handling paths"""

    def test_config_loading_error_handling(self) -> None:
        """Test error handling when config file is malformed"""
        cli = ClaudeCLI()

        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            config_path = Path(f.name)

        try:
            # Should handle the JSON error gracefully and return default config
            config = cli.load_config(config_path)

            # Should fall back to default config
            assert isinstance(config, ClaudeCodeConfig)
            assert config.claude_binary == "claude"  # Default value

        finally:
            # Clean up
            config_path.unlink()

    def test_wrapper_initialization_error_handling(self) -> None:
        """Test error handling when wrapper initialization fails"""
        cli = ClaudeCLI()
        cli.config = ClaudeCodeConfig()

        # Mock ClaudeCodeWrapper to raise a configuration error
        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper:
            mock_wrapper.side_effect = ClaudeCodeConfigurationError(
                "Test configuration error", config_field="test_field"
            )

            # Capture stderr
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                result = cli.initialize_wrapper()

                # Should return False and print error
                assert result is False
                assert cli.wrapper is None

                # Check error output
                error_output = mock_stderr.getvalue()
                assert (
                    "❌ Configuration Error: Test configuration error" in error_output
                )
                assert "Field: test_field" in error_output

    def test_session_non_interactive_error(self) -> None:
        """Test error when trying to use non-interactive session"""
        cli = ClaudeCLI()

        # Capture stderr
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cli.cmd_session(interactive=False)

            # Should return error code 1
            assert result == 1

            # Check error message
            error_output = mock_stderr.getvalue()
            assert (
                "❌ Error: Non-interactive sessions not yet implemented" in error_output
            )

    def test_session_initialization_failure(self) -> None:
        """Test session command when wrapper initialization fails"""
        cli = ClaudeCLI()

        # Mock initialize_wrapper to return False (failure)
        with patch.object(cli, "initialize_wrapper", return_value=False):
            result = cli.cmd_session(interactive=True)

            # Should return error code 1
            assert result == 1

    def test_stream_empty_query_error(self) -> None:
        """Test streaming command with empty query"""
        cli = ClaudeCLI()

        # Capture stderr
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = cli.cmd_stream("")  # Empty query

            # Should return error code 1
            assert result == 1

            # Check error message
            error_output = mock_stderr.getvalue()
            assert "❌ Error: Query cannot be empty" in error_output

    def test_main_keyboard_interrupt_handling(self) -> None:
        """Test main function KeyboardInterrupt handling"""
        # Mock sys.argv to simulate CLI usage
        test_args = ["ask-claude", "ask", "test query"]

        # Mock ClaudeCLI.cmd_ask to raise KeyboardInterrupt
        with patch("sys.argv", test_args):
            with patch("ask_claude.cli.ClaudeCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.load_config.return_value = None
                mock_cli.cmd_ask.side_effect = KeyboardInterrupt()
                mock_cli_class.return_value = mock_cli

                # Capture stderr
                with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                    result = main()

                    # Should return exit code 130 (standard for SIGINT)
                    assert result == 130

                    # Check interrupt message
                    error_output = mock_stderr.getvalue()
                    assert "⏹️  Operation interrupted by user" in error_output

    def test_main_general_exception_handling(self) -> None:
        """Test main function general exception handling"""
        # Mock sys.argv to simulate CLI usage
        test_args = ["ask-claude", "ask", "test query"]

        # Mock ClaudeCLI.cmd_ask to raise a general exception
        with patch("sys.argv", test_args):
            with patch("ask_claude.cli.ClaudeCLI") as mock_cli_class:
                mock_cli = Mock()
                mock_cli.load_config.return_value = None
                mock_cli.cmd_ask.side_effect = RuntimeError("Test error")
                mock_cli_class.return_value = mock_cli

                # Capture stderr
                with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                    result = main()

                    # Should return exit code 1
                    assert result == 1

                    # Check error message
                    error_output = mock_stderr.getvalue()
                    assert "❌ Unexpected error: Test error" in error_output

    def test_wrapper_initialization_verbose_mode(self) -> None:
        """Test wrapper initialization with verbose mode"""
        cli = ClaudeCLI()
        cli.config = ClaudeCodeConfig()

        # Mock ClaudeCodeWrapper to track if it was called
        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper:
            mock_wrapper.return_value = Mock()

            result = cli.initialize_wrapper(verbose=True)

            # Should succeed
            assert result is True

            # Should have set log level to INFO
            assert cli.config.log_level == logging.INFO

            # Should have called ClaudeCodeWrapper with the config
            mock_wrapper.assert_called_once_with(cli.config)

    def test_config_loading_with_path_conversion(self) -> None:
        """Test config loading with path conversion for working_directory"""
        cli = ClaudeCLI()

        # Create a real temporary directory for working_directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {"working_directory": temp_dir, "timeout": 120}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(config_data, f)
                config_path = Path(f.name)

            try:
                config = cli.load_config(config_path)

                # Should convert string path to Path object
                assert isinstance(config.working_directory, Path)
                assert str(config.working_directory) == temp_dir

                # Other values should be preserved
                assert config.timeout == 120

            finally:
                config_path.unlink()

    def test_stream_verbose_mode_initialization(self) -> None:
        """Test stream command with verbose mode shows initialization message"""
        cli = ClaudeCLI()
        cli.wrapper = Mock()

        # Mock the stream method to avoid complex streaming logic
        cli.wrapper.stream.side_effect = StopIteration("Mock end")

        # Capture stderr for verbose output
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            # Should catch StopIteration and return error code
            cli.cmd_stream("test query", verbose=True)

            # Check that verbose initialization message was printed
            error_output = mock_stderr.getvalue()
            assert "🌊 Starting stream..." in error_output

    def test_stream_exception_handling(self) -> None:
        """Test stream command exception handling"""
        cli = ClaudeCLI()

        # Mock initialize_wrapper to succeed and set wrapper
        with patch.object(cli, "initialize_wrapper", return_value=True):
            cli.wrapper = Mock()

            # Mock the stream method to raise an exception during iteration
            cli.wrapper.stream.return_value = iter([])  # Empty iterator
            cli.wrapper.stream.side_effect = RuntimeError("Stream failed")

            # Capture stderr for error output
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                result = cli.cmd_stream("test query")

                # Should return error code 1
                assert result == 1

                # Check error message contains the exception
                error_output = mock_stderr.getvalue()
                assert "❌ Stream Error:" in error_output


class TestMCPApprovalCLI:
    """Test MCP auto-approval CLI functionality"""

    def test_approval_flags_parsing(self) -> None:
        """Test parsing of approval-related CLI flags"""
        parser = create_parser()

        # Test with allowlist strategy
        args = parser.parse_args(
            [
                "ask",
                "Test",
                "--mcp-config",
                "mcp.json",
                "--approval-strategy",
                "allowlist",
                "--approval-allowlist",
                "tool1",
                "tool2",
            ]
        )
        assert args.approval_strategy == "allowlist"
        assert args.approval_allowlist == ["tool1", "tool2"]

        # Test with patterns strategy
        args = parser.parse_args(
            [
                "ask",
                "Test",
                "--approval-strategy",
                "patterns",
                "--approval-allow-patterns",
                ".*read.*",
                ".*list.*",
                "--approval-deny-patterns",
                ".*write.*",
            ]
        )
        assert args.approval_strategy == "patterns"
        assert args.approval_allow_patterns == [".*read.*", ".*list.*"]
        assert args.approval_deny_patterns == [".*write.*"]

        # Test with all strategy
        args = parser.parse_args(["ask", "Test", "--approval-strategy", "all"])
        assert args.approval_strategy == "all"

    def test_approval_config_construction(self) -> None:
        """Test that approval config is properly constructed"""
        cli = ClaudeCLI()

        # Mock args with approval settings
        args = Mock()
        args.query = "Test"  # Changed from prompt to query
        args.mcp_config = Path("mcp.json")  # Convert to Path
        args.approval_strategy = "allowlist"
        args.approval_allowlist = ["mcp__test__tool1", "mcp__test__tool2"]
        args.approval_allow_patterns = None
        args.approval_deny_patterns = None
        args.format = "text"  # Added format
        args.timeout = None
        args.max_turns = None
        args.session_id = None
        args.show_metadata = False

        # Build approval config
        approval_config = cli._build_approval_config(args)
        config_dict = {}
        if approval_config:
            config_dict["mcp_auto_approval"] = approval_config
        cli.config = ClaudeCodeConfig.from_dict(config_dict)

        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_response.is_error = False
            mock_wrapper.run.return_value = mock_response
            mock_wrapper_class.return_value = mock_wrapper

            # Initialize wrapper first
            cli.wrapper = mock_wrapper

            result = cli.cmd_ask(
                args.query, args.format, show_metadata=args.show_metadata
            )
            assert result == 0

            # Verify wrapper.run was called
            assert mock_wrapper.run.called
            # Verify config has correct approval settings
            assert cli.config.mcp_auto_approval["enabled"]
            assert cli.config.mcp_auto_approval["strategy"] == "allowlist"
            assert cli.config.mcp_auto_approval["allowlist"] == [
                "mcp__test__tool1",
                "mcp__test__tool2",
            ]

    def test_stream_command_with_approval(self) -> None:
        """Test stream command with approval flags"""
        parser = create_parser()
        args = parser.parse_args(
            [
                "stream",
                "Test query",
                "--mcp-config",
                "mcp.json",
                "--approval-strategy",
                "all",
            ]
        )

        cli = ClaudeCLI()

        # Build approval config
        approval_config = cli._build_approval_config(args)
        config_dict = {}
        if approval_config:
            config_dict["mcp_auto_approval"] = approval_config
        cli.config = ClaudeCodeConfig.from_dict(config_dict)

        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            # Mock streaming response
            mock_wrapper.run_streaming.return_value = [
                {"type": "system", "subtype": "init", "session_id": "123"},
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Hello"}]},
                },
                {"type": "result", "subtype": "success"},
            ]
            mock_wrapper_class.return_value = mock_wrapper

            # Initialize wrapper
            cli.wrapper = mock_wrapper

            result = cli.cmd_stream(args.query)  # Pass query as string, not args
            assert result == 0

            # Verify config has approval settings
            assert cli.config.mcp_auto_approval["enabled"]
            assert cli.config.mcp_auto_approval["strategy"] == "all"

    def test_pattern_approval_config(self) -> None:
        """Test pattern-based approval configuration"""
        cli = ClaudeCLI()

        args = Mock()
        args.query = "Test"  # Changed from prompt to query
        args.approval_strategy = "patterns"
        args.approval_allow_patterns = ["mcp__.*__read.*", "mcp__.*__list.*"]
        args.approval_deny_patterns = ["mcp__.*__admin.*"]
        args.approval_allowlist = None
        args.mcp_config = None
        args.format = "text"  # Added format
        args.timeout = None
        args.max_turns = None
        args.session_id = None
        args.show_metadata = False

        # Build approval config
        approval_config = cli._build_approval_config(args)
        config_dict = {"mcp_auto_approval": approval_config}
        cli.config = ClaudeCodeConfig.from_dict(config_dict)

        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_response = Mock()
            mock_response.content = "Test"
            mock_response.is_error = False
            mock_wrapper.run.return_value = mock_response
            mock_wrapper_class.return_value = mock_wrapper

            # Initialize wrapper
            cli.wrapper = mock_wrapper

            cli.cmd_ask(args.query, args.format, show_metadata=args.show_metadata)

            # Verify config has correct approval settings
            assert cli.config.mcp_auto_approval["strategy"] == "patterns"
            assert cli.config.mcp_auto_approval["allow_patterns"] == [
                "mcp__.*__read.*",
                "mcp__.*__list.*",
            ]
            assert cli.config.mcp_auto_approval["deny_patterns"] == ["mcp__.*__admin.*"]

    def test_no_approval_strategy(self) -> None:
        """Test that no approval config is set when strategy is not provided"""
        cli = ClaudeCLI()

        args = Mock()
        args.query = "Test"  # Changed from prompt to query
        args.approval_strategy = None
        args.approval_allowlist = None
        args.approval_allow_patterns = None
        args.approval_deny_patterns = None
        args.mcp_config = None
        args.format = "text"  # Added format
        args.timeout = None
        args.max_turns = None
        args.session_id = None
        args.show_metadata = False

        # Build approval config (should return None)
        approval_config = cli._build_approval_config(args)
        assert approval_config is None

        # Use default config
        cli.config = ClaudeCodeConfig()

        with patch("ask_claude.cli.ClaudeCodeWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_response = Mock()
            mock_response.content = "Test"
            mock_response.is_error = False
            mock_wrapper.run.return_value = mock_response
            mock_wrapper_class.return_value = mock_wrapper

            # Initialize wrapper
            cli.wrapper = mock_wrapper

            cli.cmd_ask(args.query, args.format, show_metadata=args.show_metadata)

            # Verify config has empty mcp_auto_approval
            assert cli.config.mcp_auto_approval == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
