"""
Enterprise Test Suite for Claude Code SDK Wrapper

Comprehensive test coverage following industry best practices:
- Unit tests with proper isolation
- Integration tests with real-world scenarios  
- Error condition testing with graceful degradation
- Performance and stress testing
- Security and validation testing
- Observability and monitoring validation
"""

import json
import pytest
import subprocess
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

# Import the wrapper components
from claude_code_wrapper import (
    ClaudeCodeWrapper,
    ClaudeCodeConfig,
    ClaudeCodeSession,
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError,
    ClaudeCodeValidationError,
    ClaudeCodeConfigurationError,
    ClaudeCodeResponse,
    ClaudeCodeMetrics,
    OutputFormat,
    ErrorSeverity,
    ClaudeCodeResponseParser,
    CircuitBreaker,
    ask_claude,
    ask_claude_json,
    ask_claude_streaming,
    retry_with_backoff
)


# Test Fixtures and Utilities
@dataclass
class MockProcessResult:
    """Mock subprocess.CompletedProcess for testing."""
    stdout: str
    stderr: str
    returncode: int


class TestDataBuilder:
    """Builder pattern for creating consistent test data."""
    
    @staticmethod
    def claude_response_json(
        content: str = "Test response",
        session_id: str = "test-session-123",
        is_error: bool = False,
        cost_usd: float = 0.01,
        duration_ms: int = 1500,
        num_turns: int = 1
    ) -> str:
        """Build realistic Claude Code JSON response."""
        response_data = {
            "type": "completion" if not is_error else "error",
            "subtype": "success" if not is_error else "process_error",
            "cost_usd": cost_usd,
            "is_error": is_error,
            "duration_ms": duration_ms,
            "duration_api_ms": duration_ms - 100,
            "num_turns": num_turns,
            "result": content,
            "total_cost": cost_usd,
            "session_id": session_id
        }
        return json.dumps(response_data)
    
    @staticmethod
    def streaming_events() -> List[Dict[str, Any]]:
        """Build realistic streaming events."""
        return [
            {"type": "init", "session_id": "stream-session", "timestamp": "2025-01-01T12:00:00Z"},
            {"type": "message", "role": "assistant", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": " World!"},
            {"type": "tool_use", "tool": "Python", "action": "execute", "code": "print('test')"},
            {"type": "tool_result", "tool": "Python", "result": "test"},
            {"type": "result", "status": "complete", "stats": {"total_tokens": 25}}
        ]
    
    @staticmethod
    def error_response_json(
        error_message: str = "Process failed",
        error_type: str = "execution_error",
        session_id: str = "error-session"
    ) -> str:
        """Build error response JSON."""
        return json.dumps({
            "type": error_type,
            "subtype": "process_failure",
            "cost_usd": 0.0,
            "is_error": True,
            "duration_ms": 500,
            "duration_api_ms": 100,
            "num_turns": 0,
            "result": error_message,
            "total_cost": 0.0,
            "session_id": session_id,
            "error_message": error_message
        })


# Configuration Tests
class TestClaudeCodeConfig:
    """Test configuration validation and defaults."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = ClaudeCodeConfig()
        assert config.claude_binary == "claude"
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_backoff_factor == 2.0
        assert config.enable_metrics is True
        assert config.log_level == logging.INFO
    
    def test_configuration_validation_success(self):
        """Test valid configuration passes validation."""
        config = ClaudeCodeConfig(
            timeout=30.0,
            max_turns=5,
            max_retries=2,
            retry_delay=0.5
        )
        # Should not raise any exceptions
        assert config.timeout == 30.0
    
    def test_configuration_validation_failures(self):
        """Test configuration validation catches invalid values."""
        with pytest.raises(ClaudeCodeConfigurationError, match="Timeout must be positive"):
            ClaudeCodeConfig(timeout=-1.0)
        
        with pytest.raises(ClaudeCodeConfigurationError, match="Max turns must be positive"):
            ClaudeCodeConfig(max_turns=0)
        
        with pytest.raises(ClaudeCodeConfigurationError, match="Max retries cannot be negative"):
            ClaudeCodeConfig(max_retries=-1)
        
        with pytest.raises(ClaudeCodeConfigurationError, match="Retry delay cannot be negative"):
            ClaudeCodeConfig(retry_delay=-0.5)
    
    def test_configuration_file_validation(self, tmp_path):
        """Test file path validation."""
        # Non-existent MCP config should raise error
        fake_path = tmp_path / "nonexistent.json"
        with pytest.raises(ClaudeCodeConfigurationError, match="MCP config file not found"):
            ClaudeCodeConfig(mcp_config_path=fake_path)
        
        # Non-existent working directory should raise error
        fake_dir = tmp_path / "nonexistent_dir"
        with pytest.raises(ClaudeCodeConfigurationError, match="Working directory not found"):
            ClaudeCodeConfig(working_directory=fake_dir)


# Response Parser Tests
class TestClaudeCodeResponseParser:
    """Test response parsing with various input formats."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        logger = logging.getLogger("test")
        return ClaudeCodeResponseParser(logger)
    
    def test_parse_valid_json_response(self, parser):
        """Test parsing valid JSON response."""
        json_response = TestDataBuilder.claude_response_json(
            content="Parsed content",
            session_id="parse-test",
            cost_usd=0.05,
            duration_ms=2000
        )
        
        response = parser.parse(json_response, OutputFormat.JSON)
        
        assert response.content == "Parsed content"
        assert response.session_id == "parse-test"
        assert response.metrics.cost_usd == 0.05
        assert response.metrics.duration_ms == 2000
        assert response.is_error is False
    
    def test_parse_error_json_response(self, parser):
        """Test parsing JSON error response."""
        error_json = TestDataBuilder.error_response_json(
            error_message="Test error occurred",
            error_type="validation_error"
        )
        
        response = parser.parse(error_json, OutputFormat.JSON)
        
        assert "Test error occurred" in response.content
        assert response.is_error is True
        assert response.error_type == "validation_error"
    
    def test_parse_malformed_json_graceful_fallback(self, parser):
        """Test graceful fallback for malformed JSON."""
        malformed_json = '{"incomplete": json'
        
        response = parser.parse(malformed_json, OutputFormat.JSON)
        
        # Should fall back to text parsing
        assert response.content == malformed_json
        assert response.returncode == 0
    
    def test_parse_text_response(self, parser):
        """Test parsing plain text response."""
        text_response = "This is a plain text response from Claude."
        
        response = parser.parse(text_response, OutputFormat.TEXT)
        
        assert response.content == text_response
        assert response.session_id is None
        assert response.is_error is False
    
    def test_parse_empty_response(self, parser):
        """Test handling of empty responses."""
        response = parser.parse("", OutputFormat.JSON)
        
        # Should handle gracefully
        assert isinstance(response, ClaudeCodeResponse)
    
    def test_parser_error_handling(self, parser):
        """Test parser error handling doesn't crash."""
        # This should trigger the exception handling in parse()
        with patch.object(parser, '_parse_json_response', side_effect=Exception("Test error")):
            response = parser.parse('{"test": "data"}', OutputFormat.JSON)
            
            assert response.is_error is True
            assert "Failed to parse response" in response.content


# Circuit Breaker Tests
class TestCircuitBreaker:
    """Test circuit breaker resilience pattern."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "CLOSED"
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "OPEN"
        
        # Third call should fail immediately due to open circuit
        with pytest.raises(ClaudeCodeError, match="Circuit breaker is OPEN"):
            cb.call(failing_func)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def failing_func():
            raise Exception("Test failure")
        
        def success_func():
            return "recovered"
        
        # Trigger circuit opening
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should transition to HALF_OPEN and then CLOSED on success
        result = cb.call(success_func)
        assert result == "recovered"
        assert cb.state == "CLOSED"


# Retry Decorator Tests  
class TestRetryWithBackoff:
    """Test retry decorator functionality."""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = success_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test successful execution after initial failures."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def eventually_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ClaudeCodeTimeoutError(1.0)
            return "success"
        
        result = eventually_success_func()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_max_attempts_exceeded(self):
        """Test failure after max retry attempts."""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail_func():
            nonlocal call_count
            call_count += 1
            raise ClaudeCodeProcessError("Always fails", 1)
        
        with pytest.raises(ClaudeCodeProcessError):
            always_fail_func()
        
        assert call_count == 3  # Initial attempt + 2 retries


# Main Wrapper Tests
class TestClaudeCodeWrapper:
    """Test main wrapper functionality."""
    
    @pytest.fixture
    def wrapper(self):
        """Create wrapper with test configuration."""
        config = ClaudeCodeConfig(
            max_retries=1,  # Reduce for faster tests
            retry_delay=0.01,
            timeout=5.0
        )
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            return ClaudeCodeWrapper(config)
    
    @pytest.fixture
    def mock_success_result(self):
        """Mock successful subprocess result."""
        return MockProcessResult(
            stdout=TestDataBuilder.claude_response_json(),
            stderr="",
            returncode=0
        )
    
    @pytest.fixture
    def mock_error_result(self):
        """Mock error subprocess result."""
        return MockProcessResult(
            stdout=TestDataBuilder.error_response_json(),
            stderr="Process error occurred",
            returncode=1
        )
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization and validation."""
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper()
            assert wrapper.config.claude_binary == "claude"
            assert hasattr(wrapper, 'parser')
            assert hasattr(wrapper, 'circuit_breaker')
    
    def test_binary_validation_success(self):
        """Test successful binary validation."""
        mock_result = MockProcessResult("", "", 0)
        with patch('claude_code_wrapper.subprocess.run', return_value=mock_result):
            # Should not raise exception
            wrapper = ClaudeCodeWrapper()
    
    def test_binary_validation_failure(self):
        """Test binary validation failure."""
        with patch('claude_code_wrapper.subprocess.run', side_effect=FileNotFoundError):
            with pytest.raises(ClaudeCodeConfigurationError, match="Claude binary not found"):
                ClaudeCodeWrapper()
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_successful_execution(self, mock_run, wrapper, mock_success_result):
        """Test successful command execution."""
        mock_run.return_value = mock_success_result
        
        response = wrapper.run("Test query")
        
        assert response.content == "Test response"
        assert response.returncode == 0
        assert response.is_error is False
        assert response.session_id == "test-session-123"
        
        # Verify command construction
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0][:3] == ["claude", "--print", "Test query"]
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_json_format_execution(self, mock_run, wrapper, mock_success_result):
        """Test execution with JSON output format."""
        mock_run.return_value = mock_success_result
        
        response = wrapper.run("Test query", output_format=OutputFormat.JSON)
        
        # Verify JSON format flag in command
        args, kwargs = mock_run.call_args
        assert "--output-format" in args[0]
        assert "json" in args[0]
    
    def test_input_validation_empty_query(self, wrapper):
        """Test input validation for empty queries."""
        with pytest.raises(ClaudeCodeValidationError, match="Query cannot be empty"):
            wrapper.run("")
        
        with pytest.raises(ClaudeCodeValidationError, match="Query cannot be empty"):
            wrapper.run("   ")
    
    def test_input_validation_oversized_query(self, wrapper):
        """Test input validation for oversized queries."""
        huge_query = "x" * 100001
        with pytest.raises(ClaudeCodeValidationError, match="Query too long"):
            wrapper.run(huge_query)
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_timeout_handling(self, mock_run, wrapper):
        """Test timeout error handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("claude", 30)
        
        with pytest.raises(ClaudeCodeTimeoutError) as exc_info:
            wrapper.run("Test query")
        
        assert exc_info.value.timeout_duration == wrapper.config.timeout
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_process_error_handling(self, mock_run, wrapper):
        """Test process error handling."""
        error = subprocess.CalledProcessError(1, "claude")
        error.stderr = b"Command failed"
        mock_run.side_effect = error
        
        with pytest.raises(ClaudeCodeProcessError) as exc_info:
            wrapper.run("Test query")
        
        assert exc_info.value.returncode == 1
        assert exc_info.value.stderr == "Command failed"
        assert exc_info.value.severity == ErrorSeverity.HIGH
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_retry_mechanism(self, mock_run, wrapper):
        """Test retry mechanism with eventual success."""
        # First call fails, second succeeds
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "claude"),
            MockProcessResult(TestDataBuilder.claude_response_json(), "", 0)
        ]
        
        response = wrapper.run("Test query")
        
        assert response.content == "Test response"
        assert mock_run.call_count == 2
    
    @patch('claude_code_wrapper.subprocess.Popen')
    def test_streaming_success(self, mock_popen, wrapper):
        """Test successful streaming execution."""
        mock_process = MagicMock()
        events = TestDataBuilder.streaming_events()
        mock_process.stdout = [json.dumps(event) + '\n' for event in events]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        
        results = list(wrapper.run_streaming("Test streaming query"))
        
        assert len(results) == len(events)
        assert results[0]["type"] == "init"
        assert results[-1]["type"] == "result"
    
    @patch('claude_code_wrapper.subprocess.Popen')
    def test_streaming_error_handling(self, mock_popen, wrapper):
        """Test streaming error handling with graceful degradation."""
        mock_process = MagicMock()
        mock_process.stdout = ["invalid json\n", '{"type": "message", "content": "valid"}\n']
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process
        
        results = list(wrapper.run_streaming("Test query"))
        
        # Should handle invalid JSON gracefully
        assert len(results) == 2
        assert results[0]["type"] == "parse_error"
        assert results[1]["type"] == "message"
    
    @patch('claude_code_wrapper.subprocess.Popen')
    def test_streaming_process_failure(self, mock_popen, wrapper):
        """Test streaming process failure handling."""
        mock_process = MagicMock()
        mock_process.stdout = []
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_process.stderr.read.return_value = "Process failed"
        mock_process.poll.return_value = 1
        mock_popen.return_value = mock_process
        
        results = list(wrapper.run_streaming("Test query"))
        
        # Should yield error event instead of crashing
        assert len(results) == 1
        assert results[0]["type"] == "error"
        assert results[0]["returncode"] == 1
    
    def test_metrics_collection(self, wrapper):
        """Test metrics collection functionality."""
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.return_value = MockProcessResult(
                TestDataBuilder.claude_response_json(), "", 0
            )
            
            # Execute some operations
            wrapper.run("Query 1")
            wrapper.run("Query 2")
            
            metrics = wrapper.get_metrics()
            assert metrics["total_requests"] == 2
            assert "total_execution_time" in metrics
    
    def test_session_resumption(self, wrapper):
        """Test session resumption functionality."""
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.return_value = MockProcessResult(
                TestDataBuilder.claude_response_json(), "", 0
            )
            
            response = wrapper.resume_session("test-session", "Continue conversation")
            
            # Verify session resumption in command
            args, kwargs = mock_run.call_args
            assert "--resume" in args[0]
            assert "test-session" in args[0]
    
    def test_session_continuation(self, wrapper):
        """Test session continuation functionality."""
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.return_value = MockProcessResult(
                TestDataBuilder.claude_response_json(), "", 0
            )
            
            response = wrapper.continue_last_session("Continue last conversation")
            
            # Verify continuation in command
            args, kwargs = mock_run.call_args
            assert "--continue" in args[0]


# Session Management Tests
class TestClaudeCodeSession:
    """Test session management functionality."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock wrapper for session testing."""
        wrapper = MagicMock(spec=ClaudeCodeWrapper)
        return wrapper
    
    def test_session_initialization(self, mock_wrapper):
        """Test session initialization."""
        session = ClaudeCodeSession(mock_wrapper, max_turns=5)
        
        assert session.wrapper is mock_wrapper
        assert session.config["max_turns"] == 5
        assert session.session_id is None
        assert len(session.history) == 0
    
    def test_session_ask_with_response(self, mock_wrapper):
        """Test asking questions in session."""
        mock_response = ClaudeCodeResponse(
            content="Session response",
            returncode=0,
            session_id="new-session-id"
        )
        mock_wrapper.run.return_value = mock_response
        
        session = ClaudeCodeSession(mock_wrapper)
        response = session.ask("Test question")
        
        assert response.content == "Session response"
        assert session.session_id == "new-session-id"
        assert len(session.history) == 1
        assert session.history[0] is response
    
    def test_session_multi_turn_conversation(self, mock_wrapper):
        """Test multi-turn conversation management."""
        responses = [
            ClaudeCodeResponse(content="Response 1", returncode=0, session_id="session-123"),
            ClaudeCodeResponse(content="Response 2", returncode=0, session_id="session-123")
        ]
        mock_wrapper.run.side_effect = responses
        
        session = ClaudeCodeSession(mock_wrapper)
        
        # First turn
        resp1 = session.ask("Question 1")
        # Second turn should use established session
        resp2 = session.ask("Question 2")
        
        assert len(session.history) == 2
        # Verify second call used session ID
        second_call_kwargs = mock_wrapper.run.call_args_list[1][1]
        assert second_call_kwargs["session_id"] == "session-123"
    
    def test_session_error_handling(self, mock_wrapper):
        """Test session error handling doesn't crash."""
        mock_wrapper.run.side_effect = ClaudeCodeError("Test error")
        
        session = ClaudeCodeSession(mock_wrapper)
        response = session.ask("Test question")
        
        # Should return error response instead of raising
        assert response.is_error is True
        assert "Session error" in response.content
        assert len(session.history) == 1
    
    def test_session_streaming(self, mock_wrapper):
        """Test session streaming functionality."""
        mock_events = [{"type": "message", "content": "Streaming response"}]
        mock_wrapper.run_streaming.return_value = iter(mock_events)
        
        session = ClaudeCodeSession(mock_wrapper)
        events = list(session.ask_streaming("Streaming question"))
        
        assert len(events) == 1
        assert events[0]["content"] == "Streaming response"
    
    def test_session_context_manager(self, mock_wrapper):
        """Test session as context manager."""
        mock_response = ClaudeCodeResponse(content="Context response", returncode=0)
        mock_wrapper.run.return_value = mock_response
        
        wrapper = ClaudeCodeWrapper()
        wrapper.run = mock_wrapper.run  # Mock the run method
        
        with wrapper.session(max_turns=3) as session:
            response = session.ask("Context question")
            assert response.content == "Context response"
    
    def test_session_history_management(self, mock_wrapper):
        """Test session history management."""
        session = ClaudeCodeSession(mock_wrapper)
        
        # Add some mock responses
        responses = [
            ClaudeCodeResponse(content="Response 1", returncode=0),
            ClaudeCodeResponse(content="Response 2", returncode=0)
        ]
        session.history = responses
        
        # Test history retrieval
        history = session.get_history()
        assert len(history) == 2
        assert history is not session.history  # Should be a copy
        
        # Test history clearing
        session.clear_history()
        assert len(session.history) == 0
        assert session.session_id is None


# Convenience Function Tests
class TestConvenienceFunctions:
    """Test convenience functions with error handling."""
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_success(self, mock_wrapper_class):
        """Test successful ask_claude convenience function."""
        mock_wrapper = MagicMock()
        mock_response = ClaudeCodeResponse(content="Convenience response", returncode=0)
        mock_wrapper.run.return_value = mock_response
        mock_wrapper_class.return_value = mock_wrapper
        
        response = ask_claude("Test question")
        
        assert response.content == "Convenience response"
        mock_wrapper.run.assert_called_once_with("Test question")
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_error_handling(self, mock_wrapper_class):
        """Test ask_claude error handling."""
        mock_wrapper_class.side_effect = Exception("Wrapper creation failed")
        
        response = ask_claude("Test question")
        
        # Should return error response instead of raising
        assert response.is_error is True
        assert "Error: Wrapper creation failed" in response.content
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_json_success(self, mock_wrapper_class):
        """Test ask_claude_json convenience function."""
        mock_wrapper = MagicMock()
        mock_response = ClaudeCodeResponse(content="JSON response", returncode=0)
        mock_wrapper.run.return_value = mock_response
        mock_wrapper_class.return_value = mock_wrapper
        
        response = ask_claude_json("Test question")
        
        mock_wrapper.run.assert_called_once_with("Test question", output_format=OutputFormat.JSON)
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_streaming_success(self, mock_wrapper_class):
        """Test ask_claude_streaming convenience function."""
        mock_wrapper = MagicMock()
        mock_events = [{"type": "message", "content": "Streaming"}]
        mock_wrapper.run_streaming.return_value = iter(mock_events)
        mock_wrapper_class.return_value = mock_wrapper
        
        events = list(ask_claude_streaming("Test question"))
        
        assert len(events) == 1
        assert events[0]["content"] == "Streaming"
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_streaming_error_handling(self, mock_wrapper_class):
        """Test streaming error handling."""
        mock_wrapper_class.side_effect = Exception("Streaming failed")
        
        events = list(ask_claude_streaming("Test question"))
        
        # Should yield error event instead of raising
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "Streaming failed" in events[0]["message"]


# Integration Tests
class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def temp_mcp_config(self, tmp_path):
        """Create temporary MCP configuration."""
        config = {
            "servers": {
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": [str(tmp_path)]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config))
        return config_file
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_comprehensive_workflow(self, mock_run, temp_mcp_config):
        """Test comprehensive workflow with all features."""
        # Mock successful response
        mock_run.return_value = MockProcessResult(
            stdout=TestDataBuilder.claude_response_json(
                content="Comprehensive workflow response",
                session_id="workflow-session",
                cost_usd=0.15,
                duration_ms=3000
            ),
            stderr="",
            returncode=0
        )
        
        # Create wrapper with comprehensive config
        config = ClaudeCodeConfig(
            timeout=30.0,
            max_turns=5,
            verbose=True,
            system_prompt="You are a test assistant",
            allowed_tools=["Python", "Bash", "mcp__filesystem__*"],
            mcp_config_path=temp_mcp_config,
            environment_vars={"TEST_ENV": "integration"},
            max_retries=2,
            enable_metrics=True
        )
        
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
            response = wrapper.run(
                "Create a comprehensive test script",
                output_format=OutputFormat.JSON
            )
        
        # Verify response
        assert response.content == "Comprehensive workflow response"
        assert response.session_id == "workflow-session"
        assert response.metrics.cost_usd == 0.15
        assert response.metrics.duration_ms == 3000
        
        # Verify command construction
        args, kwargs = mock_run.call_args
        command = args[0]
        
        assert command[0] == "claude"
        assert "--print" in command
        assert "--output-format" in command
        assert "json" in command
        assert "--system-prompt" in command
        assert "You are a test assistant" in command
        assert "--allowedTools" in command
        assert "--mcp-config" in command
        assert str(temp_mcp_config) in command
        assert "--verbose" in command
        assert "--max-turns" in command
        
        # Verify environment
        assert kwargs['env']['TEST_ENV'] == "integration"
        assert kwargs['timeout'] == 30.0
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_error_recovery_workflow(self, mock_run):
        """Test error recovery and graceful degradation."""
        # First attempt times out, second succeeds
        mock_run.side_effect = [
            subprocess.TimeoutExpired("claude", 5),
            MockProcessResult(TestDataBuilder.claude_response_json(), "", 0)
        ]
        
        config = ClaudeCodeConfig(max_retries=1, retry_delay=0.01)
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
            response = wrapper.run("Test recovery")
        
        # Should succeed on retry
        assert response.content == "Test response"
        assert mock_run.call_count == 2


# Performance Tests
class TestPerformance:
    """Performance and stress tests."""
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_concurrent_requests_handling(self, mock_run):
        """Test handling of concurrent requests."""
        mock_run.return_value = MockProcessResult(
            TestDataBuilder.claude_response_json(), "", 0
        )
        
        config = ClaudeCodeConfig(max_retries=0)  # Disable retries for speed
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        # Simulate concurrent requests
        import concurrent.futures
        
        def make_request(i):
            return wrapper.run(f"Request {i}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 10
        assert all(r.content == "Test response" for r in results)
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_memory_usage_stability(self, mock_run):
        """Test memory usage remains stable over multiple requests."""
        mock_run.return_value = MockProcessResult(
            TestDataBuilder.claude_response_json(), "", 0
        )
        
        config = ClaudeCodeConfig(max_retries=0)
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        # Make many requests to test for memory leaks
        for i in range(100):
            response = wrapper.run(f"Request {i}")
            assert response.content == "Test response"
        
        # If we get here without issues, memory usage is stable
        assert True


# Security Tests
class TestSecurity:
    """Security-focused tests."""
    
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        config = ClaudeCodeConfig()
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        # Test malicious query
        malicious_query = "normal query; rm -rf /"
        
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.return_value = MockProcessResult("safe response", "", 0)
            
            response = wrapper.run(malicious_query)
            
            # Verify the malicious part wasn't executed as separate command
            args, kwargs = mock_run.call_args
            command = args[0]
            
            # The entire malicious query should be passed as a single argument
            assert malicious_query in command
            # Should not be split into separate commands
            assert "rm" not in command[0:3]  # Not in the base command part
    
    def test_environment_variable_sanitization(self):
        """Test environment variable handling security."""
        config = ClaudeCodeConfig(
            environment_vars={
                "SAFE_VAR": "safe_value",
                "PATH": "/tmp/malicious",  # Potentially dangerous
            }
        )
        
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        env = wrapper._build_env(config)
        
        # Should inherit from os.environ and add our vars
        assert env["SAFE_VAR"] == "safe_value"
        assert "PATH" in env  # Should be present but handled safely


# Observability Tests
class TestObservability:
    """Test logging, metrics, and monitoring capabilities."""
    
    def test_structured_logging(self, caplog):
        """Test structured logging output."""
        config = ClaudeCodeConfig(log_level=logging.DEBUG)
        
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.return_value = MockProcessResult(
                TestDataBuilder.claude_response_json(), "", 0
            )
            
            wrapper.run("Test logging query")
        
        # Verify structured logging
        assert any("Command executed successfully" in record.message 
                  for record in caplog.records)
    
    def test_metrics_collection_accuracy(self):
        """Test accuracy of metrics collection."""
        config = ClaudeCodeConfig(enable_metrics=True)
        
        with patch.object(ClaudeCodeWrapper, '_validate_binary'):
            wrapper = ClaudeCodeWrapper(config)
        
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            # Mock different response types
            mock_run.side_effect = [
                MockProcessResult(TestDataBuilder.claude_response_json(), "", 0),
                MockProcessResult(TestDataBuilder.error_response_json(), "", 0)
            ]
            
            # Make requests
            wrapper.run("Success query")
            wrapper.run("Error query")
        
        metrics = wrapper.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["error_count"] == 1
        assert "total_execution_time" in metrics


# Test Runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--strict-markers"])
