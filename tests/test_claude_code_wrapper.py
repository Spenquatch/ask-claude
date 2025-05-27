import pytest
import json
import tempfile
import os
import time
import subprocess
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import (
    ClaudeCodeWrapper, ClaudeCodeConfig, ClaudeCodeResponse, ClaudeCodeSession,
    ClaudeCodeError, ClaudeCodeTimeoutError, ClaudeCodeProcessError,
    ClaudeCodeValidationError, ClaudeCodeConfigurationError,
    ask_claude, ask_claude_json, ask_claude_streaming,
    OutputFormat, ErrorSeverity, ClaudeCodeMetrics
)


class TestClaudeCodeConfig:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ClaudeCodeConfig()
        assert config.claude_binary == "claude"
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.verbose == False
        assert config.enable_metrics == True
        assert config.log_level == 20  # logging.INFO
        assert config.environment_vars == {}
        assert config.working_directory is None
        assert config.session_id is None
        assert config.continue_session == False
    
    def test_config_from_dict(self):
        """Test configuration from dictionary"""
        config_dict = {
            "timeout": 30,
            "max_retries": 5,
            "verbose": True,
            "session_id": "test-123"
        }
        config = ClaudeCodeConfig.from_dict(config_dict)
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.verbose == True
        assert config.session_id == "test-123"
    
    def test_config_from_json_file(self):
        """Test configuration from JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"timeout": 120, "log_level": "DEBUG"}, f)
            temp_file = f.name
        
        try:
            config = ClaudeCodeConfig.from_json_file(temp_file)
            assert config.timeout == 120
            assert config.log_level == "DEBUG"
        finally:
            os.unlink(temp_file)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = ClaudeCodeConfig()
        
        # Test valid config
        config.validate()  # Should not raise
        
        # Test invalid timeout
        config.timeout = -1
        with pytest.raises(ClaudeCodeConfigurationError):
            config.validate()
        
        # Test invalid max_retries
        config.timeout = 60
        config.max_retries = -1
        with pytest.raises(ClaudeCodeConfigurationError):
            config.validate()
        
        # Test creating config with invalid values directly
        with pytest.raises(ClaudeCodeConfigurationError):
            ClaudeCodeConfig(timeout=-1)
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = ClaudeCodeConfig(timeout=90, enable_metrics=False)
        config_dict = config.to_dict()
        assert config_dict["timeout"] == 90
        assert config_dict["enable_metrics"] == False
        assert "claude_binary" in config_dict


class TestClaudeCodeResponse:
    """Test response handling"""
    
    def test_response_creation(self):
        """Test response object creation"""
        response = ClaudeCodeResponse(
            content="Test response",
            returncode=0,
            execution_time=1.5,
            retries=0,
            raw_output="Raw output"
        )
        assert response.content == "Test response"
        assert response.returncode == 0
        assert response.execution_time == 1.5
        assert response.retries == 0
        assert response.raw_output == "Raw output"
        assert response.error_type is None
        assert isinstance(response.timestamp, float)
        assert response.metadata == {}
    
    def test_response_with_error(self):
        """Test response with error"""
        response = ClaudeCodeResponse(
            content="",
            returncode=1,
            execution_time=0.5,
            retries=2,
            is_error=True,
            error_type="command_failed"
        )
        assert response.is_error == True
        assert response.returncode == 1
        assert response.retries == 2
    
    def test_response_success_property(self):
        """Test success property"""
        # Successful response
        response = ClaudeCodeResponse(content="Success", returncode=0, execution_time=1.0, retries=0)
        assert response.success == True
        
        # Failed response
        response = ClaudeCodeResponse(content="", returncode=1, execution_time=1.0, retries=0)
        assert response.success == False
    
    def test_response_to_dict(self):
        """Test response serialization"""
        response = ClaudeCodeResponse(
            content="Test",
            returncode=0,
            execution_time=2.0,
            retries=1,
            metadata={"key": "value"}
        )
        response_dict = response.to_dict()
        assert response_dict["content"] == "Test"
        assert response_dict["returncode"] == 0
        assert response_dict["execution_time"] == 2.0
        assert response_dict["retries"] == 1
        assert response_dict["metadata"] == {"key": "value"}
        assert "timestamp" in response_dict


class TestClaudeCodeSession:
    """Test session management"""
    
    def test_session_creation(self):
        """Test session creation"""
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper, session_id="test-session")
        assert session.session_id == "test-session"
        assert session.messages == []
        assert session.total_duration == 0
        assert session.total_retries == 0
        assert isinstance(session.created_at, float)
        assert session.metadata == {}
    
    def test_add_message(self):
        """Test adding messages to session"""
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper)
        
        # Add user message
        session.add_message("user", "Hello")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"
        
        # Add assistant message with metadata
        session.add_message("assistant", "Hi there", metadata={"model": "claude"})
        assert len(session.messages) == 2
        assert session.messages[1]["metadata"] == {"model": "claude"}
    
    def test_update_metrics(self):
        """Test updating session metrics"""
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper)
        
        session.update_metrics(duration=1.5, retries=1)
        assert session.total_duration == 1.5
        assert session.total_retries == 1
        
        session.update_metrics(duration=2.0, retries=2)
        assert session.total_duration == 3.5
        assert session.total_retries == 3
    
    def test_get_context(self):
        """Test getting session context"""
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper)
        session.add_message("user", "Question 1")
        session.add_message("assistant", "Answer 1")
        session.add_message("user", "Question 2")
        
        # Get last 2 messages
        context = session.get_context(max_messages=2)
        assert len(context) == 2
        assert context[0]["content"] == "Answer 1"
        assert context[1]["content"] == "Question 2"
    
    def test_session_to_dict(self):
        """Test session serialization"""
        wrapper = ClaudeCodeWrapper()
        session = ClaudeCodeSession(wrapper, session_id="test")
        session.metadata = {"project": "test"}
        session.add_message("user", "Hello")
        session.update_metrics(1.0, 0)
        
        session_dict = session.to_dict()
        assert session_dict["session_id"] == "test"
        assert len(session_dict["messages"]) == 1
        assert session_dict["total_duration"] == 1.0
        assert session_dict["metadata"] == {"project": "test"}


class TestClaudeCodeWrapper:
    """Test main wrapper functionality"""
    
    @pytest.fixture
    def wrapper(self):
        """Create wrapper instance with mocked subprocess"""
        config = ClaudeCodeConfig(timeout=10, max_retries=2)
        return ClaudeCodeWrapper(config)
    
    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess module"""
        with patch('claude_code_wrapper.subprocess') as mock:
            yield mock
    
    def test_wrapper_initialization(self, wrapper):
        """Test wrapper initialization"""
        assert wrapper.config.timeout == 10
        assert wrapper.config.max_retries == 2
        assert wrapper._cache == {}
        assert wrapper._metrics == {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def test_validate_prompt(self, wrapper):
        """Test prompt validation"""
        # Valid prompt
        wrapper._validate_prompt("Valid prompt")  # Should not raise
        
        # Empty prompt
        with pytest.raises(ClaudeCodeValidationError):
            wrapper._validate_prompt("")
        
        # None prompt
        with pytest.raises(ClaudeCodeValidationError):
            wrapper._validate_prompt(None)
        
        # Too long prompt
        long_prompt = "x" * 100001
        with pytest.raises(ClaudeCodeValidationError):
            wrapper._validate_prompt(long_prompt)
    
    def test_build_command(self, wrapper):
        """Test command building"""
        from claude_code_wrapper import OutputFormat
        
        # Basic command
        cmd = wrapper._build_command("Hello", OutputFormat.TEXT, wrapper.config)
        assert cmd == ["claude", "--print", "Hello"]
        
        # With session ID
        config_with_session = ClaudeCodeConfig(session_id="test-123")
        cmd = wrapper._build_command("Hello", OutputFormat.TEXT, config_with_session)
        assert "--resume" in cmd
        assert "test-123" in cmd
        
        # With JSON output
        cmd = wrapper._build_command("Hello", OutputFormat.JSON, wrapper.config)
        assert "--output-format" in cmd
        assert "json" in cmd
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_run_success(self, mock_run, wrapper):
        """Test successful run execution"""
        mock_run.return_value = Mock(
            stdout='{"result": "Success response", "cost_usd": 0.01}',
            stderr="",
            returncode=0
        )
        
        response = wrapper.run("Hello")
        # The response is returned as-is in text mode
        assert response.content == '{"result": "Success response", "cost_usd": 0.01}'
        assert response.returncode == 0
    
    def test_timeout_handling(self, wrapper):
        """Test timeout handling"""
        # Set very short timeout
        wrapper.config.timeout = 0.001
        
        with patch('claude_code_wrapper.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(["claude"], 0.001)
            
            with pytest.raises(ClaudeCodeTimeoutError):
                wrapper.run("Hello")
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_retry_mechanism(self, mock_run, wrapper):
        """Test retry mechanism"""
        # First call fails, second succeeds
        mock_run.side_effect = [
            subprocess.TimeoutExpired(["claude"], 1),
            Mock(stdout='{"result": "Success"}', stderr="", returncode=0)
        ]
        
        response = wrapper.run("Hello")
        assert "Success" in response.content
        # Retries happen internally
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_all_retries_fail(self, mock_run, wrapper):
        """Test when all retries fail"""
        # All calls fail
        mock_run.side_effect = subprocess.TimeoutExpired(["claude"], 1)
        
        with pytest.raises(ClaudeCodeTimeoutError):
            wrapper.run("Hello")
        
        # Should retry max_retries + 1 times (initial + retries)
        assert mock_run.call_count == wrapper.config.max_retries + 1
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_ask_success(self, mock_run, wrapper):
        """Test successful ask operation"""
        mock_run.return_value = Mock(
            stdout="Test response",
            stderr="",
            returncode=0
        )
        
        response = wrapper.ask("What is 2+2?")
        assert response.content == "Test response"
        assert response.exit_code == 0
        assert response.success == True
        assert wrapper._metrics["total_requests"] == 1
        assert wrapper._metrics["successful_requests"] == 1
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_ask_with_options(self, mock_run, wrapper):
        """Test ask with additional options"""
        mock_run.return_value = Mock(stdout="Response", stderr="", returncode=0)
        
        response = wrapper.ask(
            "Test prompt",
            model="claude-3",
            temperature=0.7,
            verbose=True
        )
        
        # Check command includes options
        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        assert "claude-3" in call_args
        assert "--temperature" in call_args
        assert "0.7" in call_args
        assert "--verbose" in call_args
    
    def test_ask_json(self, wrapper):
        """Test JSON response parsing"""
        with patch.object(wrapper, 'ask') as mock_ask:
            mock_ask.return_value = ClaudeCodeResponse(
                content='{"result": "success", "value": 42}',
                returncode=0,
                execution_time=1.0
            )
            
            result = wrapper.ask_json("Return JSON")
            assert result == {"result": "success", "value": 42}
    
    def test_ask_json_invalid(self, wrapper):
        """Test invalid JSON response"""
        with patch.object(wrapper, 'ask') as mock_ask:
            mock_ask.return_value = ClaudeCodeResponse(
                content='Not valid JSON',
                returncode=0,
                execution_time=1.0
            )
            
            with pytest.raises(ClaudeCodeError):
                wrapper.ask_json("Return JSON")
    
    @patch('claude_code_wrapper.subprocess.Popen')
    def test_stream_success(self, mock_popen, wrapper):
        """Test streaming response"""
        # Mock process with line-by-line output
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]  # Process running, then done
        mock_process.stdout = iter(['{"type": "content", "content": "Line 1"}\n', '{"type": "content", "content": "Line 2"}\n'])
        mock_process.stderr.read.return_value = ""
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        chunks = list(wrapper.stream("Stream test"))
        assert len(chunks) == 2
        assert chunks[0] == "Line 1"
        assert chunks[1] == "Line 2"
    
    def test_create_session(self, wrapper):
        """Test session creation"""
        session = wrapper.create_session("test-session")
        assert session.session_id == "test-session"
        assert "test-session" in wrapper._sessions
    
    def test_ask_in_session(self, wrapper):
        """Test asking within a session"""
        session = wrapper.create_session("test")
        
        with patch.object(wrapper, 'run') as mock_run:
            mock_run.return_value = ClaudeCodeResponse(
                content="Session response",
                returncode=0,
                execution_time=1.0
            )
            
            response = session.ask("Hello")
            assert response.content == "Session response"
            assert len(session.messages) == 2  # User + assistant
            assert session.messages[0]["content"] == "Hello"
            assert session.messages[1]["content"] == "Session response"
    
    def test_cache_functionality(self, wrapper):
        """Test response caching"""
        wrapper.config.cache_responses = True
        wrapper.config.cache_ttl = 60
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                stdout="Cached response",
                stderr="",
                returncode=0
            )
            
            # First call - cache miss
            response1 = wrapper.ask("Test prompt")
            assert wrapper._metrics["cache_misses"] == 1
            
            # Second call - cache hit
            response2 = wrapper.ask("Test prompt")
            assert wrapper._metrics["cache_hits"] == 1
            assert response1.content == response2.content
            assert mock_run.call_count == 1  # Only called once
    
    def test_get_metrics(self, wrapper):
        """Test metrics retrieval"""
        metrics = wrapper.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 0
        assert "success_rate" in metrics
        assert "average_retries_per_request" in metrics
        assert "cache_hit_rate" in metrics
    
    def test_clear_cache(self, wrapper):
        """Test cache clearing"""
        wrapper._cache = {"key": ("value", time.time())}
        wrapper.clear_cache()
        assert wrapper._cache == {}
    
    def test_close(self, wrapper):
        """Test wrapper cleanup"""
        wrapper._sessions = {"test": ClaudeCodeSession(wrapper, session_id="test")}
        wrapper._cache = {"key": "value"}
        
        wrapper.close()
        assert wrapper._sessions == {}
        assert wrapper._cache == {}


class TestConvenienceFunctions:
    """Test module-level convenience functions"""
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude(self, mock_wrapper_class):
        """Test ask_claude convenience function"""
        mock_instance = Mock()
        mock_instance.run.return_value = ClaudeCodeResponse(
            content="Response",
            returncode=0,
            execution_time=1.0
        )
        mock_wrapper_class.return_value = mock_instance
        
        response = ask_claude("Test prompt", timeout=30)
        assert response.content == "Response"
        mock_wrapper_class.assert_called_once()
        mock_instance.run.assert_called_with("Test prompt", timeout=30)
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_json(self, mock_wrapper_class):
        """Test ask_claude_json convenience function"""
        mock_instance = Mock()
        mock_instance.run.return_value = ClaudeCodeResponse(
            content='{"result": "success"}',
            returncode=0,
            execution_time=1.0
        )
        mock_wrapper_class.return_value = mock_instance
        
        response = ask_claude_json("Return JSON")
        assert response.content == '{"result": "success"}'
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_ask_claude_streaming(self, mock_wrapper_class):
        """Test ask_claude_streaming convenience function"""
        mock_instance = Mock()
        mock_instance.run_streaming.return_value = iter([
            {"type": "content", "content": "chunk1"},
            {"type": "content", "content": "chunk2"}
        ])
        mock_wrapper_class.return_value = mock_instance
        
        events = list(ask_claude_streaming("Stream test"))
        assert len(events) == 2
        assert events[0] == {"type": "content", "content": "chunk1"}
        assert events[1] == {"type": "content", "content": "chunk2"}


class TestErrorHandling:
    """Test error handling and exceptions"""
    
    def test_exception_hierarchy(self):
        """Test exception class hierarchy"""
        # Base error
        error = ClaudeCodeError("Base error")
        assert str(error) == "Base error"
        assert isinstance(error, Exception)
        
        # Specific errors
        assert issubclass(ClaudeCodeProcessError, ClaudeCodeError)
        assert issubclass(ClaudeCodeConfigurationError, ClaudeCodeError)
        assert issubclass(ClaudeCodeTimeoutError, ClaudeCodeError)
        assert issubclass(ClaudeCodeValidationError, ClaudeCodeError)
    
    def test_process_error_with_details(self):
        """Test process error with additional details"""
        error = ClaudeCodeProcessError("Process failed", returncode=1, stderr="Error output")
        assert str(error) == "Process failed"
        assert error.returncode == 1
        assert error.stderr == "Error output"


class TestIntegration:
    """Integration tests with minimal API usage"""
    
    @pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), 
                        reason="Integration tests disabled by default")
    def test_real_api_call(self):
        """Test real API call with minimal usage"""
        wrapper = ClaudeCodeWrapper()
        try:
            response = wrapper.ask("What is 2+2? Reply with just the number.")
            assert response.success
            assert "4" in response.content
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
    
    @pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"),
                        reason="Integration tests disabled by default")
    def test_real_streaming(self):
        """Test real streaming with minimal output"""
        wrapper = ClaudeCodeWrapper()
        try:
            chunks = list(wrapper.stream("Say 'hello' and nothing else."))
            assert len(chunks) > 0
            full_response = "".join(chunks).lower()
            assert "hello" in full_response
        except Exception as e:
            pytest.skip(f"Streaming failed: {e}")


class TestMCPAutoApproval:
    """Test MCP auto-approval functionality"""
    
    def test_auto_approval_config_parsing(self):
        """Test parsing of auto-approval configuration"""
        config = ClaudeCodeConfig(
            mcp_auto_approval={
                "enabled": True,
                "strategy": "allowlist",
                "allowlist": ["tool1", "tool2"]
            }
        )
        assert config.mcp_auto_approval["enabled"] == True
        assert config.mcp_auto_approval["strategy"] == "allowlist"
        assert config.mcp_auto_approval["allowlist"] == ["tool1", "tool2"]
    
    def test_auto_approval_disabled_by_default(self):
        """Test that auto-approval is disabled by default"""
        config = ClaudeCodeConfig()
        assert config.mcp_auto_approval == {}
    
    def test_setup_approval_server_creates_config(self):
        """Test that _setup_approval_server creates proper config"""
        # Create test MCP config file
        test_mcp_config = {"mcpServers": {"test": {"command": "test"}}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_mcp_config, f)
            test_config_path = f.name
        
        try:
            # Create config with auto-approval
            config = ClaudeCodeConfig(
                mcp_config_path=Path(test_config_path),
                mcp_auto_approval={
                    "enabled": True,
                    "strategy": "all"
                }
            )
            
            # Create wrapper instance (don't initialize to avoid subprocess calls)
            import logging
            wrapper = ClaudeCodeWrapper.__new__(ClaudeCodeWrapper)
            wrapper.logger = logging.getLogger(__name__)
            
            # Call internal method
            temp_config_path = wrapper._setup_approval_server(config)
            
            # Verify temp config was created
            assert temp_config_path is not None
            assert os.path.exists(temp_config_path)
            
            # Read and verify the temp config
            with open(temp_config_path, 'r') as tf:
                config_data = json.load(tf)
            
            assert "mcpServers" in config_data
            assert "approval-server" in config_data["mcpServers"]
            assert "test" in config_data["mcpServers"]  # Original server preserved
            
            # Check approval server configuration
            approval_server = config_data["mcpServers"]["approval-server"]
            assert "env" in approval_server
            assert "APPROVAL_STRATEGY_CONFIG" in approval_server["env"]
            
            # Verify strategy config
            strategy_config = json.loads(approval_server["env"]["APPROVAL_STRATEGY_CONFIG"])
            assert strategy_config["type"] == "all"
            
            # Cleanup temp file
            os.unlink(temp_config_path)
            
        finally:
            # Cleanup test file
            os.unlink(test_config_path)
    
    def test_approval_strategy_with_allowlist(self):
        """Test that approval strategy is correctly configured with allowlist"""
        config = ClaudeCodeConfig(
            mcp_auto_approval={
                "enabled": True,
                "strategy": "allowlist",
                "allowlist": ["mcp__test__tool1", "mcp__test__tool2"]
            }
        )
        
        # Create wrapper instance without initialization
        import logging
        wrapper = ClaudeCodeWrapper.__new__(ClaudeCodeWrapper)
        wrapper.logger = logging.getLogger(__name__)
        
        # Test _setup_approval_server
        temp_config = wrapper._setup_approval_server(config)
        
        if temp_config:
            # Read the temp config
            with open(temp_config, 'r') as f:
                mcp_config = json.load(f)
            
            # Verify approval server is configured
            assert "approval-server" in mcp_config["mcpServers"]
            approval_env = mcp_config["mcpServers"]["approval-server"]["env"]
            
            # Check strategy config
            strategy_config = json.loads(approval_env["APPROVAL_STRATEGY_CONFIG"])
            assert strategy_config["type"] == "allowlist"
            assert strategy_config["allowlist"] == ["mcp__test__tool1", "mcp__test__tool2"]
            
            # Cleanup
            os.unlink(temp_config)
    
    def test_all_approval_strategies(self):
        """Test all approval strategy types are valid"""
        strategies = ["all", "none", "allowlist", "patterns"]
        
        for strategy in strategies:
            config = ClaudeCodeConfig(
                mcp_auto_approval={
                    "enabled": True,
                    "strategy": strategy
                }
            )
            assert config.mcp_auto_approval["strategy"] == strategy
    
    def test_approval_with_patterns(self):
        """Test pattern-based approval configuration"""
        config = ClaudeCodeConfig(
            mcp_auto_approval={
                "enabled": True,
                "strategy": "patterns",
                "allow_patterns": ["mcp__.*__read.*", "mcp__.*__list.*"],
                "deny_patterns": ["mcp__.*__admin.*", "mcp__.*__delete.*"]
            }
        )
        
        # Create wrapper instance without initialization
        import logging
        wrapper = ClaudeCodeWrapper.__new__(ClaudeCodeWrapper)
        wrapper.logger = logging.getLogger(__name__)
        
        # Test _setup_approval_server
        temp_config = wrapper._setup_approval_server(config)
        
        if temp_config:
            # Read the temp config
            with open(temp_config, 'r') as f:
                mcp_config = json.load(f)
            
            # Verify approval server is configured
            assert "approval-server" in mcp_config["mcpServers"]
            approval_env = mcp_config["mcpServers"]["approval-server"]["env"]
            
            # Check strategy config
            strategy_config = json.loads(approval_env["APPROVAL_STRATEGY_CONFIG"])
            assert strategy_config["type"] == "patterns"
            assert strategy_config["allow_patterns"] == ["mcp__.*__read.*", "mcp__.*__list.*"]
            assert strategy_config["deny_patterns"] == ["mcp__.*__admin.*", "mcp__.*__delete.*"]
            
            # Cleanup
            os.unlink(temp_config)
    
    def test_mcp_config_with_existing_servers(self):
        """Test that existing MCP servers are preserved when adding approval"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            original_config = {
                "mcpServers": {
                    "filesystem": {
                        "command": "mcp-filesystem",
                        "args": ["/path"]
                    }
                }
            }
            json.dump(original_config, f)
            f.flush()
            
            try:
                config = ClaudeCodeConfig(
                    mcp_config_path=Path(f.name),
                    mcp_auto_approval={
                        "enabled": True,
                        "strategy": "all"
                    }
                )
                wrapper = ClaudeCodeWrapper(config)
                
                # Create temp config
                temp_config = wrapper._setup_approval_server(config)
                
                # Read the temp config
                with open(temp_config, 'r') as tf:
                    combined_config = json.load(tf)
                
                # Verify both servers exist
                assert "filesystem" in combined_config["mcpServers"]
                assert "approval-server" in combined_config["mcpServers"]
                
                # Cleanup temp file
                os.unlink(temp_config)
                
            finally:
                os.unlink(f.name)
    
    def test_approval_disabled_no_server_added(self):
        """Test that approval server is not added when disabled"""
        # Config with disabled auto-approval
        config = ClaudeCodeConfig(
            mcp_auto_approval={
                "enabled": False,
                "strategy": "all"
            }
        )
        
        # Create wrapper instance without initialization
        import logging
        wrapper = ClaudeCodeWrapper.__new__(ClaudeCodeWrapper)
        wrapper.logger = logging.getLogger(__name__)
        
        # Test _setup_approval_server
        temp_config = wrapper._setup_approval_server(config)
        
        # Should return None when disabled
        assert temp_config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])