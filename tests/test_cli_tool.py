import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli_tool import ClaudeCLI, create_parser, main
from claude_code_wrapper import ClaudeCodeResponse, ClaudeCodeSession, ClaudeCodeConfig


class TestCLIParser:
    """Test command-line parser"""
    
    def test_create_parser(self):
        """Test parser creation"""
        parser = create_parser()
        
        # Test parser has all subcommands
        args = parser.parse_args(['ask', 'Hello'])
        assert args.command == 'ask'
        assert args.prompt == 'Hello'
        
        args = parser.parse_args(['stream', 'Test'])
        assert args.command == 'stream'
        
        args = parser.parse_args(['session', '--interactive'])
        assert args.command == 'session'
        assert args.interactive == True
        
        args = parser.parse_args(['health'])
        assert args.command == 'health'
        
        args = parser.parse_args(['benchmark'])
        assert args.command == 'benchmark'
    
    def test_ask_command_args(self):
        """Test ask command arguments"""
        parser = create_parser()
        
        args = parser.parse_args([
            'ask', 'Test prompt',
            '--json',
            '--model', 'claude-3',
            '--temperature', '0.5',
            '--timeout', '60',
            '--config', 'config.json',
            '--show-metadata',
            '--no-cache'
        ])
        
        assert args.prompt == 'Test prompt'
        assert args.json == True
        assert args.model == 'claude-3'
        assert args.temperature == 0.5
        assert args.timeout == 60
        assert args.config == 'config.json'
        assert args.show_metadata == True
        assert args.no_cache == True
    
    def test_session_command_args(self):
        """Test session command arguments"""
        parser = create_parser()
        
        args = parser.parse_args([
            'session',
            '--session-id', 'test-session',
            '--prompt', 'Hello',
            '--list',
            '--clear', 'old-session'
        ])
        
        assert args.session_id == 'test-session'
        assert args.prompt == 'Hello'
        assert args.list == True
        assert args.clear == 'old-session'


class TestClaudeCLI:
    """Test ClaudeCLI class methods"""
    
    @pytest.fixture
    def cli(self):
        """Create CLI instance"""
        return ClaudeCLI()
    
    def test_cli_initialization(self, cli):
        """Test CLI initialization"""
        assert cli.wrapper is None
        assert isinstance(cli.config, ClaudeCodeConfig)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_format_output_text(self, mock_stdout, cli):
        """Test text output formatting"""
        response = ClaudeCodeResponse(
            content="Test response",
            exit_code=0,
            duration=1.5,
            retries=0
        )
        
        cli._format_output(response, as_json=False)
        output = mock_stdout.getvalue()
        assert output.strip() == "Test response"
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_format_output_json(self, mock_stdout, cli):
        """Test JSON output formatting"""
        response = ClaudeCodeResponse(
            content="Test response",
            exit_code=0,
            duration=1.5,
            retries=1,
            metadata={"key": "value"}
        )
        
        cli._format_output(response, as_json=True)
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert data["content"] == "Test response"
        assert data["exit_code"] == 0
        assert data["duration"] == 1.5
        assert data["retries"] == 1
        assert data["metadata"] == {"key": "value"}
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_display_metadata(self, mock_stdout, cli):
        """Test metadata display"""
        response = ClaudeCodeResponse(
            content="Test",
            exit_code=0,
            duration=2.5,
            retries=1
        )
        
        cli._display_metadata(response)
        output = mock_stdout.getvalue()
        
        assert "Metadata" in output
        assert "Duration: 2.50s" in output
        assert "Retries: 1" in output
        assert "Exit Code: 0" in output
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_ask(self, mock_wrapper_class, cli):
        """Test ask command handler"""
        # Setup mock
        mock_wrapper = Mock()
        mock_response = ClaudeCodeResponse(
            content="Test response",
            exit_code=0,
            duration=1.0,
            retries=0
        )
        mock_wrapper.ask.return_value = mock_response
        mock_wrapper_class.return_value = mock_wrapper
        
        # Create args
        args = Mock(
            prompt="Test prompt",
            json=False,
            model=None,
            temperature=None,
            max_tokens=None,
            timeout=None,
            config=None,
            show_metadata=False,
            no_cache=False
        )
        
        with patch('sys.stdout', new_callable=StringIO):
            # Test execution
            result = cli.handle_ask(args)
            assert result == 0
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_ask_json(self, mock_wrapper_class, cli):
        """Test ask command with JSON output"""
        mock_wrapper = Mock()
        mock_wrapper.ask_json.return_value = {"result": "success"}
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(
            prompt="Return JSON",
            json=True,
            model=None,
            temperature=None,
            max_tokens=None,
            timeout=None,
            config=None,
            show_metadata=False,
            no_cache=False
        )
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.handle_ask(args)
            assert result == 0
            # Check JSON was printed
            printed_json = mock_stdout.getvalue()
            assert json.loads(printed_json) == {"result": "success"}
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_stream(self, mock_wrapper_class, cli):
        """Test stream command handler"""
        mock_wrapper = Mock()
        mock_wrapper.stream.return_value = iter(["chunk1", "chunk2", "chunk3"])
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(
            prompt="Stream test",
            model=None,
            temperature=None,
            timeout=None,
            config=None
        )
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.handle_stream(args)
            assert result == 0
            # Check chunks were printed
            output = mock_stdout.getvalue()
            assert "chunk1" in output
            assert "chunk2" in output
            assert "chunk3" in output
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_session_interactive(self, mock_wrapper_class, cli):
        """Test interactive session"""
        mock_wrapper = Mock()
        mock_session = Mock()
        mock_wrapper.create_session.return_value = mock_session
        mock_wrapper.ask_in_session.return_value = ClaudeCodeResponse(
            content="Response",
            exit_code=0,
            duration=1.0,
            retries=0
        )
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(
            interactive=True,
            session_id=None,
            prompt=None,
            list=False,
            clear=None,
            timeout=None,
            config=None
        )
        
        # Mock input to simulate user interaction
        with patch('builtins.input', side_effect=['Hello', 'exit']):
            with patch('builtins.print'):
                result = cli.handle_session(args)
                assert result == 0
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_session_list(self, mock_wrapper_class, cli):
        """Test listing sessions"""
        mock_wrapper = Mock()
        mock_session1 = Mock(spec=ClaudeCodeSession)
        mock_session1.session_id = "session1"
        mock_session1.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        mock_session1.messages = [1, 2, 3]
        
        mock_wrapper.get_sessions.return_value = {"session1": mock_session1}
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(
            list=True,
            interactive=False,
            session_id=None,
            prompt=None,
            clear=None,
            timeout=None,
            config=None
        )
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.handle_session(args)
            assert result == 0
            output = mock_stdout.getvalue()
            assert "session1" in output
            assert "3 messages" in output
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    def test_handle_health(self, mock_wrapper_class, cli):
        """Test health check command"""
        mock_wrapper = Mock()
        mock_wrapper.health_check.return_value = {
            "status": "healthy",
            "claude_available": True,
            "version": "1.0.0"
        }
        mock_wrapper.get_metrics.return_value = {
            "total_requests": 10,
            "successful_requests": 9,
            "failed_requests": 1,
            "success_rate": 0.9,
            "average_retries_per_request": 0.5,
            "cache_hit_rate": 0.2
        }
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(timeout=None, config=None)
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.handle_health(args)
            assert result == 0
            output = mock_stdout.getvalue()
            assert "healthy" in output
            assert "Total Requests: 10" in output
    
    @patch('claude_code_wrapper.ClaudeCodeWrapper')
    @patch('time.time')
    def test_handle_benchmark(self, mock_time, mock_wrapper_class, cli):
        """Test benchmark command"""
        # Mock time for consistent results
        mock_time.side_effect = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        mock_wrapper = Mock()
        mock_wrapper.ask.return_value = ClaudeCodeResponse(
            content="Response",
            exit_code=0,
            duration=0.1,
            retries=0
        )
        mock_wrapper_class.return_value = mock_wrapper
        
        args = Mock(
            iterations=5,
            prompt="Test",
            timeout=None,
            config=None
        )
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = cli.handle_benchmark(args)
            assert result == 0
            assert mock_wrapper.ask.call_count == 5
            output = mock_stdout.getvalue()
            assert "Average response time" in output


class TestMainFunction:
    """Test main entry point"""
    
    @patch('sys.argv', ['cli_tool.py', 'ask', 'Hello'])
    @patch('cli_tool.ClaudeCLI')
    def test_main_ask(self, mock_cli_class):
        """Test main with ask command"""
        mock_cli = Mock()
        mock_cli.handle_ask.return_value = 0
        mock_cli_class.return_value = mock_cli
        
        result = main()
        assert result == 0
        assert mock_cli.handle_ask.called
    
    @patch('sys.argv', ['cli_tool.py', 'stream', 'Hello'])
    @patch('cli_tool.ClaudeCLI')
    def test_main_stream(self, mock_cli_class):
        """Test main with stream command"""
        mock_cli = Mock()
        mock_cli.handle_stream.return_value = 0
        mock_cli_class.return_value = mock_cli
        
        result = main()
        assert result == 0
        assert mock_cli.handle_stream.called
    
    @patch('sys.argv', ['cli_tool.py', 'invalid'])
    def test_main_invalid_command(self):
        """Test main with invalid command"""
        with patch('sys.stderr', new_callable=StringIO):
            with pytest.raises(SystemExit):
                main()
    
    @patch('sys.argv', ['cli_tool.py', 'ask', 'Test'])
    @patch('cli_tool.ClaudeCLI')
    def test_main_error_handling(self, mock_cli_class):
        """Test main error handling"""
        mock_cli = Mock()
        mock_cli.handle_ask.side_effect = Exception("Test error")
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()
            assert result == 1
            assert "Test error" in mock_stderr.getvalue()


class TestCLIIntegration:
    """CLI integration tests"""
    
    def test_config_file_loading(self):
        """Test loading configuration from file"""
        config_data = {
            "timeout": 120,
            "max_retries": 5,
            "enable_logging": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            args = Mock(
                config=config_file,
                timeout=None,
                prompt="Test",
                json=False,
                model=None,
                temperature=None,
                max_tokens=None,
                show_metadata=False,
                no_cache=False
            )
            
            cli = ClaudeCLI()
            
            with patch('claude_code_wrapper.ClaudeCodeWrapper') as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper.ask.return_value = ClaudeCodeResponse(
                    content="Response",
                    exit_code=0,
                    duration=1.0,
                    retries=0
                )
                mock_wrapper_class.return_value = mock_wrapper
                
                with patch('sys.stdout', new_callable=StringIO):
                    cli.handle_ask(args)
                
                # Check config was loaded
                config_call = mock_wrapper_class.call_args[0][0]
                assert config_call.timeout == 120
                assert config_call.max_retries == 5
        finally:
            os.unlink(config_file)
    
    def test_error_output_formatting(self):
        """Test error output formatting"""
        args = Mock(
            prompt="Test",
            json=False,
            model=None,
            temperature=None,
            max_tokens=None,
            timeout=None,
            config=None,
            show_metadata=False,
            no_cache=False
        )
        
        cli = ClaudeCLI()
        
        with patch('claude_code_wrapper.ClaudeCodeWrapper') as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.ask.side_effect = Exception("API Error")
            mock_wrapper_class.return_value = mock_wrapper
            
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                result = cli.handle_ask(args)
                assert result == 1
                assert "API Error" in mock_stderr.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])