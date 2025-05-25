"""
Test runner and configuration for Claude Code SDK Wrapper

This file provides test configuration, fixtures, and utilities for running
the test suite with different configurations and environments.
"""

import os
import sys
import json
import pytest
import logging
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig


# Test configuration
class TestConfig:
    """Configuration for test runs."""
    
    # Test data directory
    TEST_DATA_DIR = Path(__file__).parent / "test_data"
    
    # Sample MCP configurations
    SAMPLE_MCP_CONFIGS = {
        "filesystem": {
            "servers": {
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": ["/tmp"]
                }
            }
        },
        "database": {
            "servers": {
                "database": {
                    "command": "mcp-server-sqlite",
                    "args": ["./test.db"]
                }
            }
        },
        "web": {
            "servers": {
                "web": {
                    "command": "mcp-server-web",
                    "args": ["--port", "8080"]
                }
            }
        }
    }
    
    # Sample responses for different scenarios
    SAMPLE_RESPONSES = {
        "simple_text": "Hello! I'm Claude, an AI assistant created by Anthropic.",
        
        "code_generation": '''Here's a Python function to process files:

```python
def process_files(directory: str) -> List[str]:
    """Process all files in a directory."""
    processed = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r') as f:
            content = f.read()
            # Process content here
            processed.append(file_path.name)
    return processed
```

This function iterates through all .txt files in the specified directory and processes them.''',
        
        "json_response": {
            "content": "I've analyzed your request and here's my response.",
            "session_id": "session_12345",
            "metadata": {
                "model": "claude-4",
                "tokens_used": 150,
                "processing_time": 2.3,
                "tools_used": ["Python", "FileSystem"]
            }
        },
        
        "streaming_response": [
            {"type": "init", "session_id": "stream_12345", "timestamp": "2025-01-01T12:00:00Z"},
            {"type": "message", "role": "assistant", "content": "I'll help you with that. Let me start by"},
            {"type": "message", "role": "assistant", "content": " analyzing the problem..."},
            {"type": "tool_use", "tool": "Python", "action": "execute", "code": "print('Hello World')"},
            {"type": "tool_result", "tool": "Python", "result": "Hello World"},
            {"type": "message", "role": "assistant", "content": " The solution is ready!"},
            {"type": "result", "status": "complete", "stats": {"total_tokens": 125, "execution_time": 3.1}}
        ],
        
        "error_response": {
            "error": "Tool execution failed",
            "details": "Permission denied when accessing file system",
            "code": "PERMISSION_DENIED"
        }
    }


# Pytest fixtures
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TestConfig()


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "claude_test_workspace"
    workspace.mkdir()
    
    # Create sample files
    (workspace / "input.txt").write_text("Sample input content")
    (workspace / "config.json").write_text('{"test": true}')
    (workspace / "scripts").mkdir()
    (workspace / "scripts" / "test.py").write_text("print('test script')")
    
    return workspace


@pytest.fixture
def sample_mcp_config(tmp_path, test_config):
    """Create sample MCP configuration files."""
    configs = {}
    for name, config in test_config.SAMPLE_MCP_CONFIGS.items():
        config_file = tmp_path / f"mcp_{name}.json"
        config_file.write_text(json.dumps(config, indent=2))
        configs[name] = config_file
    return configs


@pytest.fixture
def mock_claude_binary(tmp_path):
    """Create a mock Claude binary for integration testing."""
    binary_path = tmp_path / "mock_claude"
    binary_script = '''#!/bin/bash
# Mock Claude binary for testing
echo "Mock Claude response: $@"
'''
    binary_path.write_text(binary_script)
    binary_path.chmod(0o755)
    return binary_path


@pytest.fixture
def configured_wrapper(sample_mcp_config):
    """Create a wrapper with comprehensive configuration."""
    config = ClaudeCodeConfig(
        claude_binary="claude",
        timeout=30.0,
        max_turns=5,
        verbose=True,
        system_prompt="You are a helpful testing assistant.",
        allowed_tools=["Python", "Bash", "FileSystem"],
        mcp_config_path=sample_mcp_config["filesystem"],
        environment_vars={"CLAUDE_TEST": "1"}
    )
    return ClaudeCodeWrapper(config)


# Test utilities
class MockResponseBuilder:
    """Builder for creating mock responses."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self._stdout = ""
        self._stderr = ""
        self._returncode = 0
        self._json_data = None
        return self
    
    def with_text(self, text: str):
        """Set text response."""
        self._stdout = text
        return self
    
    def with_json(self, data: Dict[str, Any]):
        """Set JSON response."""
        self._json_data = data
        self._stdout = json.dumps(data)
        return self
    
    def with_error(self, stderr: str, returncode: int = 1):
        """Set error response."""
        self._stderr = stderr
        self._returncode = returncode
        return self
    
    def build(self):
        """Build the mock process."""
        from test_claude_code_wrapper import MockProcess
        return MockProcess(
            stdout=self._stdout,
            stderr=self._stderr,
            returncode=self._returncode
        )


class StreamingMockBuilder:
    """Builder for creating streaming mock responses."""
    
    def __init__(self):
        self.responses = []
    
    def add_init(self, session_id: str = "test_session"):
        """Add init message."""
        self.responses.append({
            "type": "init",
            "session_id": session_id,
            "timestamp": "2025-01-01T12:00:00Z"
        })
        return self
    
    def add_message(self, content: str, role: str = "assistant"):
        """Add message."""
        self.responses.append({
            "type": "message",
            "role": role,
            "content": content
        })
        return self
    
    def add_tool_use(self, tool: str, action: str, **kwargs):
        """Add tool use."""
        self.responses.append({
            "type": "tool_use",
            "tool": tool,
            "action": action,
            **kwargs
        })
        return self
    
    def add_tool_result(self, tool: str, result: str):
        """Add tool result."""
        self.responses.append({
            "type": "tool_result",
            "tool": tool,
            "result": result
        })
        return self
    
    def add_result(self, status: str = "complete", **stats):
        """Add final result."""
        self.responses.append({
            "type": "result",
            "status": status,
            "stats": stats
        })
        return self
    
    def build_mock_process(self):
        """Build mock process for streaming."""
        mock_process = MagicMock()
        mock_process.stdout = [json.dumps(resp) + '\n' for resp in self.responses]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        return mock_process


# Test scenarios
class TestScenarios:
    """Pre-defined test scenarios."""
    
    @staticmethod
    def simple_query():
        """Simple question-answer scenario."""
        return {
            "query": "What is Python?",
            "expected_response": "Python is a high-level programming language...",
            "config": {}
        }
    
    @staticmethod
    def code_generation():
        """Code generation scenario."""
        return {
            "query": "Write a Python function to read a CSV file",
            "expected_response": TestConfig.SAMPLE_RESPONSES["code_generation"],
            "config": {
                "allowed_tools": ["Python"],
                "max_turns": 3
            }
        }
    
    @staticmethod
    def multi_turn_conversation():
        """Multi-turn conversation scenario."""
        return {
            "turns": [
                {"query": "Hello, I need help with Python.", "response": "Hello! I'd be happy to help with Python."},
                {"query": "How do I read a file?", "response": "You can use the open() function..."},
                {"query": "Can you show an example?", "response": "Here's an example: with open('file.txt', 'r') as f:..."}
            ],
            "config": {"max_turns": 5}
        }
    
    @staticmethod
    def error_handling():
        """Error handling scenarios."""
        return {
            "timeout": {
                "query": "This will timeout",
                "config": {"timeout": 0.1},
                "expected_error": "ClaudeCodeTimeoutError"
            },
            "process_error": {
                "query": "This will fail",
                "mock_returncode": 1,
                "mock_stderr": "Process failed",
                "expected_error": "ClaudeCodeProcessError"
            }
        }


# Test helpers
def run_test_suite(test_filter: str = None, verbose: bool = True):
    """Run the test suite with optional filtering."""
    args = []
    
    if test_filter:
        args.extend(["-k", test_filter])
    
    if verbose:
        args.append("-v")
    
    # Add coverage if available
    try:
        import pytest_cov
        args.extend(["--cov=claude_code_wrapper", "--cov-report=html"])
    except ImportError:
        pass
    
    # Run tests
    return pytest.main(args)


def benchmark_wrapper(queries: List[str], iterations: int = 10):
    """Benchmark the wrapper with a list of queries."""
    import time
    from unittest.mock import patch
    
    results = []
    wrapper = ClaudeCodeWrapper()
    
    # Mock the subprocess to avoid actual calls
    mock_response = MockResponseBuilder().with_text("Benchmark response").build()
    
    with patch('claude_code_wrapper.subprocess.run', return_value=mock_response):
        for query in queries:
            times = []
            for _ in range(iterations):
                start = time.time()
                wrapper.run(query)
                end = time.time()
                times.append(end - start)
            
            results.append({
                "query": query,
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            })
    
    return results


# Integration test helpers
def create_integration_test_env():
    """Create environment for integration testing."""
    env = {
        "CLAUDE_TEST_MODE": "1",
        "CLAUDE_LOG_LEVEL": "DEBUG",
        "CLAUDE_CONFIG_PATH": "/tmp/claude_test_config"
    }
    return env


# Main test runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Claude Code Wrapper tests")
    parser.add_argument("--filter", "-k", help="Filter tests by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmarks
        queries = [
            "What is Python?",
            "Write a hello world program",
            "Explain machine learning",
            "Create a REST API"
        ]
        results = benchmark_wrapper(queries)
        print("Benchmark Results:")
        for result in results:
            print(f"Query: {result['query']}")
            print(f"  Avg: {result['avg_time']:.4f}s")
            print(f"  Min: {result['min_time']:.4f}s")
            print(f"  Max: {result['max_time']:.4f}s")
            print()
    else:
        # Run tests
        filter_arg = args.filter
        
        if args.integration:
            filter_arg = "integration" if not filter_arg else f"{filter_arg} and integration"
        elif args.unit:
            filter_arg = "not integration" if not filter_arg else f"{filter_arg} and not integration"
        
        exit_code = run_test_suite(filter_arg, args.verbose)
        sys.exit(exit_code)
