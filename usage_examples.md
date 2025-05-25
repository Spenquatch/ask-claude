# Claude Code SDK Wrapper

A comprehensive Python wrapper around the Claude Code SDK that provides simplified methods for programmatic integration, proper error handling, session management, and structured output parsing.

## Features

- **Simple API**: Easy-to-use methods for common operations
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Session Management**: Support for multi-turn conversations
- **Multiple Output Formats**: Text, JSON, and streaming JSON support
- **MCP Integration**: Model Context Protocol support
- **Tool Configuration**: Fine-grained control over allowed/disallowed tools
- **Async Support**: Streaming responses and async patterns
- **Comprehensive Testing**: Full test suite with mocking capabilities

## Installation

```bash
# First, install Claude Code CLI (follow official documentation)
# Then install the wrapper dependencies
pip install pytest pytest-mock
```

## Quick Start

### Basic Usage

```python
from claude_code_wrapper import ClaudeCodeWrapper, OutputFormat

# Simple question
wrapper = ClaudeCodeWrapper()
response = wrapper.run("What is Python?")
print(response.content)

# With JSON output
response = wrapper.run("Explain machine learning", output_format=OutputFormat.JSON)
print(f"Content: {response.content}")
print(f"Session ID: {response.session_id}")
print(f"Metadata: {response.metadata}")
```

### Convenience Functions

```python
from claude_code_wrapper import ask_claude, ask_claude_json, ask_claude_streaming

# Quick single questions
response = ask_claude("Write a hello world program in Python")
print(response.content)

# JSON response
response = ask_claude_json("Analyze this data structure")
print(response.metadata)

# Streaming response
for event in ask_claude_streaming("Generate a long explanation"):
    if event.get("type") == "message":
        print(event.get("content", ""), end="", flush=True)
```

## Configuration

### Basic Configuration

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

config = ClaudeCodeConfig(
    claude_binary="/usr/local/bin/claude",
    timeout=60.0,
    max_turns=10,
    verbose=True,
    system_prompt="You are a helpful coding assistant."
)

wrapper = ClaudeCodeWrapper(config)
```

### Advanced Configuration with MCP

```python
from pathlib import Path

# Create MCP configuration file
mcp_config = {
    "servers": {
        "filesystem": {
            "command": "mcp-server-filesystem",
            "args": ["/project/directory"]
        },
        "database": {
            "command": "mcp-server-sqlite",
            "args": ["./database.db"]
        }
    }
}

# Save to file
mcp_path = Path("mcp_config.json")
mcp_path.write_text(json.dumps(mcp_config))

# Configure wrapper
config = ClaudeCodeConfig(
    allowed_tools=["Python", "Bash", "mcp__filesystem__*", "mcp__database__*"],
    mcp_config_path=mcp_path,
    environment_vars={"DEBUG": "1"}
)

wrapper = ClaudeCodeWrapper(config)
```

## Session Management

### Basic Sessions

```python
# Resume specific session
response = wrapper.run("Continue our discussion", session_id="session-123")

# Continue last session
response = wrapper.run("What was my last question?", continue_session=True)
```

### Session Context Manager

```python
# Multi-turn conversation
with wrapper.session(max_turns=5) as session:
    response1 = session.ask("I need help with a Python project")
    response2 = session.ask("How do I read CSV files?")
    response3 = session.ask("Can you show me an example?")
    
    # Get conversation history
    history = session.get_history()
    print(f"Had {len(history)} exchanges")
```

### Manual Session Management

```python
from claude_code_wrapper import ClaudeCodeSession

session = ClaudeCodeSession(wrapper, system_prompt="You are a Python tutor")

# First question
response1 = session.ask("Explain list comprehensions")
print(f"Session ID: {session.session_id}")

# Follow-up (automatically uses session context)
response2 = session.ask("Show me some examples")

# Streaming in session
for event in session.ask_streaming("Generate a comprehensive tutorial"):
    process_event(event)
```

## Streaming Responses

```python
# Basic streaming
for event in wrapper.run_streaming("Write a long blog post about AI"):
    match event.get("type"):
        case "init":
            print(f"Started session: {event['session_id']}")
        case "message":
            print(event.get("content", ""), end="", flush=True)
        case "tool_use":
            print(f"\n[Using tool: {event['tool']}]")
        case "result":
            print(f"\n[Complete: {event['stats']}]")

# Streaming with session
with wrapper.session() as session:
    for event in session.ask_streaming("Explain quantum computing"):
        handle_streaming_event(event)
```

## Error Handling

```python
from claude_code_wrapper import (
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError
)

try:
    response = wrapper.run("Complex query", timeout=30.0)
except ClaudeCodeTimeoutError:
    print("Query timed out")
except ClaudeCodeProcessError as e:
    print(f"Process failed with code {e.returncode}: {e.stderr}")
except ClaudeCodeError as e:
    print(f"General error: {e}")
```

## Tool Configuration

```python
# Allow specific tools
config = ClaudeCodeConfig(
    allowed_tools=[
        "Python",
        "Bash(npm install,pip install)",  # Specific commands
        "mcp__filesystem__read",          # Specific MCP tools
        "mcp__database__*"                # All database tools
    ],
    disallowed_tools=[
        "Bash(rm,del)",                   # Dangerous commands
        "mcp__web__*"                     # All web tools
    ]
)

wrapper = ClaudeCodeWrapper(config)
response = wrapper.run("Install dependencies and read config file")
```

## Testing the Wrapper

### Running Tests

```bash
# Run all tests
python test_runner.py

# Run with verbose output
python test_runner.py --verbose

# Filter tests
python test_runner.py --filter "test_session"

# Run only unit tests
python test_runner.py --unit

# Run only integration tests
python test_runner.py --integration

# Run benchmarks
python test_runner.py --benchmark
```

### Writing Custom Tests

```python
import pytest
from unittest.mock import patch, MagicMock
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeResponse

class TestMyIntegration:
    @pytest.fixture
    def wrapper(self):
        return ClaudeCodeWrapper()
    
    @patch('claude_code_wrapper.subprocess.run')
    def test_my_scenario(self, mock_run, wrapper):
        # Mock successful response
        mock_process = MagicMock()
        mock_process.stdout = "Expected response"
        mock_process.stderr = ""
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Test the wrapper
        response = wrapper.run("My test query")
        
        # Assertions
        assert response.content == "Expected response"
        assert response.returncode == 0
        
        # Verify command construction
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "claude" in args
        assert "--print" in args
        assert "My test query" in args
```

### Using Test Utilities

```python
from test_runner import MockResponseBuilder, StreamingMockBuilder

# Create mock responses easily
mock_response = (MockResponseBuilder()
                .with_json({"content": "Test response", "session_id": "test"})
                .build())

# Create streaming mocks
streaming_mock = (StreamingMockBuilder()
                 .add_init("test_session")
                 .add_message("Hello")
                 .add_message(" World!")
                 .add_result("complete", tokens=15)
                 .build_mock_process())
```

## Best Practices

### Configuration Management

```python
# Use environment variables for configuration
import os

config = ClaudeCodeConfig(
    claude_binary=os.getenv("CLAUDE_BINARY", "claude"),
    timeout=float(os.getenv("CLAUDE_TIMEOUT", "60")),
    verbose=os.getenv("CLAUDE_VERBOSE", "").lower() == "true",
    system_prompt=os.getenv("CLAUDE_SYSTEM_PROMPT"),
    environment_vars={
        "API_KEY": os.getenv("API_KEY"),
        "DEBUG": os.getenv("DEBUG", "0")
    }
)
```

### Logging Integration

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use wrapper with logging
wrapper = ClaudeCodeWrapper(config)

try:
    response = wrapper.run("Process this data")
    logger.info(f"Processed successfully: {len(response.content)} chars")
except Exception as e:
    logger.error(f"Processing failed: {e}")
```

### Production Usage

```python
# Production wrapper with proper error handling and retries
class ProductionClaudeWrapper:
    def __init__(self, config: ClaudeCodeConfig):
        self.wrapper = ClaudeCodeWrapper(config)
        self.logger = logging.getLogger(__name__)
    
    def ask_with_retry(self, query: str, max_retries: int = 3, **kwargs):
        """Ask with automatic retries on failure."""
        for attempt in range(max_retries):
            try:
                return self.wrapper.run(query, **kwargs)
            except ClaudeCodeTimeoutError:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            except ClaudeCodeError as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
    
    def batch_process(self, queries: List[str], **kwargs) -> List[ClaudeCodeResponse]:
        """Process multiple queries efficiently."""
        results = []
        with self.wrapper.session(**kwargs) as session:
            for i, query in enumerate(queries):
                try:
                    response = session.ask(query)
                    results.append(response)
                    self.logger.info(f"Processed query {i + 1}/{len(queries)}")
                except Exception as e:
                    self.logger.error(f"Failed query {i + 1}: {e}")
                    results.append(None)
        return results
```

## API Reference

### ClaudeCodeWrapper

Main wrapper class for interacting with Claude Code.

#### Methods

- `run(query, output_format=OutputFormat.TEXT, **kwargs)` - Execute single query
- `run_streaming(query, **kwargs)` - Execute with streaming response
- `resume_session(session_id, query, **kwargs)` - Resume specific session
- `continue_last_session(query, **kwargs)` - Continue most recent session
- `session(**kwargs)` - Context manager for sessions

### ClaudeCodeConfig

Configuration dataclass for Claude Code execution.

#### Fields

- `claude_binary: str` - Path to Claude binary
- `timeout: Optional[float]` - Execution timeout
- `max_turns: Optional[int]` - Maximum conversation turns
- `verbose: bool` - Enable verbose logging
- `session_id: Optional[str]` - Session ID for resumption
- `continue_session: bool` - Continue last session
- `system_prompt: Optional[str]` - Custom system prompt
- `append_system_prompt: Optional[str]` - Additional system prompt
- `allowed_tools: List[str]` - Allowed tool patterns
- `disallowed_tools: List[str]` - Disallowed tool patterns
- `mcp_config_path: Optional[Path]` - MCP configuration file
- `permission_prompt_tool: Optional[str]` - MCP permission tool
- `working_directory: Optional[Path]` - Execution directory
- `environment_vars: Dict[str, str]` - Environment variables

### ClaudeCodeResponse

Response object containing execution results.

#### Fields

- `content: str` - Response content
- `returncode: int` - Process return code
- `stderr: str` - Error output
- `session_id: Optional[str]` - Session identifier
- `metadata: Dict[str, Any]` - Additional metadata
- `raw_output: str` - Raw process output

### Exceptions

- `ClaudeCodeError` - Base exception
- `ClaudeCodeTimeoutError` - Timeout occurred  
- `ClaudeCodeProcessError` - Process execution failed

## Examples Repository

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple queries and responses
- `session_management.py` - Multi-turn conversations
- `streaming_example.py` - Real-time streaming
- `mcp_integration.py` - Model Context Protocol usage
- `error_handling.py` - Comprehensive error handling
- `production_wrapper.py` - Production-ready implementation
- `testing_examples.py` - Custom test implementations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Run the test suite: `python test_runner.py`
5. Submit a pull request

## License

This wrapper is provided as-is under the MIT License. See LICENSE file for details.
