# API Reference

Complete reference for all classes, methods, and functions in the Claude Code SDK Wrapper.

## Main Classes

### ClaudeCodeWrapper

Primary wrapper class for interacting with Claude Code.

#### Constructor

```python
ClaudeCodeWrapper(config: Optional[ClaudeCodeConfig] = None)
```

**Parameters:**
- `config`: Configuration object. If None, uses default configuration.

**Example:**
```python
wrapper = ClaudeCodeWrapper()
# or
config = ClaudeCodeConfig(timeout=30.0)
wrapper = ClaudeCodeWrapper(config)
```

#### Methods

##### run()

Execute a single query with Claude Code.

```python
run(query: str, output_format: OutputFormat = OutputFormat.TEXT, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `query` (str): The query/prompt to send to Claude Code
- `output_format` (OutputFormat): Output format (TEXT, JSON, STREAM_JSON)
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object

**Raises:**
- `ClaudeCodeValidationError`: Invalid input parameters
- `ClaudeCodeTimeoutError`: Request timed out
- `ClaudeCodeProcessError`: Claude Code process failed
- `ClaudeCodeError`: General error

**Example:**
```python
response = wrapper.run("What is Python?")
response = wrapper.run("Generate code", output_format=OutputFormat.JSON, timeout=30.0)
```

##### run_streaming()

Execute query with streaming response.

```python
run_streaming(query: str, **kwargs) -> Iterator[Dict[str, Any]]
```

**Parameters:**
- `query` (str): The query/prompt to send to Claude Code
- `**kwargs`: Configuration overrides

**Yields:** Dictionary events from streaming response

**Example:**
```python
for event in wrapper.run_streaming("Write a long story"):
    if event.get("type") == "message":
        print(event.get("content", ""), end="")
```

##### resume_session()

Resume a specific session.

```python
resume_session(session_id: str, query: str, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `session_id` (str): Session ID to resume
- `query` (str): Query to execute in session
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object

##### continue_last_session()

Continue the most recent session.

```python
continue_last_session(query: str, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `query` (str): Query to execute
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object

##### session()

Context manager for session-based conversations.

```python
session(**session_config) -> ClaudeCodeSession
```

**Parameters:**
- `**session_config`: Session configuration overrides

**Returns:** `ClaudeCodeSession` context manager

**Example:**
```python
with wrapper.session() as session:
    response1 = session.ask("First question")
    response2 = session.ask("Follow-up question")
```

##### get_metrics()

Get performance and usage metrics.

```python
get_metrics() -> Dict[str, Any]
```

**Returns:** Dictionary with metrics:
- `total_requests`: Total number of requests made
- `error_count`: Number of requests that resulted in errors
- `total_execution_time`: Total time spent executing requests

### ClaudeCodeConfig

Configuration dataclass for Claude Code execution.

#### Constructor

```python
ClaudeCodeConfig(
    claude_binary: str = "claude",
    timeout: Optional[float] = 60.0,
    max_turns: Optional[int] = None,
    verbose: bool = False,
    session_id: Optional[str] = None,
    continue_session: bool = False,
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    allowed_tools: List[str] = field(default_factory=list),
    disallowed_tools: List[str] = field(default_factory=list),
    mcp_config_path: Optional[Path] = None,
    permission_prompt_tool: Optional[str] = None,
    working_directory: Optional[Path] = None,
    environment_vars: Dict[str, str] = field(default_factory=dict),
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    enable_metrics: bool = True,
    log_level: int = logging.INFO
)
```

#### Fields

##### Core Settings
- `claude_binary` (str): Path to Claude Code binary (default: "claude")
- `timeout` (Optional[float]): Request timeout in seconds (default: 60.0)
- `max_turns` (Optional[int]): Maximum conversation turns (default: None)
- `verbose` (bool): Enable verbose logging (default: False)

##### Session Management
- `session_id` (Optional[str]): Specific session to resume (default: None)
- `continue_session` (bool): Continue last session (default: False)

##### System Prompts
- `system_prompt` (Optional[str]): Custom system prompt (default: None)
- `append_system_prompt` (Optional[str]): Additional system prompt (default: None)

##### Tool Configuration
- `allowed_tools` (List[str]): Allowed tool patterns (default: [])
- `disallowed_tools` (List[str]): Disallowed tool patterns (default: [])

##### MCP Configuration
- `mcp_config_path` (Optional[Path]): MCP configuration file path (default: None)
- `permission_prompt_tool` (Optional[str]): MCP permission tool (default: None)
- `mcp_auto_approval` (Dict[str, Any]): Auto-approval configuration (default: {})
  - `enabled` (bool): Enable auto-approval
  - `strategy` (str): Approval strategy ("all", "none", "allowlist", "patterns")
  - `allowlist` (List[str]): Tools to approve (for "allowlist" strategy)
  - `allow_patterns` (List[str]): Regex patterns to allow (for "patterns" strategy)
  - `deny_patterns` (List[str]): Regex patterns to deny (for "patterns" strategy)

##### Environment
- `working_directory` (Optional[Path]): Execution directory (default: None)
- `environment_vars` (Dict[str, str]): Environment variables (default: {})

##### Resilience Settings
- `max_retries` (int): Maximum retry attempts (default: 3)
- `retry_delay` (float): Base retry delay in seconds (default: 1.0)
- `retry_backoff_factor` (float): Retry backoff multiplier (default: 2.0)

##### Observability
- `enable_metrics` (bool): Enable metrics collection (default: True)
- `log_level` (int): Logging level (default: logging.INFO)

### ClaudeCodeResponse

Response object containing execution results and metadata.

#### Fields

##### Core Response Data
- `content` (str): Response content from Claude Code
- `returncode` (int): Process return code
- `session_id` (Optional[str]): Session identifier
- `is_error` (bool): Whether response indicates an error
- `error_type` (Optional[str]): Type of error if is_error is True
- `error_subtype` (Optional[str]): Error subtype for detailed classification

##### Metadata
- `metadata` (Dict[str, Any]): Additional response metadata
- `raw_output` (str): Raw process output
- `stderr` (str): Error output from process
- `execution_time` (float): Time taken to execute request
- `timestamp` (float): Timestamp when response was created

##### Metrics
- `metrics` (ClaudeCodeMetrics): Performance and usage metrics

### ClaudeCodeMetrics

Metrics and telemetry data from Claude Code execution.

#### Fields

- `cost_usd` (float): Cost in USD for the request (default: 0.0)
- `duration_ms` (int): Total duration in milliseconds (default: 0)
- `duration_api_ms` (int): API-specific duration in milliseconds (default: 0)
- `num_turns` (int): Number of conversation turns (default: 0)
- `total_cost` (float): Total accumulated cost (default: 0.0)
- `tokens_used` (Optional[int]): Number of tokens used (default: None)
- `model_used` (Optional[str]): Model identifier used (default: None)

### ClaudeCodeSession

Session wrapper for multi-turn conversations with state management.

#### Constructor

```python
ClaudeCodeSession(wrapper: ClaudeCodeWrapper, **config)
```

**Parameters:**
- `wrapper`: ClaudeCodeWrapper instance
- `**config`: Session configuration overrides

#### Methods

##### ask()

Ask a question in the current session.

```python
ask(query: str, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `query` (str): Question to ask
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object

##### ask_streaming()

Ask with streaming response in session context.

```python
ask_streaming(query: str, **kwargs) -> Iterator[Dict[str, Any]]
```

**Parameters:**
- `query` (str): Question to ask
- `**kwargs`: Configuration overrides

**Yields:** Dictionary events from streaming response

##### get_history()

Get conversation history.

```python
get_history() -> List[ClaudeCodeResponse]
```

**Returns:** List of all responses in the session

##### clear_history()

Clear conversation history and reset session.

```python
clear_history() -> None
```

## Enums

### OutputFormat

Supported output formats for Claude Code.

```python
class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"
```

### ErrorSeverity

Error severity levels for classification.

```python
class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

## Exception Hierarchy

### ClaudeCodeError

Base exception for all Claude Code wrapper errors.

```python
class ClaudeCodeError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None)
```

**Attributes:**
- `severity` (ErrorSeverity): Error severity level
- `context` (Dict[str, Any]): Additional error context
- `timestamp` (float): Error occurrence timestamp

### ClaudeCodeTimeoutError

Raised when Claude Code execution times out.

```python
class ClaudeCodeTimeoutError(ClaudeCodeError):
    def __init__(self, timeout_duration: float, context: Optional[Dict[str, Any]] = None)
```

**Additional Attributes:**
- `timeout_duration` (float): Duration that caused timeout

### ClaudeCodeProcessError

Raised when Claude Code process fails.

```python
class ClaudeCodeProcessError(ClaudeCodeError):
    def __init__(self, message: str, returncode: int, stderr: str = "",
                 context: Optional[Dict[str, Any]] = None)
```

**Additional Attributes:**
- `returncode` (int): Process return code
- `stderr` (str): Standard error output

### ClaudeCodeValidationError

Raised when input validation fails.

```python
class ClaudeCodeValidationError(ClaudeCodeError):
    def __init__(self, message: str, field: str = "", value: Any = None,
                 context: Optional[Dict[str, Any]] = None)
```

**Additional Attributes:**
- `field` (str): Field that failed validation
- `value` (Any): Value that caused validation failure

### ClaudeCodeConfigurationError

Raised when configuration is invalid.

```python
class ClaudeCodeConfigurationError(ClaudeCodeError):
    def __init__(self, message: str, config_field: str = "",
                 context: Optional[Dict[str, Any]] = None)
```

**Additional Attributes:**
- `config_field` (str): Configuration field with issue

## Convenience Functions

### ask_claude()

Quick function to ask Claude with error handling.

```python
ask_claude(query: str, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `query` (str): Query to ask Claude
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object (never raises exceptions)

### ask_claude_json()

Quick function to ask Claude with JSON output.

```python
ask_claude_json(query: str, **kwargs) -> ClaudeCodeResponse
```

**Parameters:**
- `query` (str): Query to ask Claude
- `**kwargs`: Configuration overrides

**Returns:** `ClaudeCodeResponse` object with JSON output format

### ask_claude_streaming()

Quick function for streaming with comprehensive error handling.

```python
ask_claude_streaming(query: str, **kwargs) -> Iterator[Dict[str, Any]]
```

**Parameters:**
- `query` (str): Query to ask Claude
- `**kwargs`: Configuration overrides

**Yields:** Dictionary events from streaming response

## Utility Classes

### ClaudeCodeLogger

Centralized logging configuration for Claude Code operations.

#### Methods

##### setup_logger()

Set up structured logging with consistent format.

```python
@staticmethod
setup_logger(name: str, level: int = logging.INFO) -> logging.Logger
```

**Parameters:**
- `name` (str): Logger name
- `level` (int): Logging level

**Returns:** Configured logger instance

### CircuitBreaker

Circuit breaker pattern for resilient external service calls.

#### Constructor

```python
CircuitBreaker(failure_threshold: int = 5, recovery_timeout: float = 60.0)
```

**Parameters:**
- `failure_threshold` (int): Number of failures before opening circuit
- `recovery_timeout` (float): Time to wait before trying again

#### Methods

##### call()

Execute function with circuit breaker protection.

```python
call(func: Callable, *args, **kwargs)
```

**Parameters:**
- `func` (Callable): Function to execute
- `*args`: Function arguments
- `**kwargs`: Function keyword arguments

**Returns:** Function result

**Raises:** `ClaudeCodeError` if circuit is open

## Decorators

### retry_with_backoff()

Retry decorator with exponential backoff.

```python
retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0,
                  backoff_factor: float = 2.0, max_delay: float = 60.0)
```

**Parameters:**
- `max_retries` (int): Maximum retry attempts
- `base_delay` (float): Base delay between retries
- `backoff_factor` (float): Multiplier for delay
- `max_delay` (float): Maximum delay between retries

**Usage:**
```python
@retry_with_backoff(max_retries=3, base_delay=1.0)
def my_function():
    # Function that might fail
    pass
```

## Type Hints

The wrapper uses comprehensive type hints throughout:

```python
from typing import Dict, List, Optional, Union, Any, Iterator, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
```

## Best Practices

### Error Handling
Always handle exceptions appropriately:

```python
try:
    response = wrapper.run(query)
    if response.is_error:
        handle_response_error(response)
    else:
        process_success(response)
except ClaudeCodeTimeoutError:
    handle_timeout()
except ClaudeCodeError as e:
    handle_general_error(e)
```

### Resource Management
Use context managers for sessions:

```python
with wrapper.session() as session:
    # Session automatically cleaned up
    response = session.ask("Question")
```

### Configuration
Validate configuration early:

```python
try:
    config = ClaudeCodeConfig(timeout=60.0)
    wrapper = ClaudeCodeWrapper(config)
except ClaudeCodeConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Logging
Use structured logging:

```python
import logging
logger = logging.getLogger(__name__)

response = wrapper.run(query)
logger.info(f"Query completed: {len(response.content)} chars, "
           f"${response.metrics.cost_usd:.6f}")
```
