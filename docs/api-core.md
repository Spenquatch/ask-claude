# Core API Reference

Main classes and functions for using Ask Claude - Claude Code SDK Wrapper.

## ClaudeCodeWrapper

Primary wrapper class for interacting with Claude Code.

### Constructor

```python
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

wrapper = ClaudeCodeWrapper(config: Optional[ClaudeCodeConfig] = None)
```

**Parameters:**
- `config`: Configuration object. If None, uses default configuration.

### Methods

#### run()

Execute a single query with Claude.

```python
response = wrapper.run(
    query: str,
    output_format: OutputFormat = OutputFormat.TEXT,
    timeout: Optional[float] = None,
    **kwargs
) -> ClaudeCodeResponse
```

**Parameters:**
- `query`: The query to send to Claude
- `output_format`: Response format (TEXT or JSON)
- `timeout`: Request timeout in seconds
- `**kwargs`: Additional options passed to Claude CLI

**Returns:** `ClaudeCodeResponse` object

#### run_streaming()

Stream responses from Claude in real-time.

```python
for event in wrapper.run_streaming(query: str, **kwargs):
    # Process streaming events
    pass
```

**Yields:** Dictionary events from Claude Code CLI

#### session()

Create a session for multi-turn conversations.

```python
with wrapper.session() as session:
    response1 = session.ask("Hello")
    response2 = session.ask("What did I just say?")
```

**Returns:** `ClaudeCodeSession` context manager

## ClaudeCodeConfig

Configuration for the wrapper behavior.

### Constructor

```python
config = ClaudeCodeConfig(
    timeout: float = 120.0,
    max_retries: int = 3,
    enable_logging: bool = True,
    log_level: int = logging.INFO,
    working_directory: Optional[Path] = None,
    mcp_config_path: Optional[Path] = None,
    mcp_auto_approval: Optional[Dict[str, Any]] = None
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 120.0 | Request timeout in seconds |
| `max_retries` | int | 3 | Number of retry attempts |
| `enable_logging` | bool | True | Enable wrapper logging |
| `working_directory` | Path | None | Working directory for Claude |
| `mcp_auto_approval` | dict | None | MCP tool auto-approval config |

## ClaudeCodeResponse

Response object containing Claude's output and metadata.

### Properties

```python
response.content       # str: Main response content
response.is_error      # bool: Whether response contains an error
response.error_type    # Optional[str]: Error type if is_error=True
response.session_id    # str: Session identifier
response.execution_time # float: Time taken for request
response.metrics       # ResponseMetrics: Usage metrics
```

## Convenience Functions

### ask_claude()

Simple function for one-off queries.

```python
from ask_claude.wrapper import ask_claude

response = ask_claude("What is Python?")
print(response.content)
```

### ask_claude_json()

Get structured JSON responses.

```python
from ask_claude.wrapper import ask_claude_json

data = ask_claude_json("List 3 Python frameworks as JSON")
```

### ask_claude_streaming()

Stream responses with a simple function.

```python
from ask_claude.wrapper import ask_claude_streaming

for event in ask_claude_streaming("Write a story"):
    if event.get("type") == "assistant":
        print(event.get("content", ""), end="")
```

## Best Practices

1. **Use sessions** for multi-turn conversations
2. **Set timeouts** for long-running queries
3. **Handle exceptions** gracefully (see [Error Handling](api-exceptions.md))
4. **Configure logging** for production deployments
5. **Use MCP auto-approval** for trusted tool access

## Next Steps

- [Exception Handling](api-exceptions.md) - Error types and handling
- [MCP Integration](mcp-integration.md) - Model Context Protocol features
- [Configuration Guide](configuration.md) - Detailed configuration options
