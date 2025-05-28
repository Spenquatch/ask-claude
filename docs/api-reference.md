# API Reference

Complete reference for Ask Claude - Claude Code SDK Wrapper.

## 📚 Core API

**[Core Classes and Functions](api-core.md)**
- `ClaudeCodeWrapper` - Main wrapper class
- `ClaudeCodeConfig` - Configuration management
- `ClaudeCodeResponse` - Response objects
- Convenience functions: `ask_claude()`, `ask_claude_json()`, `ask_claude_streaming()`

## 🚨 Error Handling

**[Exception Types and Handling](api-exceptions.md)**
- `ClaudeCodeError` - Base exception
- `ClaudeCodeTimeoutError` - Timeout handling
- `ClaudeCodeProcessError` - CLI process errors
- Error handling patterns and best practices

## 🔧 Configuration

**[Configuration Reference](configuration.md)**
- Configuration options and examples
- Environment-specific setups
- MCP auto-approval configuration

## 🔄 Sessions

**[Session Management](session-management.md)**
- Multi-turn conversations
- Session persistence
- Advanced session patterns

## 🤖 MCP Integration

**[Model Context Protocol](mcp-integration.md)**
- MCP server configuration
- Tool auto-approval strategies
- Security considerations

## Quick Examples

### Basic Usage
```python
from ask_claude.wrapper import ask_claude

# Simple query
response = ask_claude("What is Python?")
print(response.content)
```

### Advanced Usage
```python
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Configure wrapper
config = ClaudeCodeConfig(timeout=60.0, max_retries=2)
wrapper = ClaudeCodeWrapper(config)

# Use session for conversation
with wrapper.session() as session:
    response1 = session.ask("Hello, I'm learning Python")
    response2 = session.ask("What should I learn first?")
    print(response2.content)
```

### Error Handling
```python
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeTimeoutError

try:
    wrapper = ClaudeCodeWrapper()
    response = wrapper.run("complex query", timeout=30.0)
    print(response.content)
except ClaudeCodeTimeoutError:
    print("Query took too long - try a simpler question")
```

## Migration from Old Imports

If you're updating from older versions:

```python
# OLD (deprecated)
from claude_code_wrapper import ask_claude, ClaudeCodeWrapper

# NEW (current)
from ask_claude.wrapper import ask_claude, ClaudeCodeWrapper
```

## See Also

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [CLI Usage](cli-usage.md) - Command-line interface
- [Examples](../examples/) - Working code examples
