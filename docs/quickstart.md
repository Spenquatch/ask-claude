# Quick Start Guide

Get up and running with Ask Claude - Claude Code SDK Wrapper in 5 minutes.

## Prerequisites

1. **Python 3.10+** - Check with `python --version`
2. **Claude Code CLI** - Install from [Anthropic](https://docs.anthropic.com/en/docs/claude-code)
3. **API Key** - Set up your Claude API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ask_claude

# Install with Poetry (recommended)
poetry install

# Verify it works:
poetry run python examples/getting_started.py
```

## Your First Query

### Option 1: Simple Function

```python
from ask_claude.wrapper import ask_claude

response = ask_claude("What is Python?")
print(response.content)
```

### Option 2: Using the Wrapper

```python
from ask_claude.wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
response = wrapper.run("Explain decorators in Python")
print(response.content)
```

### Option 3: Command Line

```bash
# Development
poetry run python -m ask_claude.cli ask "What is Python?"

# After installation
ask-claude ask "What is Python?"
```

## Common Patterns

### Model Selection

```python
# Use different models
response = ask_claude("Complex reasoning task", model="opus")
response = ask_claude("Balanced query", model="sonnet")
```

### Sessions (Multi-turn Conversations)

```python
from ask_claude.wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
with wrapper.session() as session:
    session.ask("I'm learning Python")
    session.ask("What are list comprehensions?")
    response = session.ask("Show me examples")
    print(response.content)
```

### Streaming Responses

```python
# In Python
for event in wrapper.run_streaming("Write a story about AI"):
    if event.get("type") == "assistant":
        print(event.get("content", ""), end="")

# From CLI
poetry run python -m ask_claude.cli stream "Write a story about AI"
```

### Error Handling

```python
from ask_claude.wrapper import ClaudeCodeError, ClaudeCodeTimeoutError

try:
    response = wrapper.run("Complex query", timeout=30.0)
except ClaudeCodeTimeoutError:
    print("Request timed out - try a shorter query")
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

## Advanced Features

For MCP integration, configuration options, and more advanced patterns, see:
- [MCP Integration Guide](mcp-integration.md)
- [Configuration Guide](configuration.md)

## Next Steps

- 📖 See [Configuration](configuration.md) for all options
- 🔧 Check [CLI Usage](cli-usage.md) for command-line features
- 🚀 Read [Production Guide](production.md) for deployment
- 💡 Explore [examples/](../examples/) for more patterns

## Troubleshooting

### "Claude not found"
Make sure Claude Code CLI is installed and in your PATH:
```bash
claude --version
```

### "No API key"
Set your API key as an environment variable or in Claude Code settings.

### Import Errors
Make sure you're in the right directory or add it to your Python path:
```python
import sys
sys.path.append('/path/to/ask_claude')
```

---

Ready to build something amazing? 🚀
