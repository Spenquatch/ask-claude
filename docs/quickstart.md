# Quick Start Guide

Get up and running with Claude Code SDK Wrapper in 5 minutes.

## Prerequisites

1. **Python 3.9+** - Check with `python --version`
2. **Claude Code CLI** - Install from [Anthropic](https://docs.anthropic.com/en/docs/claude-code)
3. **API Key** - Set up your Claude API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ask_claude

# Install any dependencies
pip install -r requirements.txt

# Verify it works:
python getting_started.py
```

## Your First Query

### Option 1: Simple Function

```python
from claude_code_wrapper import ask_claude

response = ask_claude("What is Python?")
print(response.content)
```

### Option 2: Using the Wrapper

```python
from claude_code_wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
response = wrapper.ask("Explain decorators in Python")
print(response.content)
```

### Option 3: Command Line

```bash
python cli_tool.py ask "What is Python?"
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
    if event.get("type") == "message":
        print(event.get("content", ""), end="")

# From CLI
python cli_tool.py stream "Write a story about AI"
```

### Error Handling

```python
from claude_code_wrapper import ClaudeCodeError, ClaudeCodeTimeoutError

try:
    response = wrapper.ask("Complex query", timeout=30.0)
except ClaudeCodeTimeoutError:
    print("Request timed out - try a shorter query")
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

## MCP Tools (Advanced)

### Basic MCP Usage

```python
# If you have MCP servers configured
wrapper = ClaudeCodeWrapper()
response = wrapper.ask("List files in the current directory")
```

### Auto-Approval

```python
from claude_code_wrapper import ClaudeCodeConfig

# Auto-approve specific tools
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__filesystem__read_file"]
    }
)

wrapper = ClaudeCodeWrapper(config)
response = wrapper.ask("Read the README.md file")
```

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
