# Claude Code SDK Wrapper

A lightweight Python wrapper for the Claude Code CLI that adds enterprise features like error handling, session management, and MCP auto-approval—all with zero dependencies.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Simple API** - One-line queries with `ask_claude()`
- 🔄 **Automatic Retries** - Built-in resilience with exponential backoff
- 💬 **Session Management** - Multi-turn conversations with context
- 🤖 **MCP Auto-Approval** - Bypass manual tool approval prompts
- 🌊 **Streaming Support** - Real-time response streaming
- 📦 **Zero Dependencies** - Uses only Python standard library
- 🛡️ **Enterprise Ready** - Comprehensive error handling and logging

## Installation

```bash
# 1. Install Claude Code CLI first (see Anthropic docs)
# 2. Clone this wrapper
git clone <repository-url>
cd ask_claude

# 3. Verify it works
python getting_started.py
```

## Quick Start

```python
from claude_code_wrapper import ask_claude

# Simple query
response = ask_claude("What is Python?")
print(response.content)

# With model selection
response = ask_claude("Write a haiku", model="opus")
print(response.content)
```

## Common Use Cases

### CLI Usage

```bash
# Ask a question
python cli_tool.py ask "What is Python?"

# Stream a response
python cli_tool.py stream "Write a story"

# Interactive session
python cli_tool.py session -i
```

### Session Management

```python
from claude_code_wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
with wrapper.session() as session:
    session.ask("I need help with Python")
    session.ask("How do I read CSV files?")
    response = session.ask("Show me an example")
```

### MCP Auto-Approval

```python
from claude_code_wrapper import ClaudeCodeConfig, ClaudeCodeWrapper

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

### Error Handling

```python
from claude_code_wrapper import ClaudeCodeError, ClaudeCodeTimeoutError

try:
    response = wrapper.ask("Complex query", timeout=30.0)
    print(response.content)
except ClaudeCodeTimeoutError:
    print("Request timed out")
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Configuration](docs/configuration.md) | All configuration options |
| [API Reference](docs/api-reference.md) | Complete API documentation |
| [MCP Integration](docs/mcp-integration.md) | Using MCP tools and auto-approval |
| [CLI Usage](docs/cli-usage.md) | Command-line interface guide |
| [Examples](examples/) | Working code examples |

## Project Structure

```
ask_claude/
├── claude_code_wrapper.py   # Main wrapper module
├── cli_tool.py             # CLI interface
├── approval_strategies.py   # MCP approval logic
├── docs/                   # Documentation
├── examples/               # Example scripts
└── tests/                  # Test suite
```

## Requirements

- Python 3.9+
- Claude Code CLI installed
- No Python dependencies (stdlib only)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/yourusername/ask_claude/issues)
- 💬 [Discussions](https://github.com/yourusername/ask_claude/discussions)

---

Built with ❤️ for the Claude community