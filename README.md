# Claude Code SDK Wrapper

A lightweight Python wrapper for the Claude Code CLI that adds enterprise features like error handling, session management, and MCP auto-approval.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Simple API** - One-line queries with `ask_claude()`
- 🔄 **Automatic Retries** - Built-in resilience with exponential backoff
- 💬 **Session Management** - Multi-turn conversations with context
- 🤖 **MCP Auto-Approval** - Bypass manual tool approval prompts
- 🌊 **Streaming Support** - Real-time response streaming
- 🛡️ **Enterprise Ready** - Comprehensive error handling and logging

## Installation

### Option 1: From PyPI (Coming Soon - Phase 4)
```bash
pip install ask-claude
```

### Option 2: Development Installation
```bash
# 1. Install Claude Code CLI first (see Anthropic docs)
# 2. Clone and install with Poetry
git clone <repository-url>
cd ask_claude
poetry install

# 3. Verify it works
poetry run python getting_started.py
```

### Option 3: Traditional pip install
```bash
git clone <repository-url>
cd ask_claude
pip install -e .
```

## Quick Start

```python
from ask_claude import ask_claude

# Simple query
response = ask_claude("What is Python?")
print(response.content)

# With streaming
from ask_claude import ask_claude_streaming
for chunk in ask_claude_streaming("Write a haiku"):
    print(chunk.get('content', ''), end='')
```

## Common Use Cases

### CLI Usage

```bash
# After Poetry install
ask-claude ask "What is Python?"
ask-claude stream "Write a tutorial"
ask-claude session --interactive

# During development
poetry run python -m ask_claude.cli ask "What is Python?"
poetry run python -m ask_claude.cli stream "Write a tutorial"


```

### Session Management

```python
from ask_claude import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
with wrapper.session() as session:
    session.ask("I need help with Python")
    session.ask("How do I read CSV files?")
    response = session.ask("Show me an example")
```

### MCP Auto-Approval

```python
from ask_claude import ClaudeCodeConfig, ClaudeCodeWrapper

# Auto-approve specific tools
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__sequential-thinking__*"]
    }
)

wrapper = ClaudeCodeWrapper(config)
response = wrapper.run("Think through this step by step: How do I optimize this code?")
```

### Error Handling

```python
from ask_claude import ClaudeCodeError, ClaudeCodeTimeoutError

try:
    response = wrapper.run("Complex query", timeout=30.0)
    print(response.content)
except ClaudeCodeTimeoutError:
    print("Request timed out")
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Development Guide](docs/development.md) | Setup, tools, and workflows |
| [Configuration](docs/configuration.md) | All configuration options |
| [API Reference](docs/api-reference.md) | Complete API documentation |
| [MCP Integration](docs/mcp-integration.md) | Using MCP tools and auto-approval |
| [CLI Usage](docs/cli-usage.md) | Command-line interface guide |
| [Examples](examples/) | Working code examples |

## Project Structure

```
ask_claude/
├── __init__.py             # Public API exports
├── wrapper.py              # Core ClaudeCodeWrapper class
├── cli.py                  # Command-line interface
├── session.py              # Session management
├── approval/               # MCP approval system
│   ├── server.py          # Approval server
│   └── strategies.py      # Approval strategies
├── docs/                   # Documentation
├── examples/               # Example scripts
└── tests/                  # Test suite
```

## Requirements

- Python 3.10+ (required for MCP support)
- Claude Code CLI installed

## Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for detailed setup instructions.

**Quick Start for Contributors:**
1. Fork the repository
2. Set up development environment: `pyenv local 3.10.17 && pip install pre-commit && pre-commit install`
3. Create your feature branch (`git checkout -b feature/amazing-feature`)
4. Make changes and ensure all quality checks pass: `pre-commit run --all-files`
5. Commit your changes (hooks run automatically)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

**Code Quality Standards:**
- ✅ 100% type safety with mypy
- ✅ Code formatting and linting with Ruff
- ✅ All tests must pass

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/yourusername/ask_claude/issues)
- 💬 [Discussions](https://github.com/yourusername/ask_claude/discussions)

---

Built with ❤️ for the Claude community
