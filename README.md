# Claude Code SDK Wrapper

**Enterprise-grade Python wrapper for Claude Code SDK with comprehensive error handling, session management, and production-ready features.**

## 🚀 Quick Start

```python
from claude_code_wrapper import ask_claude, ask_claude_json

# Simple text query
response = ask_claude("What is Python?")
print(response.content)

# JSON response with metadata
response = ask_claude_json("Explain machine learning briefly")
print(f"Content: {response.content}")
print(f"Cost: ${response.metrics.cost_usd:.6f}")
print(f"Session: {response.session_id}")
```

## 🌟 Key Features

- **🛡️ Enterprise Error Handling**: Graceful degradation with comprehensive exception hierarchy
- **📊 Session Management**: Multi-turn conversations with state tracking
- **🌊 Streaming Support**: Real-time streaming responses with error recovery
- **✅ Input Validation**: Comprehensive request validation and sanitization
- **🔧 CLI Tool**: Production-ready command-line interface
- **📦 Zero Dependencies**: Uses only Python standard library
- **🔄 Resilience**: Automatic retries with exponential backoff
- **🤖 MCP Auto-Approval**: Programmatic approval of MCP tools without manual prompts

## 📋 Quick Links

- [Installation Guide](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Usage Examples](docs/usage-examples.md)
- [API Reference](docs/api-reference.md)
- [CLI Documentation](docs/cli-usage.md)
- [MCP Integration](docs/mcp-integration.md)
- [Production Deployment](docs/production.md)
- [Error Handling](docs/error-handling.md)

## 📦 Installation

```bash
# Install Claude Code CLI first (follow official documentation)
# Then clone this wrapper
git clone <repository-url>
cd ask_claude
pip install -r requirements.txt
```

## 🔧 Basic Usage

### Simple Queries
```python
from claude_code_wrapper import ClaudeCodeWrapper

wrapper = ClaudeCodeWrapper()
response = wrapper.run("What is 2+2?")
print(response.content)  # "4"
```

### Session Management
```python
# Multi-turn conversation
with wrapper.session() as session:
    response1 = session.ask("I need help with Python.")
    response2 = session.ask("How do I read CSV files?")
    response3 = session.ask("Can you show an example?")
```

### CLI Usage
```bash
# Ask a question
python cli_tool.py ask "What is Python?" --format json

# Start interactive session
python cli_tool.py session --interactive

# Stream a response
python cli_tool.py stream "Write a tutorial"

# Check health
python cli_tool.py health
```

## 🧪 Testing

Run the demonstration script to verify everything works:

```bash
python getting_started.py
```

For production usage examples:

```bash
python production_example.py
```

## 🤖 MCP Auto-Approval (New!)

Enable automatic approval of MCP (Model Context Protocol) tools without manual prompts:

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig
from pathlib import Path

# Auto-approve all MCP tools (development)
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)

# Auto-approve specific tools only (production)
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__filesystem__read_file", "mcp__database__query"]
    }
)

# Pattern-based approval (advanced)
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": ["mcp__.*__read.*", "mcp__.*__list.*"],
        "deny_patterns": ["mcp__.*__write.*", "mcp__.*__delete.*"]
    }
)

wrapper = ClaudeCodeWrapper(config)
```

### CLI Usage

```bash
# Auto-approve all tools
python cli_tool.py ask "Analyze the codebase" \
    --mcp-config mcp-servers.json \
    --approval-strategy all

# Allowlist specific tools
python cli_tool.py stream "Read project documentation" \
    --mcp-config mcp-servers.json \
    --approval-strategy allowlist \
    --approval-allowlist "mcp__filesystem__read_file" "mcp__filesystem__list_directory"
```

## 📁 Project Structure

```
ask_claude/
├── claude_code_wrapper.py    # Main wrapper library
├── cli_tool.py              # Command-line interface
├── getting_started.py       # Demo and test script
├── production_example.py    # Production usage examples
├── config_examples.json     # Configuration examples
├── requirements.txt         # Dependencies
├── docs/                    # Documentation
└── safe_to_delete/         # Redundant files (safe to remove)
```

## 🛡️ Error Handling

The wrapper provides comprehensive error handling:

```python
from claude_code_wrapper import ClaudeCodeError, ClaudeCodeTimeoutError

try:
    response = wrapper.run("Complex query", timeout=30.0)
    if response.is_error:
        print(f"Response error: {response.error_type}")
    else:
        print(response.content)
except ClaudeCodeTimeoutError:
    print("Query timed out")
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

## 🤝 Contributing

1. Check the current implementation works: `python getting_started.py`
2. Make your changes
3. Test thoroughly
4. Update documentation as needed

## 📄 License

MIT License - see LICENSE file for details.

---

**Built for Production** | **Zero Dependencies** | **Enterprise Ready**
