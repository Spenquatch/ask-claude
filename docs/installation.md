# Installation Guide

## Prerequisites

### 1. Claude Code CLI

First, install the Claude Code CLI following the official Anthropic documentation.

Verify installation:
```bash
claude --version
claude --help
```

### 2. Python Requirements

- Python 3.9 or higher
- No external dependencies (uses Python standard library only)

Check your Python version:
```bash
python --version
```

## Installation Steps

### Method 1: Clone Repository (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ask_claude

# Install dependencies (minimal - only for development/testing)
pip install -r requirements.txt

# Verify installation
python getting_started.py
```

### Method 2: Copy Files

If you just need the core functionality:

1. Copy `claude_code_wrapper.py` to your project
2. Copy `cli_tool.py` if you want CLI functionality
3. Import and use:

```python
from claude_code_wrapper import ask_claude
response = ask_claude("Hello Claude!")
print(response.content)
```

## Verification

### Quick Test

```bash
# Test the wrapper
python getting_started.py
```

Expected output:
```
🧪 Claude Code SDK Wrapper - Getting Started Tests
✅ Claude binary found (exit code: 0)
✅ Basic functionality: Working
✅ JSON format: Working
✅ Streaming: Working
```

### CLI Test

```bash
# Test CLI
python cli_tool.py ask "What is 2+2?"
```

Expected output:
```
4
```

### Health Check

```bash
# Comprehensive health check
python cli_tool.py health
```

## Troubleshooting

### Common Issues

#### "Claude binary not found"
- Ensure Claude Code CLI is installed and in PATH
- Try specifying full path in configuration:
  ```python
  config = ClaudeCodeConfig(claude_binary="/full/path/to/claude")
  ```

#### Permission Errors
- Ensure Claude Code CLI has proper permissions
- Check your Claude Code configuration and authentication

#### Import Errors
- Ensure you're in the correct directory
- Check Python path: `python -c "import sys; print(sys.path)"`

### Debug Mode

Run with verbose logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from claude_code_wrapper import ClaudeCodeWrapper
wrapper = ClaudeCodeWrapper()
response = wrapper.run("Test query")
```

## Configuration

Create a basic configuration file `config.json`:

```json
{
  "claude_binary": "claude",
  "timeout": 60.0,
  "max_retries": 3,
  "enable_metrics": true,
  "log_level": 20
}
```

Load configuration:

```python
import json
from claude_code_wrapper import ClaudeCodeConfig, ClaudeCodeWrapper

with open('config.json') as f:
    config_data = json.load(f)

config = ClaudeCodeConfig(**config_data)
wrapper = ClaudeCodeWrapper(config)
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Usage Examples](usage-examples.md) - Common usage patterns
- [API Reference](api-reference.md) - Complete API documentation
- [CLI Usage](cli-usage.md) - Command-line interface guide
