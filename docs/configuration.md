# Configuration Guide

Ask Claude - Claude Code SDK Wrapper provides extensive configuration options for production use.

## Configuration Methods

### 1. Using ClaudeCodeConfig Class

```python
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

config = ClaudeCodeConfig(
    claude_binary="claude",
    timeout=60.0,
    max_retries=3,
    verbose=True,
    system_prompt="You are a helpful coding assistant."
)

wrapper = ClaudeCodeWrapper(config)
```

### 2. Using JSON Configuration File

Create `config.json`:
```json
{
  "claude_binary": "claude",
  "timeout": 60.0,
  "max_turns": 10,
  "verbose": false,
  "system_prompt": "You are a helpful assistant.",
  "max_retries": 3,
  "retry_delay": 1.0,
  "enable_metrics": true
}
```

Load configuration:
```python
import json
from pathlib import Path

config_path = Path("config.json")
with open(config_path) as f:
    config_data = json.load(f)

config = ClaudeCodeConfig(**config_data)
wrapper = ClaudeCodeWrapper(config)
```

### 3. Environment Variables

```bash
export CLAUDE_BINARY="/usr/local/bin/claude"
export CLAUDE_TIMEOUT="60"
export CLAUDE_MAX_RETRIES="3"
export CLAUDE_VERBOSE="true"
```

```python
import os

config = ClaudeCodeConfig(
    claude_binary=os.getenv("CLAUDE_BINARY", "claude"),
    timeout=float(os.getenv("CLAUDE_TIMEOUT", "60")),
    max_retries=int(os.getenv("CLAUDE_MAX_RETRIES", "3")),
    verbose=os.getenv("CLAUDE_VERBOSE", "").lower() == "true"
)
```

## Configuration Parameters

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `claude_binary` | str | "claude" | Path to Claude Code binary |
| `timeout` | float | 60.0 | Request timeout in seconds |
| `max_turns` | int | None | Maximum conversation turns |
| `verbose` | bool | False | Enable verbose logging |

### Session Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | str | None | Specific session to resume |
| `continue_session` | bool | False | Continue last session |

### System Prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | str | None | Custom system prompt |
| `append_system_prompt` | str | None | Additional system prompt |

### Tool Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allowed_tools` | List[str] | [] | Allowed tool patterns |
| `disallowed_tools` | List[str] | [] | Disallowed tool patterns |

### Environment Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_directory` | Path | None | Execution directory |
| `environment_vars` | Dict[str, str] | {} | Environment variables |

### Resilience Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | int | 3 | Maximum retry attempts |
| `retry_delay` | float | 1.0 | Base retry delay |
| `retry_backoff_factor` | float | 2.0 | Retry backoff multiplier |

### Observability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_metrics` | bool | True | Enable metrics collection |
| `log_level` | int | 20 | Logging level (INFO=20) |

## Configuration Examples

### Basic Development Setup

```python
config = ClaudeCodeConfig(
    timeout=30.0,
    verbose=True,
    enable_metrics=True,
    log_level=10  # DEBUG
)
```

### Production Setup

```python
config = ClaudeCodeConfig(
    timeout=60.0,
    max_retries=5,
    retry_delay=2.0,
    system_prompt="You are a professional assistant providing accurate information.",
    enable_metrics=True,
    log_level=20,  # INFO
    environment_vars={
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "INFO"
    }
)
```

### High-Performance Setup

```python
config = ClaudeCodeConfig(
    timeout=30.0,
    max_retries=2,
    retry_delay=0.5,
    retry_backoff_factor=1.5,
    enable_metrics=True,
    max_turns=5,
    verbose=False
)
```

### Security-Focused Setup

```python
from pathlib import Path

config = ClaudeCodeConfig(
    allowed_tools=[
        "Python(import,def,class,print)",  # Specific Python operations
        "Bash(ls,cat,grep,head,tail)"      # Safe bash commands only
    ],
    disallowed_tools=[
        "Bash(rm,del,sudo,chmod)",         # Dangerous commands
        "Python(exec,eval,import os)"      # Potentially unsafe Python
    ],
    working_directory=Path("./secure_workspace"),
    timeout=30.0,
    max_turns=3
)
```

### Model Context Protocol (MCP) Setup

First, create `mcp_config.json`:
```json
{
  "servers": {
    "filesystem": {
      "command": "mcp-server-filesystem",
      "args": ["/project/workspace"]
    },
    "database": {
      "command": "mcp-server-sqlite",
      "args": ["./data/app.db"]
    }
  }
}
```

Then configure the wrapper:
```python
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp_config.json"),
    allowed_tools=[
        "Python",
        "mcp__filesystem__*",  # All filesystem MCP tools
        "mcp__database__read", # Specific database operations
        "mcp__database__query"
    ],
    disallowed_tools=[
        "mcp__filesystem__delete",
        "mcp__database__write"
    ]
)
```

## Configuration Validation

The wrapper automatically validates configuration:

```python
try:
    config = ClaudeCodeConfig(
        timeout=-1.0,  # Invalid: negative timeout
        max_retries=-1  # Invalid: negative retries
    )
except ClaudeCodeConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Field: {e.config_field}")
```

## Environment-Specific Configurations

### Development
```python
# config/development.json
{
  "timeout": 30.0,
  "verbose": true,
  "log_level": 10,
  "max_retries": 1,
  "enable_metrics": true
}
```

### Staging
```python
# config/staging.json
{
  "timeout": 45.0,
  "verbose": false,
  "log_level": 20,
  "max_retries": 3,
  "enable_metrics": true
}
```

### Production
```python
# config/production.json
{
  "timeout": 60.0,
  "verbose": false,
  "log_level": 30,
  "max_retries": 5,
  "retry_delay": 2.0,
  "enable_metrics": true,
  "system_prompt": "You are a professional AI assistant."
}
```

Load environment-specific config:
```python
import os

env = os.getenv("ENVIRONMENT", "development")
config_file = f"config/{env}.json"

with open(config_file) as f:
    config_data = json.load(f)

config = ClaudeCodeConfig(**config_data)
```

## Advanced Configuration

### Custom Retry Logic

```python
config = ClaudeCodeConfig(
    max_retries=5,
    retry_delay=1.0,        # Start with 1 second
    retry_backoff_factor=2.0  # Double delay each retry
)
# Retry delays: 1s, 2s, 4s, 8s, 16s
```

### Environment Variables in Configuration

```python
config = ClaudeCodeConfig(
    environment_vars={
        "API_KEY": os.getenv("API_KEY"),
        "DEBUG": "1" if os.getenv("DEBUG") else "0",
        "WORKSPACE": "/app/workspace"
    }
)
```

### Dynamic Configuration Updates

```python
# Create base config
config = ClaudeCodeConfig()
wrapper = ClaudeCodeWrapper(config)

# Override for specific requests
response = wrapper.run(
    "Complex query",
    timeout=120.0,           # Override timeout
    max_turns=10,           # Override max turns
    verbose=True            # Override verbosity
)
```

## Configuration Best Practices

### 1. Use Environment-Specific Configs
- Keep development, staging, and production configs separate
- Use environment variables for secrets and environment-specific values

### 2. Validate Configuration Early
```python
def load_config(config_path: str) -> ClaudeCodeConfig:
    """Load and validate configuration."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)

        config = ClaudeCodeConfig(**config_data)

        # Additional validation
        if config.timeout < 10:
            raise ValueError("Timeout too low for production")

        return config

    except Exception as e:
        raise ClaudeCodeConfigurationError(f"Config load failed: {e}")
```

### 3. Use Sensible Defaults
```python
config = ClaudeCodeConfig(
    timeout=float(os.getenv("CLAUDE_TIMEOUT", "60")),
    max_retries=int(os.getenv("CLAUDE_MAX_RETRIES", "3")),
    verbose=os.getenv("CLAUDE_VERBOSE", "false").lower() == "true"
)
```

### 4. Document Your Configuration
```python
# config/README.md
"""
Configuration Guide:
- timeout: Request timeout (60s recommended for production)
- max_retries: Retry attempts (3-5 for production)
- verbose: Enable for debugging only
- system_prompt: Customize for your use case
"""
```

## Troubleshooting Configuration

### Common Issues

1. **Invalid file paths**: Use `Path()` objects and verify existence
2. **Type mismatches**: Ensure JSON types match expected Python types
3. **Missing MCP servers**: Verify MCP configuration file exists and servers are available
4. **Permission issues**: Check working directory permissions

### Debug Configuration

```python
import logging

logging.basicConfig(level=logging.DEBUG)

config = ClaudeCodeConfig(
    verbose=True,
    log_level=logging.DEBUG
)

wrapper = ClaudeCodeWrapper(config)
print(f"Config: {config}")
```

This will show detailed information about configuration loading and validation.
