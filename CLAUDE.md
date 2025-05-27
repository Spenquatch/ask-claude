# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code SDK Wrapper** - an enterprise-grade Python wrapper around the Claude Code CLI that provides comprehensive error handling, session management, and production-ready features. The wrapper uses zero external dependencies (only Python standard library).

# IMPORTANT
We ALWAYS follow industry standards and best practices for AI/ML pipelines/development. 
## Key Architecture

### Core Components

1. **claude_code_wrapper.py** - Main wrapper module containing:
   - `ClaudeCodeWrapper`: Primary class with retry logic, circuit breaker, and metrics
   - `ClaudeCodeConfig`: Configuration validation and management
   - `ClaudeCodeResponse`: Structured responses with metadata
   - `ClaudeCodeSession`: Multi-turn conversation management
   - Comprehensive exception hierarchy
   - Convenience functions: `ask_claude()`, `ask_claude_json()`, `ask_claude_streaming()`

2. **cli_tool.py** - Command-line interface with commands:
   - `ask`: Single query execution
   - `stream`: Streaming responses
   - `session`: Interactive sessions
   - `health`: Health check
   - `benchmark`: Performance testing

3. **Configuration** - JSON-based with examples in `config_examples.json`:
   - Production, development, minimal, high-security, and performance configs
   - MCP (Model Context Protocol) support
   - Environment variable support

## Development Commands

```bash
# Run basic functionality tests
python getting_started.py

# Run comprehensive production examples
python production_example.py

# CLI usage examples
python cli_tool.py ask "What is Python?"
python cli_tool.py stream "Write a tutorial"
python cli_tool.py session --interactive
python cli_tool.py health
python cli_tool.py benchmark

# Test with specific configuration
python cli_tool.py ask "Hello" --config config.json
```

## Important Design Principles

1. **Zero Dependencies**: This project intentionally uses only Python standard library. Do not add external dependencies unless absolutely necessary.

2. **Enterprise Features**: When modifying code, maintain:
   - Comprehensive error handling with specific exception types
   - Retry logic with exponential backoff
   - Circuit breaker pattern for fault tolerance
   - Structured logging and metrics
   - Input validation and sanitization

3. **Error Handling Pattern**: The wrapper uses a hierarchical exception structure:
   - `ClaudeCodeError` (base)
   - `ClaudeCodeAPIError`, `ClaudeCodeConfigError`, `ClaudeCodeTimeoutError`, etc.
   - Always preserve error context and provide actionable error messages

4. **Session Management**: Sessions track conversation state and metrics. When working with sessions, ensure proper state management and cleanup.

5. **Configuration**: The wrapper supports multiple configuration sources (files, environment variables, programmatic). Maintain backward compatibility when modifying configuration structure.

## Testing Issues to Fix

When working on tests, these issues need to be addressed:

1. **CLI Method Names**: The ClaudeCLI class uses `cmd_*` methods, not `handle_*`:
   - `cmd_ask()`, `cmd_stream()`, `cmd_session()`, `cmd_health()`, `cmd_benchmark()`

2. **Response Object Attributes**: ClaudeCodeResponse uses:
   - `returncode` (not `exit_code`)
   - `execution_time` (not `duration`)
   - No `success` property exists

3. **Exception Classes**: Use the correct names:
   - `ClaudeCodeProcessError` (not `ClaudeCodeAPIError`)
   - `ClaudeCodeConfigurationError` (not `ClaudeCodeConfigError`)

4. **CLI Command Construction**: Test how the wrapper builds the `claude` command with various flags
5. **CLI Output Parsing**: Test how it parses JSON output from the Claude CLI
6. **Subprocess Execution**: Test how it handles the CLI process lifecycle


### MCP Auto-Approval System

The wrapper now supports automatic MCP tool approval to bypass manual prompts:

**Implemented Strategies:**
- `allowlist`: Only approve specific tools by name
- `patterns`: Approve/deny based on regex patterns
- `all`: Approve all tools (use with caution!)
- `none`: Deny all tools

**Usage in Wrapper:**
```python
config = {
    "mcp_auto_approval": {
        "enabled": true,
        "strategy": "allowlist",
        "allowlist": ["mcp__sequential-thinking__*"]
    }
}
```

**Usage in CLI:**
```bash
python cli_tool.py ask "Query" --approval-strategy allowlist --approval-allowlist "tool1" "tool2"
```

**Future Enhancement - Claude-Based Approval:**
A planned feature will use a separate Claude instance to make intelligent approval decisions based on:
- Project context from CLAUDE.md
- Recent conversation history
- Security considerations
- Task relevance

This will provide dynamic, context-aware approval decisions while maintaining security.

## Naming Conventions (PEP 8 Compliant)

### File Naming
- **Python modules**: `snake_case.py` (e.g., `claude_code_wrapper.py`)
- **Test files**: `test_*.py` located in `tests/` directory
- **Example files**: `*_example.py` located in `examples/` directory
- **Documentation**: `lowercase-with-hyphens.md` in `docs/` directory
- **Special files**: `README.md`, `CLAUDE.md` remain uppercase (standard convention)
- **Configuration**: `snake_case.json` or `descriptive-name.json`

### Directory Structure
- **Python packages**: `tests`, `examples` (lowercase, no underscores)
- **Documentation**: `docs`
- **Temporary**: `safe_to_delete`

### Code Conventions
- **Classes**: `PascalCase` (e.g., `ClaudeCodeWrapper`, `ClaudeCodeConfig`)
- **Functions/Methods**: `snake_case` (e.g., `ask_claude()`, `get_response()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private**: `_leading_underscore` (e.g., `_internal_method()`, `_private_var`)
- **Module-level private**: `_single_leading_underscore`
- **Name mangling**: `__double_leading_underscore` (use sparingly)