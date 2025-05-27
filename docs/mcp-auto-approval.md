# MCP Auto-Approval Guide

This guide explains how to use the MCP (Model Context Protocol) auto-approval feature in the Claude Code Wrapper to automatically approve or deny MCP tool requests without manual intervention.

## Overview

When Claude Code runs in non-interactive mode (e.g., through the wrapper), MCP tools normally require manual approval via prompts. The auto-approval system bypasses these prompts by programmatically approving or denying tool requests based on configurable strategies.

## How It Works

1. **Approval Server**: A dynamic approval server is spawned when auto-approval is enabled
2. **Approval Logic**: The server evaluates each tool request against the configured strategy
3. **Automatic Response**: Tools are automatically approved or denied without user interaction
4. **Cleanup**: The approval server is automatically stopped after the request completes

## Approval Strategies

### 1. Allowlist Strategy

Only explicitly listed tools are approved.

```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": [
            "mcp__sequential-thinking__sequentialthinking",
            "mcp__filesystem__read_file",
            "mcp__filesystem__list_directory"
        ]
    }
)
```

### 2. Pattern Strategy

Approve/deny based on regex patterns.

```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": [
            r"mcp__.*__read.*",      # Allow all read operations
            r"mcp__.*__list.*",      # Allow all list operations
        ],
        "deny_patterns": [
            r"mcp__.*__write.*",     # Deny all write operations
            r"mcp__.*__delete.*"     # Deny all delete operations
        ]
    }
)
```

### 3. All Strategy

Approves all tools (use with extreme caution!).

```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)
```

### 4. None Strategy

Denies all tools (useful for testing).

```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "none"
    }
)
```

## Usage Examples

### Python API

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Create config with auto-approval
config = ClaudeCodeConfig(
    mcp_config_path="mcp-servers.json",
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__sequential-thinking__*"]
    }
)

# Initialize wrapper
wrapper = ClaudeCodeWrapper(config)

# Use MCP tools without manual approval
response = wrapper.ask("Use sequential thinking to plan a project")
```

### CLI Usage

```bash
# Allowlist strategy
python cli_tool.py ask "Use sequential thinking to plan a task" \
    --approval-strategy allowlist \
    --approval-allowlist "mcp__sequential-thinking__*" "mcp__filesystem__read*"

# Pattern strategy
python cli_tool.py ask "Read project files" \
    --approval-strategy patterns \
    --approval-allow-patterns "mcp__.*__read.*" "mcp__.*__list.*" \
    --approval-deny-patterns "mcp__.*__write.*"

# All strategy (careful!)
python cli_tool.py ask "Use any tool you need" \
    --approval-strategy all
```

### Configuration File

Create a JSON config file:

```json
{
  "claude_binary": "claude",
  "mcp_config_path": "./mcp_config.json",
  "mcp_auto_approval": {
    "enabled": true,
    "strategy": "allowlist",
    "allowlist": [
      "mcp__sequential-thinking__sequentialthinking",
      "mcp__filesystem__read_file"
    ]
  }
}
```

Then use it:

```bash
python cli_tool.py --config auto_approval_config.json ask "Read the README file"
```

## Security Considerations

1. **Principle of Least Privilege**: Only approve tools that are absolutely necessary
2. **Use Allowlist When Possible**: The allowlist strategy is the most secure
3. **Avoid "all" Strategy**: Only use the "all" strategy in trusted, controlled environments
4. **Review Tool Names**: Understand what each MCP tool does before approving it
5. **Audit Logs**: The approval server logs all decisions for auditing

## Common Patterns

### Read-Only Access

Allow only read operations:

```python
"allow_patterns": [
    r"mcp__.*__read.*",
    r"mcp__.*__list.*",
    r"mcp__.*__get.*",
    r"mcp__.*__fetch.*",
    r"mcp__.*__query.*"
]
```

### Development vs Production

Development (more permissive):
```python
"strategy": "patterns",
"allow_patterns": ["mcp__.*"],
"deny_patterns": ["mcp__.*__delete.*", "mcp__.*__admin.*"]
```

Production (restrictive):
```python
"strategy": "allowlist",
"allowlist": [
    "mcp__filesystem__read_file",
    "mcp__database__query"
]
```

## Troubleshooting

### Approval Server Not Starting

1. Check that `approval_strategies.py` and `dynamic_approval_server.py` are available
2. Verify Python can spawn subprocesses
3. Check logs for startup errors

### Tools Not Being Approved

1. Verify the tool name matches exactly (case-sensitive)
2. Check regex patterns with a regex tester
3. Enable verbose logging to see approval decisions
4. Check the approval log file in temp directory

### Performance Issues

1. The approval server adds minimal overhead
2. Server startup/shutdown happens per request
3. Consider caching if making many sequential requests

## Future Enhancements

As documented in CLAUDE.md, a planned enhancement will use a separate Claude instance to make intelligent, context-aware approval decisions based on:

- Project context from CLAUDE.md
- Recent conversation history
- Security considerations
- Task relevance

This will provide dynamic approval decisions while maintaining security.