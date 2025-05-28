# MCP Integration Guide

Model Context Protocol (MCP) allows Claude to securely access external tools and data sources. Ask Claude provides comprehensive MCP support with auto-approval features.

## Quick Setup

### 1. Check Existing MCP Servers
```bash
# See what's already configured in Claude Code
claude mcp list
```

### 2. Basic Usage with Pre-configured Servers
```python
from ask_claude.wrapper import ClaudeCodeWrapper

# Use existing MCP servers (recommended)
wrapper = ClaudeCodeWrapper()
response = wrapper.run("List files in the current directory")
```

### 3. Auto-Approval for Automation
```python
from ask_claude.wrapper import ClaudeCodeConfig, ClaudeCodeWrapper

# Auto-approve specific tools
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"]
    }
)

wrapper = ClaudeCodeWrapper(config)
response = wrapper.run("Read the README.md file")  # No manual approval needed
```

## Auto-Approval Strategies

### Allowlist Strategy (Safest)
Only approve specific tools by name:
```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": [
            "mcp__filesystem__read_file",
            "mcp__sequential-thinking__*"  # Wildcards supported
        ]
    }
)
```

### Pattern Strategy (Flexible)
Use regex patterns for approval/denial:
```python
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": [r"mcp__filesystem__(read|list).*"],
        "deny_patterns": [r".*delete.*", r".*write.*"]
    }
)
```

### All Strategy (Development Only)
```python
# ⚠️ Use with caution - approves ALL tools
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)
```

## CLI Auto-Approval

Use auto-approval from the command line:

```bash
# Approve specific tools
poetry run python -m ask_claude.cli ask "Read config.json" \
  --approval-strategy allowlist \
  --approval-allowlist "mcp__filesystem__read_file"

# Use patterns
poetry run python -m ask_claude.cli stream "Analyze files" \
  --approval-strategy patterns \
  --approval-allow-patterns "mcp__filesystem__read.*"

# Approve all (development)
poetry run python -m ask_claude.cli ask "Help me code" \
  --approval-strategy all
```

## Adding New MCP Servers

### Option 1: Using Claude Code CLI (Recommended)
```bash
# Add a server to user scope (available everywhere)
claude mcp add deepwiki --command "uvx deepwiki" -s user

# Add to project scope (shared with team)
claude mcp add filesystem --command "uvx mcp-server-filesystem" -s project

# List all servers
claude mcp list
```

### Option 2: JSON Configuration (Temporary)
Create `mcp-config.json`:
```json
{
  "servers": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/path/to/allowed/directory"]
    },
    "deepwiki": {
      "command": "uvx",
      "args": ["deepwiki"]
    }
  }
}
```

Then use it:
```python
config = ClaudeCodeConfig(mcp_config_path=Path("mcp-config.json"))
wrapper = ClaudeCodeWrapper(config)
```

## Security Best Practices

1. **Use allowlist strategy** for production applications
2. **Review tool permissions** before enabling auto-approval
3. **Use project scope** for team-shared MCP servers
4. **Test with manual approval** before enabling auto-approval
5. **Monitor tool usage** in production logs

## Common MCP Servers

| Server | Tools | Use Case |
|--------|-------|----------|
| `mcp-server-filesystem` | File operations | Read/write local files |
| `deepwiki` | Documentation fetch | Access project docs |
| `mcp-server-git` | Git operations | Repository management |
| `mcp-server-sequential-thinking` | Enhanced reasoning | Complex problem solving |

## Troubleshooting

### Tool Permission Denied
If you see permission errors:
1. Check `claude mcp list` to see available servers
2. Verify tool names match exactly (including prefixes like `mcp__`)
3. Add tools to allowlist or use `--approval-strategy all` for testing

### Server Not Found
```bash
# Check server status
claude mcp list

# Add missing server
claude mcp add servername --command "command" -s user
```

### Auto-Approval Not Working
1. Verify `enabled: true` in configuration
2. Check tool names match allowlist patterns exactly
3. Test with `--approval-strategy all` to isolate the issue

## Advanced Configuration

For complex scenarios, see:
- [Configuration Guide](configuration.md) - Detailed config options
- [Error Handling](api-exceptions.md) - Handle MCP-related errors
- [Examples](../examples/mcp_example.py) - Working code examples

## Next Steps

1. **Start simple**: Use existing MCP servers with manual approval
2. **Add auto-approval**: Use allowlist strategy for trusted tools
3. **Customize**: Add your own MCP servers as needed
4. **Secure**: Review and audit tool permissions regularly
