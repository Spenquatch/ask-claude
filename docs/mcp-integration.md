# MCP (Model Context Protocol) Integration Guide

## Overview

Model Context Protocol (MCP) is an open protocol that enables Claude to securely access external tools and data sources. The Claude Code Wrapper provides comprehensive support for MCP integration, allowing you to extend Claude's capabilities with custom servers.

## What is MCP?

MCP allows you to:
- Connect specialized tools and services to Claude
- Access external data sources securely
- Extend Claude's capabilities with custom functionality
- Maintain security through explicit tool permissions

## Critical: Understanding Claude Code's MCP Architecture

Claude Code manages MCP servers at three scopes:
1. **User scope** (`-s user`) - Available across all projects for you
2. **Project scope** (`-s project`) - Shared via `.mcp.json` file in version control
3. **Local scope** (`-s local`, default) - Project-specific, private to you

### Two Approaches to MCP Integration

#### Approach 1: Pre-configured Servers (Recommended for Production)
If MCP servers are already configured in Claude Code (via `claude mcp add`), you don't need JSON config files:

```python
# Check what's already configured
# Run in terminal: claude mcp list

# Use pre-configured servers (no JSON needed!)
config = ClaudeCodeConfig(
    # No mcp_config_path - servers already in Claude Code
    allowed_tools=["mcp__deepwiki__deepwiki_fetch"],
    permission_prompt_tool="mcp__deepwiki__deepwiki_fetch",
    use_existing_mcp_servers=True  # Default
)
wrapper = ClaudeCodeWrapper(config)
```

#### Approach 2: Dynamic Configuration via JSON (Testing/Temporary)
For temporary or testing scenarios, use JSON config files:

```python
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-config.json"),
    allowed_tools=["mcp__deepwiki__deepwiki_fetch"]
)
```

## Configuration

### Basic MCP Configuration

Create an MCP configuration file (`mcp-config.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/allowed/directory"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_TOKEN": "your-github-token"
      }
    }
  }
}
```

### Using MCP with the Wrapper

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig
from pathlib import Path

# Method 1: Load MCP config from file
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-config.json")
)
wrapper = ClaudeCodeWrapper(config)

# Method 2: Create MCP config programmatically
from claude_code_wrapper import MCPServerConfig, MCPConfig

filesystem_server = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
)

github_server = MCPServerConfig(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": "ghp_xxxxx"}
)

mcp_config = wrapper.create_mcp_config({
    "filesystem": filesystem_server,
    "github": github_server
})

# Save for later use
wrapper.save_mcp_config(mcp_config, "my-mcp-config.json")
```

## Auto-Approval for MCP Tools (New Feature)

The Claude Code Wrapper now supports automatic approval of MCP tool usage, eliminating manual prompts during execution. This is especially useful for automation and CI/CD pipelines.

### Quick Start

```python
# Auto-approve all MCP tools
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)

# Auto-approve specific tools only
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": ["mcp__filesystem__read_file", "mcp__deepwiki__fetch"]
    }
)
```

### Approval Strategies

1. **`all`** - Approve all MCP tool requests automatically
2. **`none`** - Deny all MCP tool requests (useful for testing)
3. **`allowlist`** - Only approve tools in the allowlist
4. **`patterns`** - Approve/deny based on regex patterns

### CLI Usage

```bash
# Allow all tools
python cli_tool.py ask "Analyze this codebase" \
    --mcp-config mcp-servers.json \
    --approval-strategy all

# Allow specific tools
python cli_tool.py ask "Read the README" \
    --mcp-config mcp-servers.json \
    --approval-strategy allowlist \
    --approval-allowlist "mcp__filesystem__read_file"

# Pattern-based approval
python cli_tool.py ask "Query the database" \
    --mcp-config mcp-servers.json \
    --approval-strategy patterns \
    --approval-allow-patterns "mcp__.*__read.*" "mcp__.*__query.*" \
    --approval-deny-patterns "mcp__.*__write.*" "mcp__.*__delete.*"
```

## Tool Permissions

### Understanding MCP Tool Names

MCP tools follow the naming pattern: `mcp__<serverName>__<toolName>`

Examples:
- `mcp__filesystem__read_file`
- `mcp__github__get_repository`
- `mcp__database__query`

### Allowing MCP Tools

```python
# Allow all tools from a specific server
wrapper.allow_mcp_tools("filesystem")

# Allow specific tools only
wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])

# Get list of available tools
tools = wrapper.get_mcp_tools("filesystem")
print(f"Available filesystem tools: {tools}")

# Manually add to allowed tools
wrapper.config.allowed_tools.extend([
    "mcp__filesystem__read_file",
    "mcp__filesystem__write_file"
])
```

### Security Best Practices

```python
# Restrict to read-only operations
wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])

# Disallow dangerous operations
wrapper.config.disallowed_tools.extend([
    "mcp__filesystem__delete_file",
    "mcp__filesystem__execute_command"
])

# Use permission prompt for sensitive operations
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-config.json"),
    permission_prompt_tool="always"  # Always prompt for tool usage
)
```

## Common MCP Servers

### 1. Filesystem Server

Provides file system access within allowed directories.

```python
# Configuration
filesystem_config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/safe_directory"  # Restrict to specific directory
    ]
)

# Usage example
wrapper.allow_mcp_tools("filesystem", [
    "read_file",
    "write_file",
    "list_directory",
    "create_directory"
])

response = wrapper.ask(
    "Read the README.md file in the current directory",
    allowed_tools=["mcp__filesystem__read_file"]
)
```

### 2. GitHub Server

Integrates with GitHub repositories.

```python
# Configuration with authentication
github_config = MCPServerConfig(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN")}
)

# Usage example
wrapper.allow_mcp_tools("github", [
    "get_repository",
    "list_repositories",
    "get_file_contents"
])

response = wrapper.ask(
    "What are the open issues in anthropics/claude-code repo?",
    allowed_tools=["mcp__github__list_issues"]
)
```

### 3. Custom MCP Server

Create your own MCP server for specialized functionality.

```python
# Custom server configuration
custom_config = MCPServerConfig(
    name="myapp",
    command="/path/to/my-mcp-server",
    args=["--port", "8080"],
    env={"API_KEY": "secret"}
)

# Register custom tools
wrapper.allow_mcp_tools("myapp", [
    "process_data",
    "generate_report",
    "sync_database"
])
```

## Advanced Examples

### Dynamic Tool Selection

```python
class SmartMCPWrapper:
    """Wrapper that dynamically enables MCP tools based on query."""
    
    def __init__(self, mcp_config_path: str):
        self.wrapper = ClaudeCodeWrapper(
            ClaudeCodeConfig(mcp_config_path=Path(mcp_config_path))
        )
        
    def ask_with_context(self, query: str) -> ClaudeCodeResponse:
        """Enable appropriate MCP tools based on query content."""
        allowed_tools = []
        
        # Analyze query to determine needed tools
        if "file" in query.lower() or "read" in query.lower():
            allowed_tools.extend([
                "mcp__filesystem__read_file",
                "mcp__filesystem__list_directory"
            ])
        
        if "github" in query.lower() or "repository" in query.lower():
            allowed_tools.extend([
                "mcp__github__get_repository",
                "mcp__github__list_repositories"
            ])
        
        # Update allowed tools and execute
        self.wrapper.config.allowed_tools = allowed_tools
        return self.wrapper.ask(query)
```

### MCP with Sessions

```python
# Create session with MCP tools
with wrapper.session() as session:
    # Enable filesystem tools for this session
    wrapper.allow_mcp_tools("filesystem")
    
    # First query: read file
    response1 = session.ask("Read the config.json file")
    
    # Second query: modify based on content
    response2 = session.ask("Update the version number in the config")
    
    # Tools remain available throughout session
    response3 = session.ask("Save the changes")
```

### Conditional MCP Access

```python
def create_restricted_wrapper(user_role: str) -> ClaudeCodeWrapper:
    """Create wrapper with role-based MCP access."""
    config = ClaudeCodeConfig(
        mcp_config_path=Path("mcp-config.json")
    )
    wrapper = ClaudeCodeWrapper(config)
    
    if user_role == "admin":
        # Full access
        wrapper.allow_mcp_tools("filesystem")
        wrapper.allow_mcp_tools("github")
    elif user_role == "developer":
        # Read-only access
        wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])
        wrapper.allow_mcp_tools("github", ["get_repository", "get_file_contents"])
    else:
        # No MCP access
        pass
    
    return wrapper
```

## Security Considerations

### 1. **Trust Your Servers**
```python
# Only use trusted MCP servers
trusted_servers = ["filesystem", "github", "database"]
for server in wrapper.get_mcp_servers():
    if server not in trusted_servers:
        raise SecurityError(f"Untrusted MCP server: {server}")
```

### 2. **Limit Tool Access**
```python
# Whitelist approach - explicitly allow needed tools
wrapper.config.allowed_tools = [
    "mcp__filesystem__read_file",
    "mcp__github__get_repository"
]

# Blacklist dangerous tools
wrapper.config.disallowed_tools = [
    "mcp__filesystem__execute_command",
    "mcp__network__send_request"
]
```

### 3. **Environment Isolation**
```python
# Use separate MCP configs for different environments
env = os.environ.get("ENVIRONMENT", "development")
mcp_config_file = f"mcp-config-{env}.json"
config = ClaudeCodeConfig(mcp_config_path=Path(mcp_config_file))
```

### 4. **Audit Tool Usage**
```python
class AuditedWrapper(ClaudeCodeWrapper):
    """Wrapper that logs all MCP tool usage."""
    
    def run(self, query: str, **kwargs) -> ClaudeCodeResponse:
        # Log the query and allowed tools
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Allowed MCP tools: {self.config.allowed_tools}")
        
        response = super().run(query, **kwargs)
        
        # Log which tools were actually used (would need response parsing)
        self.logger.info(f"Response generated with MCP access")
        
        return response
```

## MCP Auto-Approval

MCP tools require approval before use by default. The auto-approval feature eliminates manual approval prompts by automatically approving or denying tools based on configured strategies.

### How Auto-Approval Works

The auto-approval system works by:
1. Creating a special MCP approval server that implements the `permissions__approve` tool
2. Configuring this server based on your chosen strategy (all, none, allowlist, or patterns)
3. Claude calls this tool automatically when it needs to use other MCP tools
4. The approval server instantly responds based on your configured rules

This happens transparently - you won't see approval prompts, and Claude can use tools seamlessly.

### Available Strategies

#### 1. **Allow All Strategy** (Development)
```python
# Allow all tools - useful for development
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)
```

#### 2. **Deny All Strategy** (Maximum Security)
```python
# Deny all tools - maximum security
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "none"
    }
)
```

#### 3. **Allowlist Strategy** (Recommended)
```python
# Only allow specific tools
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": [
            "mcp__filesystem__read_file",
            "mcp__filesystem__list_directory",
            "mcp__github__get_repository"
        ]
    }
)
```

#### 4. **Pattern-Based Strategy** (Flexible)
```python
# Use regex patterns for approval
config = ClaudeCodeConfig(
    mcp_auto_approval={
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": [
            r"mcp__.*__read.*",      # Allow all read operations
            r"mcp__.*__list.*",      # Allow all list operations
            r"mcp__.*__get.*"        # Allow all get operations
        ],
        "deny_patterns": [
            r"mcp__.*__delete.*",    # Deny all delete operations
            r"mcp__.*__admin.*"      # Deny all admin operations
        ]
    }
)
```

### CLI Usage

```bash
# Using allowlist strategy
claude ask "Read the config file" \
    --mcp-config mcp.json \
    --approval-strategy allowlist \
    --approval-allowlist mcp__filesystem__read_file

# Using pattern strategy
claude stream "Analyze the repository" \
    --mcp-config mcp.json \
    --approval-strategy patterns \
    --approval-allow-patterns "mcp__.*__read.*" "mcp__.*__list.*" \
    --approval-deny-patterns "mcp__.*__write.*"

# Using allow all strategy (development)
claude ask "Help me debug this" \
    --mcp-config mcp.json \
    --approval-strategy all
```

### Environment Variable Configuration

```bash
# Set approval strategy via environment
export APPROVAL_STRATEGY=allowlist
export APPROVAL_ALLOWLIST="mcp__filesystem__read_file,mcp__github__get_repository"

# The configurable_approval_server.py reads these automatically
```

### Security Considerations for Auto-Approval

```python
# Development: Allow all for convenience
dev_config = {
    "mcp_auto_approval": {
        "enabled": True,
        "strategy": "all"
    }
}

# Production: Strict allowlist
prod_config = {
    "mcp_auto_approval": {
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": [
            "mcp__filesystem__read_file",
            "mcp__database__query"  # Read-only operations
        ]
    }
}

# High-security: Pattern-based with explicit denies
secure_config = {
    "mcp_auto_approval": {
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": ["mcp__.*__read.*", "mcp__.*__list.*"],
        "deny_patterns": ["mcp__.*__write.*", "mcp__.*__delete.*", "mcp__.*__admin.*"]
    }
}
```

## Troubleshooting

### Common Issues

1. **MCP Server Not Found**
```python
try:
    config = ClaudeCodeConfig(mcp_config_path=Path("mcp-config.json"))
except ClaudeCodeConfigurationError as e:
    print(f"Failed to load MCP config: {e}")
```

2. **Tool Permission Denied**
```python
# Check if tool is allowed
tools = wrapper.get_mcp_tools()
print(f"Available tools: {tools}")
print(f"Allowed tools: {wrapper.config.allowed_tools}")
```

3. **Server Connection Failed**
```python
# Verify server command exists
import shutil
if not shutil.which("npx"):
    print("npx not found - install Node.js")
```

4. **Auto-Approval Not Working**
```python
# Check if approval server is configured
wrapper.list_available_mcp_servers()  # Should show 'approval-server'

# Verify approval configuration
print(wrapper.config.mcp_auto_approval)

# Test with verbose mode to see approval decisions
config.verbose = True
```

### Debug Mode

```python
# Enable verbose logging for MCP debugging
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-config.json"),
    verbose=True,
    log_level=logging.DEBUG
)
wrapper = ClaudeCodeWrapper(config)

# Test MCP functionality
servers = wrapper.get_mcp_servers()
print(f"Loaded MCP servers: {list(servers.keys())}")

for server_name in servers:
    tools = wrapper.get_mcp_tools(server_name)
    print(f"{server_name} tools: {tools}")
```

## Best Practices

1. **Principle of Least Privilege**: Only enable tools that are absolutely necessary
2. **Environment Separation**: Use different MCP configurations for dev/staging/prod
3. **Regular Audits**: Review MCP tool usage and permissions regularly
4. **Secure Storage**: Store sensitive configuration (API keys) in environment variables
5. **Version Control**: Track MCP configuration files in version control (without secrets)

## Future Enhancements

The MCP ecosystem is growing. Future possibilities include:
- Database connectors
- API integrations
- Cloud service adapters
- Custom business logic servers
- Real-time data feeds

Stay updated with the latest MCP servers and capabilities at the official MCP documentation.