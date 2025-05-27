#!/usr/bin/env python3
"""Comprehensive MCP integration example showing best practices."""

from pathlib import Path
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

print("🔍 MCP Integration - Best Practices Example")
print("=" * 60)

# First, let's check what MCP servers are already configured in Claude Code
print("\n1. Checking Claude Code's configured MCP servers:")
config = ClaudeCodeConfig(
    verbose=False,
    cache_responses=False
)
wrapper = ClaudeCodeWrapper(config)

# List available MCP servers
server_list = wrapper.list_available_mcp_servers()
print("Available MCP servers:")
print(server_list.content)

# Approach 1: Using pre-configured MCP servers (recommended for production)
print("\n" + "=" * 60)
print("2. Using pre-configured MCP servers (no JSON config needed):")
print("   This approach uses servers already added via 'claude mcp add'")

# Since servers are already configured, we just need to specify allowed tools
config_preconfig = ClaudeCodeConfig(
    # No mcp_config_path needed - servers are already in Claude Code
    allowed_tools=[
        "mcp__sequential-thinking__sequentialthinking",
        "mcp__deepwiki__deepwiki_fetch"
    ],
    permission_prompt_tool="mcp__sequential-thinking__sequentialthinking",
    use_existing_mcp_servers=True,  # This is the default
    verbose=False,
    timeout=120,
    cache_responses=False
)

wrapper_preconfig = ClaudeCodeWrapper(config_preconfig)

# Test with pre-configured servers
print("\nTesting sequential-thinking (pre-configured):")
response = wrapper_preconfig.ask(
    "Use the mcp__sequential-thinking__sequentialthinking tool to analyze: What are the steps to debug a Python script?"
)
print(f"Response preview: {response.content[:200]}...")

# Approach 2: Using JSON config file (for dynamic/temporary configurations)
print("\n" + "=" * 60)
print("3. Using JSON config file (for temporary or project-specific servers):")

# Check if JSON config exists
if Path("sequential-thinking-mcp-production.json").exists():
    config_json = ClaudeCodeConfig(
        mcp_config_path=Path("sequential-thinking-mcp-production.json"),
        allowed_tools=["mcp__sequential-thinking__sequentialthinking"],
        permission_prompt_tool="mcp__sequential-thinking__sequentialthinking",
        use_existing_mcp_servers=False,  # Override to use JSON config
        verbose=False,
        timeout=120,
        cache_responses=False
    )
    
    wrapper_json = ClaudeCodeWrapper(config_json)
    print("Using JSON config: sequential-thinking-mcp-production.json")
else:
    print("JSON config file not found - skipping this approach")

# Best Practice Summary
print("\n" + "=" * 60)
print("Best Practices Summary:")
print("1. For production: Use 'claude mcp add' to configure servers globally/per-project")
print("2. For testing: Use JSON config files for temporary configurations")
print("3. Always specify allowed_tools explicitly for security")
print("4. Use permission_prompt_tool for non-interactive execution")
print("5. Consider scope: 'user' (global), 'project' (.mcp.json), or 'local'")

# Show how to check if a specific tool is available
print("\n" + "=" * 60)
print("4. Checking tool availability:")

# This would work if we had access to the actual tool listing
# For now, we check based on our allowed tools
allowed = config_preconfig.allowed_tools
print(f"Allowed tools: {allowed}")
print(f"Sequential thinking available: {'mcp__sequential-thinking__sequentialthinking' in allowed}")
print(f"DeepWiki available: {'mcp__deepwiki__deepwiki_fetch' in allowed}")