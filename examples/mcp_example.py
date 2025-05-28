#!/usr/bin/env python3
"""
Example: Using MCP (Model Context Protocol) with Claude Code Wrapper.

This example demonstrates how to configure and use MCP servers to extend
Claude's capabilities with external tools and data sources.
"""

import os
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import (
    ClaudeCodeWrapper, 
    ClaudeCodeConfig,
    MCPServerConfig,
    MCPConfig
)


def create_example_mcp_config():
    """Create an example MCP configuration file."""
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    # Restrict to current directory for safety
                    str(Path.cwd())
                ]
            }
        }
    }
    
    config_path = Path("example-mcp-config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created example MCP config at: {config_path}")
    return config_path


def example_basic_mcp_usage():
    """Basic example of using MCP with the wrapper."""
    print("\n=== Basic MCP Usage ===\n")
    
    # Create example config
    config_path = create_example_mcp_config()
    
    try:
        # Initialize wrapper with MCP config
        config = ClaudeCodeConfig(
            mcp_config_path=config_path,
            # Important: We need to allow the MCP tools
            allowed_tools=["mcp__filesystem__list_directory"]
        )
        wrapper = ClaudeCodeWrapper(config)
        
        # Check loaded servers
        servers = wrapper.get_mcp_servers()
        print(f"Loaded MCP servers: {list(servers.keys())}")
        
        # Get available tools
        tools = wrapper.get_mcp_tools("filesystem")
        print(f"Available filesystem tools: {tools[:3]}...")  # Show first 3
        
        # Make a query that uses MCP
        print("\nMaking query with MCP access...")
        response = wrapper.ask(
            "List the files in the current directory"
        )
        print(f"Response: {response.content[:200]}...")
        
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()
            print("\nCleaned up example config file")


def example_programmatic_mcp_config():
    """Example of creating MCP configuration programmatically."""
    print("\n=== Programmatic MCP Configuration ===\n")
    
    # Create wrapper
    wrapper = ClaudeCodeWrapper()
    
    # Create MCP server configs
    filesystem_server = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp"  # Restrict to /tmp for safety
        ]
    )
    
    # Create GitHub server (if token available)
    github_server = None
    if os.environ.get("GITHUB_TOKEN"):
        github_server = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]}
        )
    
    # Create MCP config
    servers = {"filesystem": filesystem_server}
    if github_server:
        servers["github"] = github_server
    
    mcp_config = wrapper.create_mcp_config(servers)
    
    # Save to file
    config_path = Path("programmatic-mcp-config.json")
    wrapper.save_mcp_config(mcp_config, config_path)
    print(f"Saved MCP config to: {config_path}")
    
    # Show the generated config
    with open(config_path) as f:
        print("\nGenerated MCP configuration:")
        print(json.dumps(json.load(f), indent=2))
    
    # Cleanup
    config_path.unlink()


def example_tool_permissions():
    """Example of managing MCP tool permissions."""
    print("\n=== MCP Tool Permissions ===\n")
    
    # Create wrapper with empty config
    wrapper = ClaudeCodeWrapper()
    
    # Simulate having MCP servers
    # In reality, these would be loaded from mcp_config
    print("Managing tool permissions...")
    
    # Allow all tools from a server
    wrapper.allow_mcp_tools("filesystem")
    print(f"After allowing all filesystem tools: {len(wrapper.config.allowed_tools)} tools")
    
    # Allow specific tools only
    wrapper.config.allowed_tools = []  # Reset
    wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])
    print(f"After allowing specific tools: {wrapper.config.allowed_tools}")
    
    # Add dangerous tools to disallowed list
    wrapper.config.disallowed_tools = [
        "mcp__filesystem__delete_file",
        "mcp__filesystem__execute_command"
    ]
    print(f"Disallowed tools: {wrapper.config.disallowed_tools}")


def example_security_patterns():
    """Example of security best practices with MCP."""
    print("\n=== MCP Security Patterns ===\n")
    
    # Pattern 1: Role-based access
    def create_wrapper_for_role(role: str) -> ClaudeCodeWrapper:
        """Create wrapper with role-specific MCP permissions."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)
        
        if role == "admin":
            # Full access
            wrapper.allow_mcp_tools("filesystem")
            wrapper.allow_mcp_tools("github")
            print(f"Admin role: Full MCP access")
        elif role == "developer":
            # Read-only access
            wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])
            wrapper.allow_mcp_tools("github", ["get_repository", "list_repositories"])
            print(f"Developer role: Read-only MCP access")
        else:
            # No MCP access
            print(f"Guest role: No MCP access")
        
        return wrapper
    
    # Create wrappers for different roles
    admin_wrapper = create_wrapper_for_role("admin")
    dev_wrapper = create_wrapper_for_role("developer")
    guest_wrapper = create_wrapper_for_role("guest")
    
    # Pattern 2: Environment-based configuration
    def get_mcp_config_for_env(env: str) -> Path:
        """Get appropriate MCP config for environment."""
        configs = {
            "development": "mcp-dev.json",
            "staging": "mcp-staging.json",
            "production": "mcp-prod.json"
        }
        config_file = configs.get(env, "mcp-dev.json")
        print(f"Using MCP config for {env}: {config_file}")
        return Path(config_file)
    
    # Example usage
    env = os.environ.get("ENVIRONMENT", "development")
    config_path = get_mcp_config_for_env(env)


def example_mcp_with_sessions():
    """Example of using MCP with sessions."""
    print("\n=== MCP with Sessions ===\n")
    
    # Create wrapper with simulated MCP
    wrapper = ClaudeCodeWrapper()
    
    # Enable filesystem tools
    wrapper.allow_mcp_tools("filesystem", ["read_file", "write_file"])
    
    print("Starting session with MCP tools...")
    with wrapper.session() as session:
        print(f"Session created with {len(wrapper.config.allowed_tools)} allowed tools")
        
        # Simulate file operations
        print("\n1. Reading configuration...")
        response1 = session.ask("Read the package.json file")
        
        print("\n2. Analyzing content...")
        response2 = session.ask("What version is specified in the file?")
        
        print("\n3. Making changes...")
        response3 = session.ask("Update the version to 2.0.0")
        
        print("\nSession completed with MCP tool access throughout")


def example_mcp_tool_discovery():
    """Example of discovering available MCP tools."""
    print("\n=== MCP Tool Discovery ===\n")
    
    wrapper = ClaudeCodeWrapper()
    
    # Simulate different server types
    server_types = ["filesystem", "github", "database", "custom_api"]
    
    for server in server_types:
        tools = wrapper.get_mcp_tools(server)
        if tools:
            print(f"\n{server} server tools:")
            for tool in tools[:3]:  # Show first 3
                print(f"  - {tool}")
            if len(tools) > 3:
                print(f"  ... and {len(tools) - 3} more")


def main():
    """Run all MCP examples."""
    print("=== Claude Code Wrapper - MCP Examples ===")
    
    # Check if npx is available (required for most MCP servers)
    import shutil
    if not shutil.which("npx"):
        print("\nWARNING: 'npx' not found. Install Node.js to use MCP servers.")
        print("Examples will show MCP patterns but may not execute actual servers.\n")
    
    # Run examples
    example_basic_mcp_usage()
    example_programmatic_mcp_config()
    example_tool_permissions()
    example_security_patterns()
    example_mcp_with_sessions()
    example_mcp_tool_discovery()
    
    print("\n=== MCP Integration Complete ===")
    print("\nKey Takeaways:")
    print("• MCP extends Claude with external tools and data sources")
    print("• Always explicitly allow required tools for security")
    print("• Use role-based and environment-based access control")
    print("• MCP servers must be trusted - they can access external resources")
    print("• Tool names follow pattern: mcp__<server>__<tool>")


if __name__ == "__main__":
    main()