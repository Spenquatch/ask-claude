#!/usr/bin/env python3
"""
Comprehensive MCP (Model Context Protocol) Examples for Claude Code Wrapper

This example demonstrates:
1. Basic MCP setup and configuration
2. Programmatic MCP configuration
3. Auto-approval strategies
4. Tool permissions and security
5. Session management with MCP
6. Production best practices
"""

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ask_claude.wrapper import ClaudeCodeConfig, ClaudeCodeWrapper, MCPServerConfig


def example_basic_mcp_usage() -> None:
    """Basic example of using MCP with the wrapper."""
    print("\n=== Basic MCP Usage ===")

    # Use pre-configured MCP servers (recommended for production)
    config = ClaudeCodeConfig(
        allowed_tools=[
            "mcp__sequential-thinking__sequentialthinking",
            "mcp__deepwiki__deepwiki_fetch",
        ],
        verbose=False,
        timeout=120,
    )

    wrapper = ClaudeCodeWrapper(config)

    # List available MCP servers
    print("Checking available MCP servers...")
    server_list = wrapper.list_available_mcp_servers()
    print(f"Available servers: {server_list.content[:200]}...")

    # Test with sequential thinking
    print("\nTesting sequential-thinking tool:")
    response = wrapper.ask(
        "Use the sequential thinking tool to analyze: What are the steps to debug a Python script?"
    )
    print(f"Response preview: {response.content[:200]}...")


def example_programmatic_mcp_config() -> None:
    """Example of creating MCP configuration programmatically."""
    print("\n=== Programmatic MCP Configuration ===")

    # Create wrapper
    wrapper = ClaudeCodeWrapper()

    # Create MCP server configs
    filesystem_server = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(Path.cwd()),  # Restrict to current directory
        ],
    )

    # Create GitHub server (if token available)
    github_server = None
    if os.environ.get("GITHUB_TOKEN"):
        github_server = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
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
    if config_path.exists():
        config_path.unlink()


def example_auto_approval_allowlist() -> None:
    """Example using allowlist auto-approval strategy."""
    print("\n=== Auto-Approval: Allowlist Strategy ===")

    config = ClaudeCodeConfig(
        mcp_auto_approval={
            "enabled": True,
            "strategy": "allowlist",
            "allowlist": [
                "mcp__sequential-thinking__sequentialthinking",
                "mcp__filesystem__read_file",
                "mcp__filesystem__list_directory",
            ],
        }
    )

    wrapper = ClaudeCodeWrapper(config)

    print("Auto-approving only specific tools...")
    response = wrapper.ask(
        "Use the sequential thinking tool to plan a simple web application"
    )
    print(f"Response: {response.content[:200]}...")


def example_auto_approval_patterns() -> None:
    """Example using pattern-based auto-approval strategy."""
    print("\n=== Auto-Approval: Pattern Strategy ===")

    config = ClaudeCodeConfig(
        mcp_auto_approval={
            "enabled": True,
            "strategy": "patterns",
            "allow_patterns": [
                r"mcp__.*__read.*",  # Allow all read operations
                r"mcp__.*__list.*",  # Allow all list operations
                r"mcp__.*__get.*",  # Allow all get operations
            ],
            "deny_patterns": [
                r"mcp__.*__write.*",  # Deny all write operations
                r"mcp__.*__delete.*",  # Deny all delete operations
                r"mcp__.*__modify.*",  # Deny all modify operations
            ],
        }
    )

    wrapper = ClaudeCodeWrapper(config)

    print("Auto-approving based on patterns (read-only operations)...")
    response = wrapper.ask("Read the contents of README.md")
    print(f"Response: {response.content[:200]}...")


def example_tool_permissions() -> None:
    """Example of managing MCP tool permissions."""
    print("\n=== MCP Tool Permissions ===")

    wrapper = ClaudeCodeWrapper()

    print("Managing tool permissions...")

    # Allow all tools from a server
    wrapper.allow_mcp_tools("filesystem")
    print(
        f"After allowing all filesystem tools: {len(wrapper.config.allowed_tools)} tools"
    )

    # Allow specific tools only
    wrapper.config.allowed_tools = []  # Reset
    wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])
    print(f"After allowing specific tools: {wrapper.config.allowed_tools}")

    # Add dangerous tools to disallowed list
    wrapper.config.disallowed_tools = [
        "mcp__filesystem__delete_file",
        "mcp__filesystem__execute_command",
    ]
    print(f"Disallowed tools: {wrapper.config.disallowed_tools}")


def example_security_patterns() -> None:
    """Example of security best practices with MCP."""
    print("\n=== MCP Security Patterns ===")

    # Pattern 1: Role-based access
    def create_wrapper_for_role(role: str) -> ClaudeCodeWrapper:
        """Create wrapper with role-specific MCP permissions."""
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)

        if role == "admin":
            # Full access
            wrapper.allow_mcp_tools("filesystem")
            wrapper.allow_mcp_tools("github")
            print("Admin role: Full MCP access")
        elif role == "developer":
            # Read-only access
            wrapper.allow_mcp_tools("filesystem", ["read_file", "list_directory"])
            wrapper.allow_mcp_tools("github", ["get_repository", "list_repositories"])
            print("Developer role: Read-only MCP access")
        else:
            # No MCP access
            print("Guest role: No MCP access")

        return wrapper

    # Create wrappers for different roles
    admin_wrapper = create_wrapper_for_role("admin")
    dev_wrapper = create_wrapper_for_role("developer")
    guest_wrapper = create_wrapper_for_role("guest")

    # Pattern 2: Environment-based configuration
    def get_mcp_config_for_env(env: str) -> str:
        """Get appropriate MCP config for environment."""
        configs = {
            "development": "mcp-dev.json",
            "staging": "mcp-staging.json",
            "production": "mcp-prod.json",
        }
        config_file = configs.get(env, "mcp-dev.json")
        print(f"Using MCP config for {env}: {config_file}")
        return config_file

    # Example usage
    env = os.environ.get("ENVIRONMENT", "development")
    config_path = get_mcp_config_for_env(env)


def example_mcp_with_sessions() -> None:
    """Example of using MCP with sessions."""
    print("\n=== MCP with Sessions ===")

    config = ClaudeCodeConfig(
        allowed_tools=[
            "mcp__filesystem__read_file",
            "mcp__filesystem__write_file",
            "mcp__sequential-thinking__sequentialthinking",
        ]
    )
    wrapper = ClaudeCodeWrapper(config)

    print("Starting session with MCP tools...")
    with wrapper.session() as session:
        print(f"Session created with {len(wrapper.config.allowed_tools)} allowed tools")

        # Simulate file operations
        print("\n1. Reading configuration...")
        response1 = session.ask("Read the package.json file")

        print("\n2. Analyzing content...")
        response2 = session.ask("What version is specified in the file?")

        print("\n3. Using sequential thinking...")
        response3 = session.ask("Use sequential thinking to plan version update steps")

        print("\nSession completed with MCP tool access throughout")


def example_dynamic_approval() -> None:
    """Example of dynamically changing approval strategy."""
    print("\n=== Dynamic Approval Example ===")

    wrapper = ClaudeCodeWrapper()

    # Start with restrictive approval
    print("1. Restrictive mode - only read operations allowed:")
    response = wrapper.ask(
        "List files in the current directory",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "patterns",
            "allow_patterns": [r"mcp__.*__read.*", r"mcp__.*__list.*"],
        },
    )
    print(f"Response: {response.content[:200]}...")

    # Switch to more permissive approval
    print("\n2. Using allowlist for specific tools:")
    response = wrapper.ask(
        "Use sequential thinking to analyze the file structure",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "allowlist",
            "allowlist": ["mcp__sequential-thinking__sequentialthinking"],
        },
    )
    print(f"Response: {response.content[:200]}...")


def example_cli_usage() -> None:
    """Example showing CLI usage with MCP auto-approval."""
    print("\n=== CLI Usage Examples ===")

    print("1. Using allowlist strategy:")
    print(
        """
    python cli_tool.py ask "Use sequential thinking to plan a task" \\
        --approval-strategy allowlist \\
        --approval-allowlist "mcp__sequential-thinking__*" "mcp__filesystem__read*"
    """
    )

    print("\n2. Using pattern strategy:")
    print(
        """
    python cli_tool.py ask "Read project files" \\
        --approval-strategy patterns \\
        --approval-allow-patterns "mcp__.*__read.*" "mcp__.*__list.*" \\
        --approval-deny-patterns "mcp__.*__write.*"
    """
    )

    print("\n3. Interactive session with approval:")
    print(
        """
    python cli_tool.py session --interactive \\
        --approval-strategy allowlist \\
        --approval-allowlist "mcp__sequential-thinking__*"
    """
    )

    print("\n4. Streaming with auto-approval:")
    print(
        """
    python cli_tool.py stream "Analyze this codebase" \\
        --approval-strategy all
    """
    )


def example_production_best_practices() -> None:
    """Production deployment best practices."""
    print("\n=== Production Best Practices ===")

    print("1. Server Configuration:")
    print("   • Use 'claude mcp add' to configure servers globally/per-project")
    print("   • Scope: 'user' (global), 'project' (.mcp.json), or 'local'")
    print("   • Store sensitive configs in environment variables")

    print("\n2. Security:")
    print("   • Always specify allowed_tools explicitly")
    print("   • Use least-privilege principle")
    print("   • Regularly audit tool permissions")
    print("   • Use allowlist or pattern strategies, avoid 'all'")

    print("\n3. Configuration Management:")
    print("   • Environment-specific configs (dev/staging/prod)")
    print("   • Version control MCP configs (without secrets)")
    print("   • Use JSON schema validation")

    print("\n4. Monitoring:")
    print("   • Log MCP tool usage")
    print("   • Monitor approval patterns")
    print("   • Track performance metrics")

    # Example production config
    prod_config = {
        "mcp_auto_approval": {
            "enabled": True,
            "strategy": "allowlist",
            "allowlist": [
                "mcp__sequential-thinking__sequentialthinking",
                "mcp__filesystem__read_file",
                "mcp__filesystem__list_directory",
            ],
        },
        "allowed_tools": [
            "mcp__sequential-thinking__sequentialthinking",
            "mcp__filesystem__read_file",
        ],
        "timeout": 300,
        "max_retries": 3,
        "cache_responses": True,
    }

    print("\nExample production config:")
    print(json.dumps(prod_config, indent=2))


def main() -> None:
    """Run all MCP examples."""
    print("=== Claude Code Wrapper - Comprehensive MCP Examples ===")
    print("=" * 70)

    # Check if npx is available (required for most MCP servers)
    if not shutil.which("npx"):
        print("\nWARNING: 'npx' not found. Install Node.js to use MCP servers.")
        print("Examples will show MCP patterns but may not execute actual servers.\n")

    try:
        # Core examples
        example_basic_mcp_usage()
        example_programmatic_mcp_config()

        # Auto-approval examples
        example_auto_approval_allowlist()
        example_auto_approval_patterns()

        # Security and permissions
        example_tool_permissions()
        example_security_patterns()

        # Advanced usage
        example_mcp_with_sessions()
        example_dynamic_approval()

        # CLI and production
        example_cli_usage()
        example_production_best_practices()

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have:")
        print("• MCP servers configured ('claude mcp list' to check)")
        print("• Required MCP server dependencies installed")
        print("• Appropriate permissions for tool usage")

    print("\n" + "=" * 70)
    print("MCP Integration Complete")
    print("\nKey Takeaways:")
    print("• MCP extends Claude with external tools and data sources")
    print("• Use pre-configured servers for production reliability")
    print("• Always explicitly allow required tools for security")
    print("• Implement role-based and environment-based access control")
    print("• MCP servers must be trusted - they can access external resources")
    print("• Tool names follow pattern: mcp__<server>__<tool>")
    print("• Use auto-approval strategies to reduce manual intervention")


if __name__ == "__main__":
    main()
