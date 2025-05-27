#!/usr/bin/env python3
"""
MCP Auto-Approval Example

This example demonstrates how to use the MCP auto-approval feature
to automatically approve or deny MCP tool requests without manual intervention.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig


def example_allowlist_strategy():
    """Example using allowlist strategy - only specific tools are approved."""
    print("=== Allowlist Strategy Example ===\n")
    
    config = ClaudeCodeConfig(
        mcp_config_path="mcp-servers.json",  # Your MCP servers config
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
    
    wrapper = ClaudeCodeWrapper(config)
    
    # This will auto-approve the sequential thinking tool
    response = wrapper.ask(
        "Use the sequential thinking tool to plan a simple web application"
    )
    print(f"Response: {response.content[:200]}...\n")


def example_pattern_strategy():
    """Example using pattern strategy - approve/deny based on regex patterns."""
    print("=== Pattern Strategy Example ===\n")
    
    config = ClaudeCodeConfig(
        mcp_config_path="mcp-servers.json",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "patterns",
            "allow_patterns": [
                r"mcp__.*__read.*",      # Allow all read operations
                r"mcp__.*__list.*",      # Allow all list operations
                r"mcp__.*__get.*"        # Allow all get operations
            ],
            "deny_patterns": [
                r"mcp__.*__write.*",     # Deny all write operations
                r"mcp__.*__delete.*",    # Deny all delete operations
                r"mcp__.*__modify.*"     # Deny all modify operations
            ]
        }
    )
    
    wrapper = ClaudeCodeWrapper(config)
    
    # This will auto-approve read operations but deny write operations
    response = wrapper.ask(
        "Read the contents of README.md (this should be approved)"
    )
    print(f"Response: {response.content[:200]}...\n")


def example_all_strategy():
    """Example using 'all' strategy - approves everything (use with caution!)."""
    print("=== All Strategy Example (CAUTION) ===\n")
    
    config = ClaudeCodeConfig(
        mcp_config_path="mcp-servers.json",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "all"  # Approves all tools - be careful!
        }
    )
    
    wrapper = ClaudeCodeWrapper(config)
    
    response = wrapper.ask(
        "Use any available MCP tools to help me understand the project structure"
    )
    print(f"Response: {response.content[:200]}...\n")


def example_cli_usage():
    """Example showing CLI usage with auto-approval."""
    print("=== CLI Usage Examples ===\n")
    
    print("1. Using allowlist strategy:")
    print("""
    python cli_tool.py ask "Use sequential thinking to plan a task" \\
        --approval-strategy allowlist \\
        --approval-allowlist "mcp__sequential-thinking__*" "mcp__filesystem__read*"
    """)
    
    print("\n2. Using pattern strategy:")
    print("""
    python cli_tool.py ask "Read project files" \\
        --approval-strategy patterns \\
        --approval-allow-patterns "mcp__.*__read.*" "mcp__.*__list.*" \\
        --approval-deny-patterns "mcp__.*__write.*"
    """)
    
    print("\n3. Interactive session with approval:")
    print("""
    python cli_tool.py session --interactive \\
        --approval-strategy allowlist \\
        --approval-allowlist "mcp__sequential-thinking__*"
    """)


def example_dynamic_approval():
    """Example of dynamically changing approval strategy."""
    print("=== Dynamic Approval Example ===\n")
    
    wrapper = ClaudeCodeWrapper()
    
    # Start with restrictive approval
    print("1. Restrictive mode - only read operations allowed:")
    response = wrapper.ask(
        "List files in the current directory",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "patterns",
            "allow_patterns": [r"mcp__.*__read.*", r"mcp__.*__list.*"]
        }
    )
    print(f"Response: {response.content[:200]}...\n")
    
    # Switch to more permissive approval
    print("2. Permissive mode - all operations allowed:")
    response = wrapper.ask(
        "Create a new file called test.txt",
        mcp_auto_approval={
            "enabled": True,
            "strategy": "all"
        }
    )
    print(f"Response: {response.content[:200]}...\n")


if __name__ == "__main__":
    print("MCP Auto-Approval Examples\n")
    print("=" * 50 + "\n")
    
    # Note: These examples require MCP servers to be configured
    # Make sure you have the appropriate MCP servers set up
    
    try:
        # Uncomment the examples you want to run:
        
        # example_allowlist_strategy()
        # example_pattern_strategy()
        # example_all_strategy()
        example_cli_usage()
        # example_dynamic_approval()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. MCP servers configured (mcp-servers.json)")
        print("2. The approval_strategies.py module available")
        print("3. Any required MCP server dependencies installed")