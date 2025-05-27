#!/usr/bin/env python3
"""Production-ready MCP integration guide for Claude Code Wrapper."""

from pathlib import Path
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig
import logging

class MCPEnabledWrapper:
    """Production wrapper with MCP diagnostics and fallback strategies."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.wrapper = None
        
    def initialize_with_diagnostics(self):
        """Initialize wrapper with MCP diagnostics."""
        # Step 1: Check available MCP servers
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)
        
        server_list = wrapper.list_available_mcp_servers()
        self.logger.info(f"Available MCP servers:\n{server_list.content}")
        
        # Step 2: Determine best configuration approach
        if "deepwiki" in server_list.content and "sequential-thinking" in server_list.content:
            self.logger.info("Pre-configured MCP servers found")
            return self._init_with_preconfig()
        else:
            self.logger.info("No pre-configured servers, using JSON config")
            return self._init_with_json()
    
    def _init_with_preconfig(self):
        """Initialize using pre-configured MCP servers."""
        config = ClaudeCodeConfig(
            # Try without JSON config first
            allowed_tools=[
                "mcp__sequential-thinking__sequentialthinking",
                "mcp__deepwiki__deepwiki_fetch"
            ],
            permission_prompt_tool="mcp__sequential-thinking__sequentialthinking",
            verbose=True,
            cache_responses=False
        )
        self.wrapper = ClaudeCodeWrapper(config)
        return self.wrapper
    
    def _init_with_json(self):
        """Initialize with JSON configuration as fallback."""
        # Look for available JSON configs
        json_configs = list(Path(".").glob("*-mcp-*.json"))
        
        if json_configs:
            config = ClaudeCodeConfig(
                mcp_config_path=json_configs[0],
                allowed_tools=["mcp__sequential-thinking__sequentialthinking"],
                permission_prompt_tool="mcp__sequential-thinking__sequentialthinking",
                verbose=True,
                cache_responses=False
            )
            self.wrapper = ClaudeCodeWrapper(config)
            self.logger.info(f"Using JSON config: {json_configs[0]}")
        else:
            self.logger.warning("No MCP configuration available")
            config = ClaudeCodeConfig(verbose=True)
            self.wrapper = ClaudeCodeWrapper(config)
        
        return self.wrapper
    
    def test_mcp_availability(self, tool_name: str) -> bool:
        """Test if a specific MCP tool is actually available."""
        test_query = f"Can you access the {tool_name} tool? Just say yes or no."
        response = self.wrapper.ask(test_query)
        return "yes" in response.content.lower()
    
    def ask_with_fallback(self, query: str) -> str:
        """Ask with automatic fallback if MCP tools aren't available."""
        # First attempt with MCP
        response = self.wrapper.ask(query)
        
        # Check if MCP tools were used
        if "tool is not available" in response.content or "don't have access" in response.content:
            self.logger.warning("MCP tools not accessible, trying with JSON config")
            
            # Reinitialize with JSON config
            self._init_with_json()
            response = self.wrapper.ask(query)
        
        return response.content

# Production usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize with diagnostics
    mcp_wrapper = MCPEnabledWrapper()
    mcp_wrapper.initialize_with_diagnostics()
    
    # Test MCP availability
    print("\nTesting MCP tool availability:")
    tools = [
        "mcp__sequential-thinking__sequentialthinking",
        "mcp__deepwiki__deepwiki_fetch"
    ]
    
    for tool in tools:
        available = mcp_wrapper.test_mcp_availability(tool)
        print(f"{tool}: {'✓ Available' if available else '✗ Not available'}")
    
    # Use with automatic fallback
    print("\nTesting with fallback:")
    response = mcp_wrapper.ask_with_fallback(
        "Use the sequential thinking tool to analyze how to set up a CI/CD pipeline"
    )
    print(f"Response: {response[:200]}...")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("1. Configure MCP servers at the appropriate scope:")
    print("   - User scope: claude mcp add -s user <server> <command>")
    print("   - Project scope: claude mcp add -s project <server> <command>") 
    print("2. For production, use environment-specific configurations")
    print("3. Always test MCP availability before relying on it")
    print("4. Implement fallback strategies for when MCP isn't available")