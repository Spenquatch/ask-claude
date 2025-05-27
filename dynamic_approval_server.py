#!/usr/bin/env python3
"""
Dynamic MCP approval server that can be spawned programmatically.

This server integrates with the approval strategies to provide automatic
approval/denial of MCP tool requests.
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import logging
from typing import Dict, Optional
from datetime import datetime

from approval_strategies import ApprovalStrategy, create_strategy

logger = logging.getLogger(__name__)


class DynamicApprovalServer:
    """
    Manages a dynamic approval server process.
    """
    
    def __init__(self, strategy: ApprovalStrategy, port: Optional[int] = None):
        self.strategy = strategy
        self.port = port
        self.server_process = None
        self.server_script_path = None
        self.log_file_path = None
    
    def _create_server_script(self) -> str:
        """Create a temporary Python script for the approval server."""
        script_content = '''#!/usr/bin/env python3
"""
Auto-generated MCP approval server.
"""

import sys
import os
import json
import asyncio
from datetime import datetime

# Add parent directory to path to import approval_strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if fastmcp is available
try:
    from mcp.server.fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    
    # Fallback implementation using standard library
    class FastMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}
            
        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func
            return decorator
            
        async def run(self):
            # Simple stdio server implementation
            while True:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                    if not line:
                        break
                    
                    request = json.loads(line)
                    method = request.get("method", "")
                    
                    if method == "tools/list":
                        response = {
                            "id": request.get("id"),
                            "result": {
                                "tools": [{"name": name} for name in self.tools.keys()]
                            }
                        }
                    elif method == "tools/call":
                        tool_name = request["params"]["name"]
                        if tool_name in self.tools:
                            result = await self.tools[tool_name](**request["params"]["arguments"])
                            response = {
                                "id": request.get("id"),
                                "result": result
                            }
                        else:
                            response = {
                                "id": request.get("id"),
                                "error": {"code": -32601, "message": "Method not found"}
                            }
                    else:
                        response = {
                            "id": request.get("id"),
                            "error": {"code": -32601, "message": "Method not found"}
                        }
                    
                    sys.stdout.write(json.dumps(response) + "\\n")
                    sys.stdout.flush()
                    
                except Exception as e:
                    sys.stderr.write(f"Error: {e}\\n")
                    sys.stderr.flush()

# Load strategy configuration
STRATEGY_CONFIG = json.loads('{strategy_config}')

# Create strategy
from approval_strategies import create_strategy
strategy = create_strategy(STRATEGY_CONFIG['type'], STRATEGY_CONFIG)

# Create MCP server
mcp = FastMCP("approval-server")

def log_to_file(message):
    """Log approval decisions."""
    log_path = "{log_path}"
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\\n")

@mcp.tool()
async def permissions__approve(tool_name: str, input: dict, reason: str = "") -> dict:
    """
    Approve or deny permission requests from Claude.
    
    Returns dict with behavior:"allow"/"deny"
    """
    # Use the strategy to make the decision
    approved = strategy.should_approve(tool_name, input)
    
    # Log the decision
    log_to_file(f"Tool: {tool_name}, Approved: {approved}, Input: {json.dumps(input)}")
    
    if approved:
        return {
            "behavior": "allow",
            "updatedInput": input
        }
    else:
        return {
            "behavior": "deny",
            "message": strategy.get_denial_reason(tool_name)
        }

if __name__ == "__main__":
    # Run the server
    if HAS_FASTMCP:
        asyncio.run(mcp.run())
    else:
        # Fallback stdio server
        try:
            asyncio.run(mcp.run())
        except KeyboardInterrupt:
            pass
'''
        
        # Create strategy configuration
        strategy_config = {
            'type': self._get_strategy_type()
        }
        
        # Handle allowlist (convert set to list if needed)
        if hasattr(self.strategy, 'allowed_tools'):
            allowed_tools = self.strategy.allowed_tools
            if isinstance(allowed_tools, set):
                allowed_tools = list(allowed_tools)
            strategy_config['allowlist'] = allowed_tools
        
        # Handle patterns
        if hasattr(self.strategy, 'allow_patterns'):
            strategy_config['allow_patterns'] = [p.pattern for p in self.strategy.allow_patterns]
        if hasattr(self.strategy, 'deny_patterns'):
            strategy_config['deny_patterns'] = [p.pattern for p in self.strategy.deny_patterns]
        
        # Create log file
        log_dir = tempfile.gettempdir()
        self.log_file_path = os.path.join(log_dir, f"approval_server_{os.getpid()}.log")
        
        # Format the script with actual values
        script_content = script_content.replace('{strategy_config}', json.dumps(strategy_config))
        script_content = script_content.replace('{log_path}', self.log_file_path)
        
        # Write to temporary file
        fd, self.server_script_path = tempfile.mkstemp(suffix='.py', prefix='approval_server_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(self.server_script_path, 0o755)
        
        return self.server_script_path
    
    def _get_strategy_type(self) -> str:
        """Get the strategy type name."""
        class_name = self.strategy.__class__.__name__
        if 'AllowAll' in class_name:
            return 'all'
        elif 'DenyAll' in class_name:
            return 'none'
        elif 'AllowList' in class_name:
            return 'allowlist'
        elif 'Pattern' in class_name:
            return 'patterns'
        else:
            return 'unknown'
    
    def start(self) -> Dict[str, any]:
        """
        Start the approval server.
        
        Returns:
            Dictionary with MCP configuration for this server
        """
        if self.server_process:
            raise RuntimeError("Server already running")
        
        # Create the server script
        script_path = self._create_server_script()
        
        # Start the server process
        self.server_process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Started approval server with PID {self.server_process.pid}")
        
        # Give the server a moment to start
        import time
        time.sleep(0.5)
        
        # Check if process is still running
        if self.server_process.poll() is not None:
            stderr = self.server_process.stderr.read() if self.server_process.stderr else ""
            raise RuntimeError(f"Approval server failed to start: {stderr}")
        
        # Return MCP configuration for this server
        return {
            "approval-server": {
                "command": sys.executable,
                "args": [script_path]
            }
        }
    
    def stop(self):
        """Stop the approval server."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
            logger.info("Stopped approval server")
        
        # Clean up temporary files
        if self.server_script_path and os.path.exists(self.server_script_path):
            os.unlink(self.server_script_path)
            self.server_script_path = None
        
        if self.log_file_path and os.path.exists(self.log_file_path):
            # Optionally read and log the approval decisions
            try:
                with open(self.log_file_path, 'r') as f:
                    logger.debug(f"Approval decisions:\n{f.read()}")
            except Exception:
                pass
            os.unlink(self.log_file_path)
            self.log_file_path = None
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on exit."""
        self.stop()