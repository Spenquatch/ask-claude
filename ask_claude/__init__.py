"""
Ask Claude - A production-ready Python wrapper for the Claude Code CLI.

This package provides a comprehensive interface to Claude Code with enterprise features
including error handling, retry logic, session management, and MCP integration.
"""

from .session import SessionManager
from .wrapper import (
    ClaudeCodeConfig,
    ClaudeCodeConfigurationError,
    # Exceptions
    ClaudeCodeError,
    ClaudeCodeProcessError,
    ClaudeCodeResponse,
    ClaudeCodeSession,
    ClaudeCodeTimeoutError,
    ClaudeCodeValidationError,
    ClaudeCodeWrapper,
    ask_claude,
    ask_claude_json,
    ask_claude_streaming,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "ClaudeCodeWrapper",
    "ClaudeCodeConfig",
    "ClaudeCodeResponse",
    "ClaudeCodeSession",
    "SessionManager",
    # Convenience functions
    "ask_claude",
    "ask_claude_json",
    "ask_claude_streaming",
    # Exceptions
    "ClaudeCodeError",
    "ClaudeCodeTimeoutError",
    "ClaudeCodeProcessError",
    "ClaudeCodeConfigurationError",
    "ClaudeCodeValidationError",
]
