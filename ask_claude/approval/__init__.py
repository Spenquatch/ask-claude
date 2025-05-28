"""
Approval system for MCP (Model Context Protocol) tool requests.

This module provides strategies for automatically approving or denying
MCP tool requests without manual intervention.
"""

from .strategies import (
    AllowAllStrategy,
    AllowListStrategy,
    ApprovalStrategy,
    CompositeStrategy,
    DenyAllStrategy,
    PatternStrategy,
    create_approval_strategy,
)

__all__ = [
    # Base class
    "ApprovalStrategy",
    # Strategy implementations
    "AllowAllStrategy",
    "DenyAllStrategy",
    "AllowListStrategy",
    "PatternStrategy",
    "CompositeStrategy",
    # Factory function
    "create_approval_strategy",
]
