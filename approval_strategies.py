#!/usr/bin/env python3
"""
Approval strategies for MCP tool auto-approval system.

This module provides different strategies for automatically approving or denying
MCP tool requests without manual intervention.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Pattern

logger = logging.getLogger(__name__)


class ApprovalStrategy(ABC):
    """Base class for approval strategies."""
    
    @abstractmethod
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        """
        Determine if a tool should be approved.
        
        Args:
            tool_name: The name of the MCP tool
            input_data: The input data for the tool
            
        Returns:
            True if the tool should be approved, False otherwise
        """
        pass
    
    @abstractmethod
    def get_denial_reason(self, tool_name: str) -> str:
        """Get the reason for denying a tool."""
        pass


class AllowAllStrategy(ApprovalStrategy):
    """Approves all tools - use with caution in production."""
    
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        logger.debug(f"AllowAllStrategy: Approving {tool_name}")
        return True
    
    def get_denial_reason(self, tool_name: str) -> str:
        return "This strategy approves all tools"


class DenyAllStrategy(ApprovalStrategy):
    """Denies all tools - useful for testing or high-security environments."""
    
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        logger.debug(f"DenyAllStrategy: Denying {tool_name}")
        return False
    
    def get_denial_reason(self, tool_name: str) -> str:
        return "All tools are denied by policy"


class AllowListStrategy(ApprovalStrategy):
    """Only approves tools explicitly listed in the allowlist."""
    
    def __init__(self, allowed_tools: List[str]):
        self.allowed_tools = set(allowed_tools)
        logger.info(f"AllowListStrategy initialized with {len(self.allowed_tools)} allowed tools")
    
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        approved = tool_name in self.allowed_tools
        logger.debug(f"AllowListStrategy: {tool_name} {'approved' if approved else 'denied'}")
        return approved
    
    def get_denial_reason(self, tool_name: str) -> str:
        return f"Tool '{tool_name}' is not in the allowlist"


class PatternStrategy(ApprovalStrategy):
    """Approves/denies based on regex patterns."""
    
    def __init__(self, allow_patterns: Optional[List[str]] = None, 
                 deny_patterns: Optional[List[str]] = None):
        self.allow_patterns: List[Pattern] = []
        self.deny_patterns: List[Pattern] = []
        
        if allow_patterns:
            self.allow_patterns = [re.compile(p) for p in allow_patterns]
            logger.info(f"PatternStrategy: {len(self.allow_patterns)} allow patterns")
        
        if deny_patterns:
            self.deny_patterns = [re.compile(p) for p in deny_patterns]
            logger.info(f"PatternStrategy: {len(self.deny_patterns)} deny patterns")
    
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        # Check deny patterns first (deny takes precedence)
        for pattern in self.deny_patterns:
            if pattern.search(tool_name):
                logger.debug(f"PatternStrategy: {tool_name} matches deny pattern {pattern.pattern}")
                return False
        
        # If no allow patterns specified, approve by default (unless denied)
        if not self.allow_patterns:
            logger.debug(f"PatternStrategy: {tool_name} approved (no allow patterns)")
            return True
        
        # Check allow patterns
        for pattern in self.allow_patterns:
            if pattern.search(tool_name):
                logger.debug(f"PatternStrategy: {tool_name} matches allow pattern {pattern.pattern}")
                return True
        
        logger.debug(f"PatternStrategy: {tool_name} denied (no matching allow pattern)")
        return False
    
    def get_denial_reason(self, tool_name: str) -> str:
        for pattern in self.deny_patterns:
            if pattern.search(tool_name):
                return f"Tool '{tool_name}' matches deny pattern '{pattern.pattern}'"
        return f"Tool '{tool_name}' does not match any allow patterns"


class CompositeStrategy(ApprovalStrategy):
    """Combines multiple strategies with AND/OR logic."""
    
    def __init__(self, strategies: List[ApprovalStrategy], require_all: bool = False):
        self.strategies = strategies
        self.require_all = require_all
        logger.info(f"CompositeStrategy: {len(strategies)} strategies, require_all={require_all}")
    
    def should_approve(self, tool_name: str, input_data: dict) -> bool:
        if self.require_all:
            # All strategies must approve (AND logic)
            for strategy in self.strategies:
                if not strategy.should_approve(tool_name, input_data):
                    return False
            return True
        else:
            # At least one strategy must approve (OR logic)
            for strategy in self.strategies:
                if strategy.should_approve(tool_name, input_data):
                    return True
            return False
    
    def get_denial_reason(self, tool_name: str) -> str:
        if self.require_all:
            reasons = []
            for strategy in self.strategies:
                if not strategy.should_approve(tool_name, {}):
                    reasons.append(strategy.get_denial_reason(tool_name))
            return " AND ".join(reasons)
        else:
            return "No strategy approved this tool"


def create_strategy(strategy_type: str, config: Dict) -> ApprovalStrategy:
    """
    Factory function to create approval strategies.
    
    Args:
        strategy_type: Type of strategy ('all', 'none', 'allowlist', 'patterns')
        config: Configuration dictionary for the strategy
        
    Returns:
        An ApprovalStrategy instance
    """
    if strategy_type == 'all':
        return AllowAllStrategy()
    elif strategy_type == 'none':
        return DenyAllStrategy()
    elif strategy_type == 'allowlist':
        allowed_tools = config.get('allowlist', [])
        return AllowListStrategy(allowed_tools)
    elif strategy_type == 'patterns':
        allow_patterns = config.get('allow_patterns', [])
        deny_patterns = config.get('deny_patterns', [])
        return PatternStrategy(allow_patterns, deny_patterns)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")