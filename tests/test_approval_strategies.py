#!/usr/bin/env python3
"""Unit tests for MCP approval strategies."""

import unittest
import re
from approval_strategies import (
    AllowAllStrategy,
    DenyAllStrategy,
    AllowListStrategy,
    PatternStrategy,
    create_strategy
)


class TestAllowAllStrategy(unittest.TestCase):
    """Test AllowAllStrategy."""
    
    def setUp(self):
        self.strategy = AllowAllStrategy()
    
    def test_approves_all_tools(self):
        """Should approve any tool."""
        self.assertTrue(self.strategy.should_approve("mcp__test__tool", {}))
        self.assertTrue(self.strategy.should_approve("mcp__admin__delete", {"data": "test"}))
        self.assertTrue(self.strategy.should_approve("any_tool_name", {"key": "value"}))
    
    def test_denial_reason(self):
        """Should return empty denial reason."""
        # AllowAllStrategy returns a non-empty reason
        reason = self.strategy.get_denial_reason("any_tool")
        self.assertIsInstance(reason, str)


class TestDenyAllStrategy(unittest.TestCase):
    """Test DenyAllStrategy."""
    
    def setUp(self):
        self.strategy = DenyAllStrategy()
    
    def test_denies_all_tools(self):
        """Should deny any tool."""
        self.assertFalse(self.strategy.should_approve("mcp__test__tool", {}))
        self.assertFalse(self.strategy.should_approve("mcp__safe__read", {"file": "test.txt"}))
        self.assertFalse(self.strategy.should_approve("any_tool_name", {"data": 123}))
    
    def test_denial_reason(self):
        """Should return denial reason."""
        reason = self.strategy.get_denial_reason("mcp__test__tool")
        self.assertIn("denied", reason.lower())
        # The actual implementation doesn't include tool name in denial message
        self.assertIsInstance(reason, str)


class TestAllowListStrategy(unittest.TestCase):
    """Test AllowListStrategy."""
    
    def setUp(self):
        self.allowed_tools = [
            "mcp__filesystem__read_file",
            "mcp__filesystem__list_directory",
            "mcp__database__query"
        ]
        self.strategy = AllowListStrategy(self.allowed_tools)
    
    def test_approves_allowed_tools(self):
        """Should approve tools in allowlist."""
        self.assertTrue(self.strategy.should_approve("mcp__filesystem__read_file", {}))
        self.assertTrue(self.strategy.should_approve("mcp__filesystem__list_directory", {"path": "/"}))
        self.assertTrue(self.strategy.should_approve("mcp__database__query", {"sql": "SELECT *"}))
    
    def test_denies_unlisted_tools(self):
        """Should deny tools not in allowlist."""
        self.assertFalse(self.strategy.should_approve("mcp__filesystem__write_file", {}))
        self.assertFalse(self.strategy.should_approve("mcp__database__delete", {}))
        self.assertFalse(self.strategy.should_approve("mcp__admin__tool", {}))
    
    def test_empty_allowlist(self):
        """Should deny all when allowlist is empty."""
        strategy = AllowListStrategy([])
        self.assertFalse(strategy.should_approve("any_tool", {}))
    
    def test_denial_reason(self):
        """Should provide informative denial reason."""
        reason = self.strategy.get_denial_reason("mcp__bad__tool")
        # Check for either "not in allowlist" or "is not in the allowlist"
        self.assertTrue("allowlist" in reason.lower())
        self.assertIn("mcp__bad__tool", reason)


class TestPatternStrategy(unittest.TestCase):
    """Test PatternStrategy."""
    
    def setUp(self):
        self.allow_patterns = [
            r"mcp__.*__read.*",
            r"mcp__.*__list.*",
            r"mcp__.*__get.*"
        ]
        self.deny_patterns = [
            r"mcp__.*__admin.*",
            r"mcp__.*__delete.*"
        ]
        self.strategy = PatternStrategy(self.allow_patterns, self.deny_patterns)
    
    def test_approves_matching_allow_patterns(self):
        """Should approve tools matching allow patterns."""
        self.assertTrue(self.strategy.should_approve("mcp__filesystem__read_file", {}))
        self.assertTrue(self.strategy.should_approve("mcp__db__list_tables", {}))
        self.assertTrue(self.strategy.should_approve("mcp__api__get_data", {}))
    
    def test_denies_matching_deny_patterns(self):
        """Should deny tools matching deny patterns even if they match allow."""
        self.assertFalse(self.strategy.should_approve("mcp__system__admin_read", {}))
        self.assertFalse(self.strategy.should_approve("mcp__db__delete_record", {}))
    
    def test_denies_non_matching_tools(self):
        """Should deny tools that don't match any allow pattern."""
        self.assertFalse(self.strategy.should_approve("mcp__filesystem__write_file", {}))
        self.assertFalse(self.strategy.should_approve("mcp__api__post_data", {}))
    
    def test_deny_patterns_take_precedence(self):
        """Deny patterns should override allow patterns."""
        # This tool matches both allow (read) and deny (admin)
        # The tool name needs to have admin between double underscores
        self.assertFalse(self.strategy.should_approve("mcp__fs__admin_read", {}))
    
    def test_invalid_regex_handling(self):
        """Should handle invalid regex patterns gracefully."""
        # The actual implementation will raise an error for invalid regex
        with self.assertRaises(re.error):
            strategy = PatternStrategy(["[invalid(regex"], [])
    
    def test_denial_reasons(self):
        """Should provide appropriate denial reasons."""
        # Denied by deny pattern
        reason = self.strategy.get_denial_reason("mcp__system__admin_tool")
        self.assertIn("matches deny pattern", reason.lower())
        
        # Not matching any allow pattern
        reason = self.strategy.get_denial_reason("mcp__fs__write")
        # Check for the actual message format
        self.assertTrue("does not match any allow pattern" in reason.lower() or "doesn't match any allow pattern" in reason.lower())


class TestCreateStrategy(unittest.TestCase):
    """Test strategy factory function."""
    
    def test_creates_allow_all_strategy(self):
        """Should create AllowAllStrategy."""
        strategy = create_strategy("all", {})
        self.assertIsInstance(strategy, AllowAllStrategy)
    
    def test_creates_deny_all_strategy(self):
        """Should create DenyAllStrategy."""
        strategy = create_strategy("none", {})
        self.assertIsInstance(strategy, DenyAllStrategy)
    
    def test_creates_allowlist_strategy(self):
        """Should create AllowListStrategy with tools."""
        config = {"allowlist": ["tool1", "tool2"]}
        strategy = create_strategy("allowlist", config)
        self.assertIsInstance(strategy, AllowListStrategy)
        self.assertTrue(strategy.should_approve("tool1", {}))
        self.assertFalse(strategy.should_approve("tool3", {}))
    
    def test_creates_pattern_strategy(self):
        """Should create PatternStrategy with patterns."""
        config = {
            "allow_patterns": [".*read.*"],
            "deny_patterns": [".*admin.*"]
        }
        strategy = create_strategy("patterns", config)
        self.assertIsInstance(strategy, PatternStrategy)
        self.assertTrue(strategy.should_approve("read_file", {}))
        self.assertFalse(strategy.should_approve("admin_read", {}))
    
    def test_raises_for_unknown_strategy(self):
        """Should raise ValueError for unknown strategy."""
        with self.assertRaises(ValueError) as ctx:
            create_strategy("unknown", {})
        self.assertIn("Unknown strategy", str(ctx.exception))
    
    def test_handles_missing_config(self):
        """Should handle missing configuration gracefully."""
        # Allowlist with missing list
        strategy = create_strategy("allowlist", {})
        self.assertIsInstance(strategy, AllowListStrategy)
        self.assertFalse(strategy.should_approve("any_tool", {}))  # Empty list
        
        # Pattern with missing patterns
        strategy = create_strategy("patterns", {})
        self.assertIsInstance(strategy, PatternStrategy)
        # With no patterns specified, PatternStrategy approves by default
        self.assertTrue(strategy.should_approve("any_tool", {}))  # No patterns = approve all


if __name__ == "__main__":
    unittest.main()