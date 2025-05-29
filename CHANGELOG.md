# Changelog

All notable changes to Ask Claude - Claude Code SDK Wrapper will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Ask Claude - Claude Code SDK Wrapper
- Core `ClaudeCodeWrapper` class for interacting with Claude Code CLI
- Session management for multi-turn conversations
- MCP (Model Context Protocol) integration with auto-approval
- Comprehensive error handling with custom exception hierarchy
- CLI interface with commands: ask, stream, session, health, benchmark
- Convenience functions: `ask_claude()`, `ask_claude_json()`, `ask_claude_streaming()`
- Configuration management with JSON file and environment variable support
- Retry logic with exponential backoff and circuit breaker pattern
- Response caching for performance optimization
- Streaming response support
- Type safety with mypy strict mode
- Pre-commit hooks for code quality
- Poetry-based dependency management
- Comprehensive test suite with pytest
- Documentation with usage examples
- Multi-environment testing with tox

### Security
- MCP tool approval strategies (allowlist, patterns, all, none)
- Secure handling of API credentials
- Input validation and sanitization

## [0.1.0] - TBD

### Notes
- First public release
- Python 3.10+ required
- Claude Code CLI must be installed separately

[Unreleased]: https://github.com/spenquatch/ask-claude/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/spenquatch/ask-claude/releases/tag/v0.1.0
