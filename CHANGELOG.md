# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0rc1] - 2025-05-29

### Added
- Initial release candidate
- Core wrapper functionality with `ClaudeCodeWrapper` class
- Session management for multi-turn conversations
- MCP auto-approval system with multiple strategies
- CLI interface with commands: ask, stream, session, health, benchmark
- Comprehensive error handling with custom exception hierarchy
- Retry logic with exponential backoff
- Configuration management from files and environment variables
- Streaming response support
- Test coverage at 80%
- Full type safety with mypy
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Documentation for all major features

### Features
- Simple API with `ask_claude()` convenience function
- JSON response parsing with `ask_claude_json()`
- Real-time streaming with `ask_claude_streaming()`
- Session persistence and branching
- Detailed metrics and logging
- Cross-platform support (Linux, macOS, Windows)

[0.1.0rc1]: https://github.com/Spenquatch/ask-claude/releases/tag/v0.1.0rc1
