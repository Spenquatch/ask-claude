# Contributing to Ask Claude

Thank you for your interest in contributing to Ask Claude - Claude Code SDK Wrapper! We welcome contributions from the community.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install with Poetry: `poetry install`
4. Create a branch: `git checkout -b feature/your-feature-name`
5. Make your changes
6. Run tests: `poetry run pytest`
7. Submit a pull request

## Development Setup

For detailed development instructions, see our [Development Guide](docs/development.md).

### Prerequisites
- Python 3.10+
- Poetry for dependency management
- Claude Code CLI installed
- Git for version control

### Key Commands
```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Format code
poetry run ruff format .

# Type checking
poetry run mypy ask_claude/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## Code Standards

We maintain high code quality standards:
- ✅ **100% type safety** with mypy
- ✅ **Code formatting** with Ruff
- ✅ **Comprehensive tests** with pytest
- ✅ **Pre-commit hooks** for consistency

All code must pass these checks before merging.

## Testing

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=ask_claude --cov-report=html

# Run specific test file
poetry run pytest tests/test_wrapper.py

# Run tests in verbose mode
poetry run pytest -v
```

### Writing Tests
- Add tests for all new features
- Maintain >80% code coverage
- Use descriptive test names
- Mock external dependencies (Claude CLI)

## Pull Request Process

1. **Update Documentation** - Update relevant docs for your changes
2. **Add Tests** - Include tests for new functionality
3. **Update CHANGELOG.md** - Add your changes under "Unreleased"
4. **Pass All Checks** - Ensure all tests and linting pass
5. **Request Review** - Tag maintainers for review

### PR Title Format
Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Project Structure

```
ask_claude/
├── ask_claude/          # Main package
│   ├── wrapper.py       # Core wrapper functionality
│   ├── cli.py          # CLI interface
│   ├── session.py      # Session management
│   └── approval/       # MCP approval strategies
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Usage examples
└── pyproject.toml      # Project configuration
```

## Reporting Issues

### Bug Reports
Include:
- Python version
- Ask Claude version
- Claude Code CLI version
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests
- Describe the use case
- Provide examples of how it would work
- Explain why it's valuable

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior
- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments
- Publishing others' private information

## Getting Help

- 📖 [Documentation](docs/README.md)
- 💬 [GitHub Discussions](https://github.com/yourusername/ask-claude/discussions)
- 🐛 [Issue Tracker](https://github.com/yourusername/ask-claude/issues)

## Recognition

Contributors will be recognized in:
- [CHANGELOG.md](CHANGELOG.md) for their contributions
- GitHub contributors page
- Release notes when applicable

Thank you for contributing to Ask Claude! 🎉
