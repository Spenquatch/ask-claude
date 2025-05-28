# Development Guide

This guide covers the development setup, tools, and workflows for the Claude Code SDK Wrapper.

## Development Environment Setup

### Prerequisites

- **Python 3.10+** (required for MCP support and modern typing features)
- **Poetry** (recommended for dependency management and packaging)
- **pyenv** (recommended for Python version management)
- **Git** (for version control and pre-commit hooks)

### Initial Setup

1. **Clone and enter the repository:**
   ```bash
   git clone <repository-url>
   cd ask_claude
   ```

2. **Set Python version (if using pyenv):**
   ```bash
   pyenv local 3.10.17  # or your preferred 3.10+ version
   ```

3. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"  # Add to your shell profile
   ```

4. **Install dependencies with Poetry:**
   ```bash
   poetry install  # Installs all dependencies including dev tools
   ```

5. **Install pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

### Alternative: Traditional pip Installation
```bash
pip install -e .                    # Install package in editable mode
pip install pytest ruff mypy        # Install dev dependencies manually
pre-commit install
```

## Code Quality Tools

We maintain high code quality standards using automated tools:

### 🦀 **Ruff** - Linting and Formatting
- **Purpose**: Fast Python linter and formatter replacing black, flake8, isort, and more
- **Features**: Error detection, import sorting, code cleanup, code formatting
- **Configuration**: See `pyproject.toml` → `[tool.ruff]`
- **Usage**:
  ```bash
  # With Poetry (recommended)
  poetry run ruff check ask_claude/        # Check for issues
  poetry run ruff check ask_claude/ --fix  # Auto-fix issues
  poetry run ruff format ask_claude/       # Format code
  poetry run ruff check . && poetry run ruff format .  # Check and format entire project

  # Traditional
  ruff check ask_claude/
  ruff check ask_claude/ --fix
  ruff format ask_claude/
  ```

### 🔍 **mypy** - Static Type Checking
- **Purpose**: Ensure 100% type safety and catch type-related bugs
- **Configuration**: Strict settings in `pyproject.toml` → `[tool.mypy]`
- **Usage**:
  ```bash
  # With Poetry (recommended)
  poetry run mypy ask_claude/              # Type check main package
  poetry run mypy examples/                # Type check examples
  poetry run mypy .                        # Check entire project

  # Traditional
  mypy ask_claude/
  mypy examples/
  ```

### 🪝 **Pre-commit Hooks**
All tools run automatically on git commits via pre-commit hooks:

```bash
# With Poetry (recommended)
poetry run pre-commit run --all-files    # Manual run on all files
poetry run pre-commit run                # Manual run on staged files only
poetry run pre-commit autoupdate         # Update hook versions

# Traditional (if pre-commit installed globally)
pre-commit run --all-files
pre-commit run
pre-commit autoupdate
```

## Development Workflow

### **Quick Commands Summary**
```bash
# Development commands (use during development)
poetry run python -m ask_claude.cli ask "Your question"
poetry run python -m ask_claude.cli stream "Your query"
poetry run python -m ask_claude.cli session --interactive

# Production commands (after poetry install)
ask-claude ask "Your question"
ask-claude stream "Your query"
ask-claude session --interactive
ask-claude health
ask-claude benchmark
```

### 1. **Feature Development**
```bash
# Setup environment
poetry install

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... code, test, document ...

# Run quality checks
poetry run pre-commit run --all-files

# Commit (hooks run automatically)
git add .
git commit -m "feat: add your feature description"
```

### 2. **Testing Strategy**
```bash
# With Poetry (recommended)
poetry run python -m pytest                    # Run all tests
poetry run python -m pytest --cov=ask_claude   # Run with coverage
poetry run python -m pytest tests/test_wrapper.py  # Run specific test file
poetry run python -m pytest -v                 # Run with verbose output

# Traditional
python -m pytest
python -m pytest --cov=ask_claude
python -m pytest tests/test_wrapper.py
python -m pytest -v
```

### 3. **Type Safety Verification**
```bash
# With Poetry (recommended)
poetry run mypy ask_claude/                     # Check main package (must pass with 0 errors)
poetry run mypy ask_claude/wrapper.py           # Check specific module
poetry run mypy examples/ --ignore-missing-imports  # Check examples

# Traditional
mypy ask_claude/
mypy ask_claude/wrapper.py
mypy examples/ --ignore-missing-imports
```

## Code Architecture Guidelines

### **Package Structure**
```
ask_claude/
├── __init__.py              # Public API exports
├── wrapper.py               # Core ClaudeCodeWrapper class
├── cli.py                   # Command-line interface
├── session.py              # Session management
└── approval/               # MCP approval system
    ├── __init__.py
    ├── server.py           # Approval server
    └── strategies.py       # Approval strategies
```

### **Type Safety Standards**
- ✅ **100% mypy compliance** in main package
- ✅ All functions have return type annotations
- ✅ All parameters have type hints
- ✅ Use `TypeVar` for generic functions
- ✅ Proper `Optional[T]` for nullable values
- ✅ Forward references with `TYPE_CHECKING`

### **Error Handling Patterns**
```python
# Hierarchical exceptions
try:
    result = wrapper.run(query)
except ClaudeCodeTimeoutError:
    # Handle timeouts specifically
except ClaudeCodeProcessError:
    # Handle process issues
except ClaudeCodeError:
    # Handle any wrapper error
```

### **Configuration Management**
```python
# Prefer composition over inheritance
config = ClaudeCodeConfig(
    timeout=30.0,
    max_retries=3,
    cache_responses=True
)

# Support multiple config sources
wrapper = ClaudeCodeWrapper(config)  # Explicit config
wrapper = ClaudeCodeWrapper()        # Default config
```

## Project Standards

### **Naming Conventions (PEP 8)**
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### **Documentation Standards**
- **Docstrings**: Google style for all public functions
- **Type hints**: Required for all function signatures
- **Comments**: Explain "why", not "what"
- **Examples**: Include usage examples in docstrings

### **Testing Standards**
- **Coverage**: Aim for >90% code coverage
- **Naming**: `test_*` pattern for test functions
- **Structure**: One test file per module
- **Mocking**: Use `pytest-mock` for external dependencies

## Advanced Development

### **MCP Integration Development**
```python
# Test approval strategies
from ask_claude.approval import create_approval_strategy

strategy = create_approval_strategy("allowlist", {
    "allowlist": ["mcp__sequential-thinking__*"]
})

# Test with different configurations
config = {
    "mcp_auto_approval": {
        "enabled": True,
        "strategy": "patterns",
        "patterns": {"allow": ["safe_*"], "deny": ["dangerous_*"]}
    }
}
```

### **Performance Testing**
```bash
# With Poetry (recommended)
poetry run python -m ask_claude.cli benchmark    # Benchmark CLI
poetry run python examples/production_example.py # Custom performance testing

# After Poetry install
ask-claude benchmark                              # Direct command

# Traditional
python -m ask_claude.cli benchmark
python examples/production_example.py
```

### **Configuration Testing**
```bash
# With Poetry (recommended)
poetry run python examples/cache_configuration_example.py  # Test different configurations
poetry run python examples/session_manager_demo.py         # Test session management

# Traditional
python examples/cache_configuration_example.py
python examples/session_manager_demo.py
```

## Troubleshooting

### **Common Issues**

1. **mypy errors after changes**:
   ```bash
   # With Poetry
   poetry run mypy --cache-clear ask_claude/

   # Traditional
   mypy --cache-clear ask_claude/
   ```

2. **Pre-commit hook failures**:
   ```bash
   # With Poetry
   poetry run pre-commit run ruff --all-files
   poetry run pre-commit run ruff-format --all-files

   # Traditional
   pre-commit run ruff --all-files
   pre-commit run ruff-format --all-files
   ```

3. **Python version conflicts**:
   ```bash
   # Verify Python version
   python --version  # Should be 3.10+

   # Check pyenv
   pyenv versions
   pyenv local 3.10.17

   # Recreate Poetry environment
   poetry env remove python
   poetry install
   ```

4. **Import path issues**:
   ```bash
   # With Poetry (automatic)
   poetry install        # Installs package in editable mode

   # Traditional
   pip install -e .      # Install in development mode
   ```

5. **Poetry environment issues**:
   ```bash
   # Check Poetry environment
   poetry env info

   # Recreate environment
   poetry env remove python
   poetry install

   # Show available environments
   poetry env list
   ```

### **Getting Help**

- **Code Review**: Submit PRs for review before merging
- **Documentation**: Update docs with any API changes
- **Issues**: Report bugs with minimal reproduction cases
- **Questions**: Include context and error messages

## Release Process

### **Poetry-Based Release Workflow**

1. **Version Updates**:
   ```bash
   poetry version patch    # 0.1.0 -> 0.1.1
   poetry version minor    # 0.1.1 -> 0.2.0
   poetry version major    # 0.2.0 -> 1.0.0
   ```

2. **Quality Validation**:
   ```bash
   poetry run python -m pytest                    # Full test suite must pass
   poetry run pre-commit run --all-files          # All quality hooks must pass
   poetry run mypy ask_claude/                    # 100% type safety
   ```

3. **Build Package**:
   ```bash
   poetry build                                   # Creates dist/ with wheel and sdist
   ```

4. **Documentation**: Update any affected documentation

5. **Git Workflow**:
   ```bash
   git add pyproject.toml                         # Commit version bump
   git commit -m "bump: version $(poetry version -s)"
   git tag "v$(poetry version -s)"               # Create version tag
   git push origin main --tags                   # Push with tags
   ```

6. **Publish to PyPI** (Phase 4):
   ```bash
   poetry publish                                # Publish to PyPI
   poetry publish --repository testpypi          # Publish to Test PyPI first
   ```

This development guide ensures consistent, high-quality contributions to the Claude Code SDK Wrapper project.
