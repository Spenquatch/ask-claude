# Exception Handling API

Error types and exception handling for Ask Claude - Claude Code SDK Wrapper.

## Exception Hierarchy

All wrapper exceptions inherit from `ClaudeCodeError`.

```python
from ask_claude.wrapper import (
    ClaudeCodeError,               # Base exception
    ClaudeCodeConfigurationError,  # Configuration issues
    ClaudeCodeProcessError,        # CLI process errors
    ClaudeCodeTimeoutError,        # Timeout errors
    ClaudeCodeValidationError,     # Input validation errors
)
```

## Exception Types

### ClaudeCodeError

Base exception for all wrapper errors.

```python
try:
    response = wrapper.run("query")
except ClaudeCodeError as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
```

### ClaudeCodeConfigurationError

Raised when configuration is invalid.

**Common causes:**
- Invalid configuration parameters
- Missing required configuration
- Conflicting settings

```python
try:
    config = ClaudeCodeConfig(timeout=-1)  # Invalid
except ClaudeCodeConfigurationError as e:
    print(f"Config error: {e}")
```

### ClaudeCodeProcessError

Raised when the Claude CLI process fails.

**Properties:**
- `returncode`: CLI exit code
- `stderr`: Error output from CLI

```python
try:
    response = wrapper.run("query")
except ClaudeCodeProcessError as e:
    print(f"CLI failed with code {e.returncode}")
    print(f"Error: {e.stderr}")
```

### ClaudeCodeTimeoutError

Raised when requests exceed timeout limits.

```python
try:
    response = wrapper.run("complex query", timeout=5.0)
except ClaudeCodeTimeoutError:
    print("Query took too long - try simplifying it")
```

### ClaudeCodeValidationError

Raised when input parameters are invalid.

```python
try:
    response = wrapper.run("")  # Empty query
except ClaudeCodeValidationError as e:
    print(f"Invalid input: {e}")
```

## Error Handling Patterns

### Basic Error Handling

```python
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeError

wrapper = ClaudeCodeWrapper()

try:
    response = wrapper.run("What is Python?")
    print(response.content)
except ClaudeCodeError as e:
    print(f"Error: {e}")
```

### Specific Error Handling

```python
from ask_claude.wrapper import (
    ClaudeCodeWrapper,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError,
    ClaudeCodeConfigurationError
)

try:
    response = wrapper.run("complex query", timeout=30.0)

except ClaudeCodeTimeoutError:
    print("Query timed out - try a simpler question")

except ClaudeCodeProcessError as e:
    if e.returncode == 1:
        print("Claude CLI error - check your API key")
    else:
        print(f"Unexpected CLI error: {e.stderr}")

except ClaudeCodeConfigurationError:
    print("Configuration issue - check your settings")
```

### Retry Logic

```python
import time
from ask_claude.wrapper import ClaudeCodeWrapper, ClaudeCodeTimeoutError

def robust_query(query: str, max_retries: int = 3):
    wrapper = ClaudeCodeWrapper()

    for attempt in range(max_retries):
        try:
            return wrapper.run(query, timeout=60.0)

        except ClaudeCodeTimeoutError:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### Response Error Checking

Even successful responses can contain errors from Claude:

```python
response = wrapper.run("query")

if response.is_error:
    print(f"Claude error: {response.error_type}")
    # Handle Claude-level errors
else:
    print(response.content)
```

## Best Practices

1. **Always catch ClaudeCodeError** as a base case
2. **Handle timeouts gracefully** - they're common with long queries
3. **Check response.is_error** even for successful API calls
4. **Log errors** for debugging in production
5. **Implement retry logic** for transient failures
6. **Validate inputs** before sending to avoid validation errors

## Common Error Scenarios

### API Key Issues
```python
# Usually raises ClaudeCodeProcessError with returncode=1
```

### Network Problems
```python
# Usually raises ClaudeCodeTimeoutError
```

### Invalid Queries
```python
# Usually raises ClaudeCodeValidationError
```

### Configuration Problems
```python
# Usually raises ClaudeCodeConfigurationError
```
