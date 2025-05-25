# Error Handling Guide

The Claude Code SDK Wrapper provides comprehensive error handling with graceful degradation and detailed error information.

## Error Hierarchy

The wrapper uses a structured exception hierarchy for precise error handling:

```
ClaudeCodeError (base)
├── ClaudeCodeTimeoutError
├── ClaudeCodeProcessError
├── ClaudeCodeValidationError
└── ClaudeCodeConfigurationError
```

## Exception Types

### ClaudeCodeError (Base Exception)

All wrapper exceptions inherit from this base class.

```python
class ClaudeCodeError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None)
```

**Attributes:**
- `severity`: Error severity level (LOW, MEDIUM, HIGH, CRITICAL)
- `context`: Additional error context dictionary
- `timestamp`: When the error occurred

**Example:**
```python
try:
    response = wrapper.run(query)
except ClaudeCodeError as e:
    print(f"Error: {e}")
    print(f"Severity: {e.severity}")
    print(f"Context: {e.context}")
    print(f"Timestamp: {e.timestamp}")
```

### ClaudeCodeTimeoutError

Raised when requests exceed the configured timeout.

```python
try:
    response = wrapper.run(query, timeout=5.0)
except ClaudeCodeTimeoutError as e:
    print(f"Request timed out after {e.timeout_duration}s")
    # Handle timeout - maybe retry with longer timeout
    response = wrapper.run(query, timeout=30.0)
```

**When it occurs:**
- Claude Code process takes longer than `timeout` seconds
- Network issues causing delays
- Complex queries requiring more processing time

**How to handle:**
- Increase timeout for complex queries
- Implement retry logic with longer timeouts
- Break complex queries into smaller parts

### ClaudeCodeProcessError

Raised when the Claude Code process fails or returns a non-zero exit code.

```python
try:
    response = wrapper.run(query)
except ClaudeCodeProcessError as e:
    print(f"Process failed with code {e.returncode}")
    print(f"Error output: {e.stderr}")
    
    if e.returncode == 1:
        print("General Claude Code error")
    elif e.returncode == 2:
        print("Command line argument error")
    # Handle based on return code
```

**Common causes:**
- Invalid Claude Code command syntax
- Authentication issues
- Claude Code binary not properly installed
- Invalid tool configurations

**How to handle:**
- Check Claude Code installation and configuration
- Verify authentication
- Review command construction
- Check tool permissions

### ClaudeCodeValidationError

Raised when input validation fails before sending to Claude Code.

```python
try:
    response = wrapper.run("")  # Empty query
except ClaudeCodeValidationError as e:
    print(f"Validation failed for field '{e.field}': {e.value}")
    # e.field = "query", e.value = ""
```

**Common validation failures:**
- Empty queries
- Queries exceeding length limits
- Invalid configuration parameters
- Invalid file paths

**How to handle:**
- Validate input before calling wrapper
- Provide user-friendly error messages
- Sanitize input data

### ClaudeCodeConfigurationError

Raised when wrapper configuration is invalid.

```python
try:
    config = ClaudeCodeConfig(
        timeout=-1.0,  # Invalid negative timeout
        max_retries=-1  # Invalid negative retries
    )
except ClaudeCodeConfigurationError as e:
    print(f"Configuration error in field '{e.config_field}': {e}")
```

**Common configuration errors:**
- Invalid file paths
- Negative timeouts or retry counts
- Missing required binaries
- Invalid MCP configurations

**How to handle:**
- Validate configuration at startup
- Provide configuration templates
- Use environment-specific configs

## Error Handling Patterns

### Basic Error Handling

```python
from claude_code_wrapper import (
    ClaudeCodeWrapper, 
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError
)

def safe_query(query: str) -> str:
    """Execute query with basic error handling."""
    try:
        wrapper = ClaudeCodeWrapper()
        response = wrapper.run(query)
        
        # Check response-level errors
        if response.is_error:
            return f"Response error: {response.error_type}"
        
        return response.content
        
    except ClaudeCodeTimeoutError:
        return "Request timed out. Please try a simpler question."
        
    except ClaudeCodeProcessError as e:
        return f"Service temporarily unavailable (code {e.returncode})"
        
    except ClaudeCodeError as e:
        return f"An error occurred: {e}"

# Usage
result = safe_query("What is machine learning?")
print(result)
```

### Comprehensive Error Handling

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def robust_query(query: str, max_retries: int = 3) -> Optional[str]:
    """Execute query with comprehensive error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            # Configure wrapper with appropriate settings
            config = ClaudeCodeConfig(
                timeout=30.0 + (attempt * 10),  # Increase timeout on retries
                max_retries=1,  # Handle retries manually
                verbose=attempt > 0  # Enable verbose logging on retries
            )
            
            wrapper = ClaudeCodeWrapper(config)
            response = wrapper.run(query)
            
            # Check for response-level errors
            if response.is_error:
                error_msg = f"Response error: {response.error_type}"
                if response.error_subtype:
                    error_msg += f" ({response.error_subtype})"
                
                logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                
                # Some errors are retryable, others are not
                if response.error_type in ["timeout", "rate_limit", "server_error"]:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                return None
            
            # Success
            logger.info(f"Query succeeded on attempt {attempt + 1}")
            return response.content
            
        except ClaudeCodeTimeoutError as e:
            logger.warning(f"Attempt {attempt + 1} timed out after {e.timeout_duration}s")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            
        except ClaudeCodeProcessError as e:
            logger.error(f"Attempt {attempt + 1} process error: {e.returncode}")
            
            # Some process errors are retryable
            if e.returncode in [2, 124]:  # Argument errors, timeout
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            # Fatal process errors
            logger.error(f"Fatal process error: {e.stderr}")
            return None
            
        except ClaudeCodeValidationError as e:
            # Validation errors are not retryable
            logger.error(f"Validation error: {e.field} = {e.value}")
            return None
            
        except ClaudeCodeConfigurationError as e:
            # Configuration errors are not retryable
            logger.error(f"Configuration error: {e.config_field}")
            return None
            
        except ClaudeCodeError as e:
            logger.error(f"Attempt {attempt + 1} general error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    
    logger.error(f"All {max_retries} attempts failed")
    return None

# Usage
result = robust_query("Explain quantum computing")
if result:
    print(result)
else:
    print("Failed to get response after multiple attempts")
```

### Error Recovery Strategies

```python
class ClaudeService:
    """Service with advanced error recovery strategies."""
    
    def __init__(self):
        self.wrapper = ClaudeCodeWrapper()
        self.fallback_responses = {
            "timeout": "I'm taking longer than expected. Please try again with a simpler question.",
            "process_error": "I'm temporarily unavailable. Please try again in a moment.",
            "validation_error": "Please check your question and try again.",
            "general_error": "Something went wrong. Please try again."
        }
    
    def ask_with_fallback(self, query: str) -> dict:
        """Ask with intelligent fallback responses."""
        try:
            response = self.wrapper.run(query)
            
            if response.is_error:
                fallback = self._get_fallback_response(response.error_type)
                return {
                    "success": False,
                    "content": fallback,
                    "error_type": response.error_type,
                    "original_error": response.content
                }
            
            return {
                "success": True,
                "content": response.content,
                "session_id": response.session_id,
                "metrics": {
                    "cost": response.metrics.cost_usd,
                    "duration": response.metrics.duration_ms
                }
            }
            
        except ClaudeCodeTimeoutError:
            return {
                "success": False,
                "content": self.fallback_responses["timeout"],
                "error_type": "timeout",
                "retry_suggested": True
            }
            
        except ClaudeCodeProcessError as e:
            # Try to provide specific guidance based on error
            if "authentication" in e.stderr.lower():
                content = "Authentication issue. Please check your Claude Code setup."
            elif "not found" in e.stderr.lower():
                content = "Claude Code binary not found. Please check installation."
            else:
                content = self.fallback_responses["process_error"]
            
            return {
                "success": False,
                "content": content,
                "error_type": "process_error",
                "details": e.stderr
            }
            
        except ClaudeCodeValidationError as e:
            return {
                "success": False,
                "content": f"Please check your {e.field}: {self.fallback_responses['validation_error']}",
                "error_type": "validation_error",
                "field": e.field
            }
            
        except ClaudeCodeError as e:
            return {
                "success": False,
                "content": self.fallback_responses["general_error"],
                "error_type": "general_error",
                "severity": e.severity.value
            }
    
    def _get_fallback_response(self, error_type: str) -> str:
        """Get appropriate fallback response for error type."""
        return self.fallback_responses.get(error_type, self.fallback_responses["general_error"])
```

## Response-Level Error Handling

In addition to exceptions, the wrapper also handles errors at the response level:

```python
response = wrapper.run(query)

if response.is_error:
    print(f"Response error: {response.error_type}")
    
    if response.error_subtype:
        print(f"Subtype: {response.error_subtype}")
    
    # Handle different error types
    match response.error_type:
        case "tool_error":
            print("Tool execution failed")
            
        case "permission_error":
            print("Permission denied for requested action")
            
        case "rate_limit":
            print("Rate limit exceeded, please wait")
            
        case "content_filter":
            print("Content was filtered, please modify your query")
            
        case _:
            print(f"Unknown error: {response.error_type}")
else:
    print(f"Success: {response.content}")
```

## Streaming Error Handling

Streaming responses require special error handling:

```python
def handle_streaming_with_errors(query: str):
    """Handle streaming response with comprehensive error recovery."""
    
    try:
        error_count = 0
        content_parts = []
        
        for event in wrapper.run_streaming(query):
            event_type = event.get("type", "unknown")
            
            match event_type:
                case "error":
                    error_count += 1
                    error_msg = event.get("message", "Unknown streaming error")
                    print(f"Stream error: {error_msg}", file=sys.stderr)
                    
                    # Decide whether to continue or abort
                    if error_count > 3:
                        print("Too many streaming errors, aborting", file=sys.stderr)
                        break
                    
                case "parse_error":
                    # JSON parsing errors in stream
                    raw_line = event.get("raw_line", "")
                    print(f"Parse error: {raw_line[:50]}...", file=sys.stderr)
                    
                case "message":
                    content = event.get("content", "")
                    if content:
                        content_parts.append(content)
                        print(content, end="", flush=True)
                        
                case "result":
                    status = event.get("status", "unknown")
                    if status != "complete":
                        print(f"\nStream ended unexpectedly: {status}", file=sys.stderr)
        
        print()  # Final newline
        
        # Return results with error information
        return {
            "content": "".join(content_parts),
            "error_count": error_count,
            "success": error_count == 0
        }
        
    except KeyboardInterrupt:
        print("\nStream interrupted by user", file=sys.stderr)
        return {"content": "", "error_count": 1, "success": False}
        
    except Exception as e:
        print(f"\nStreaming failed: {e}", file=sys.stderr)
        return {"content": "", "error_count": 1, "success": False}

# Usage
result = handle_streaming_with_errors("Write a long story")
if result["success"]:
    print(f"Streaming completed successfully: {len(result['content'])} chars")
else:
    print(f"Streaming completed with {result['error_count']} errors")
```

## Logging and Observability

### Structured Error Logging

```python
import logging
import json
from datetime import datetime

class ErrorLogger:
    """Structured error logging for Claude wrapper."""
    
    def __init__(self):
        self.logger = logging.getLogger("claude_wrapper_errors")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_error(self, error: ClaudeCodeError, query: str, context: dict = None):
        """Log error with structured information."""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": error.severity.value,
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "context": context or {}
        }
        
        # Add specific error details
        if isinstance(error, ClaudeCodeTimeoutError):
            error_data["timeout_duration"] = error.timeout_duration
        elif isinstance(error, ClaudeCodeProcessError):
            error_data["return_code"] = error.returncode
            error_data["stderr"] = error.stderr
        elif isinstance(error, ClaudeCodeValidationError):
            error_data["field"] = error.field
            error_data["value"] = str(error.value)
        
        self.logger.error(json.dumps(error_data))

# Usage
error_logger = ErrorLogger()

try:
    response = wrapper.run(query)
except ClaudeCodeError as e:
    error_logger.log_error(e, query, {"user_id": "user123", "session": "abc"})
    raise
```

### Error Metrics Collection

```python
from collections import defaultdict, Counter
import time

class ErrorMetrics:
    """Collect and analyze error patterns."""
    
    def __init__(self):
        self.error_counts = Counter()
        self.error_history = []
        self.start_time = time.time()
    
    def record_error(self, error: ClaudeCodeError, query: str):
        """Record error occurrence."""
        error_record = {
            "timestamp": time.time(),
            "type": type(error).__name__,
            "severity": error.severity.value,
            "query_length": len(query),
            "message": str(error)
        }
        
        self.error_counts[type(error).__name__] += 1
        self.error_history.append(error_record)
    
    def get_error_summary(self) -> dict:
        """Get error summary statistics."""
        total_time = time.time() - self.start_time
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / (total_time / 60),  # errors per minute
            "error_types": dict(self.error_counts),
            "most_common_error": self.error_counts.most_common(1)[0] if self.error_counts else None,
            "recent_errors": self.error_history[-10:]  # Last 10 errors
        }

# Usage
metrics = ErrorMetrics()

try:
    response = wrapper.run(query)
except ClaudeCodeError as e:
    metrics.record_error(e, query)
    # Handle error...

# Get error summary
summary = metrics.get_error_summary()
print(f"Error summary: {summary}")
```

## Best Practices

### 1. Graceful Degradation

Always provide fallback responses rather than failing completely:

```python
def get_response(query: str) -> str:
    """Get response with graceful degradation."""
    try:
        response = wrapper.run(query)
        return response.content if not response.is_error else "I encountered an issue processing your request."
    except ClaudeCodeError:
        return "I'm temporarily unavailable. Please try again later."
```

### 2. User-Friendly Error Messages

Transform technical errors into user-friendly messages:

```python
def user_friendly_error(error: ClaudeCodeError) -> str:
    """Convert technical error to user-friendly message."""
    if isinstance(error, ClaudeCodeTimeoutError):
        return "Your request is taking longer than expected. Please try a simpler question."
    elif isinstance(error, ClaudeCodeValidationError):
        return "Please check your input and try again."
    elif isinstance(error, ClaudeCodeProcessError):
        return "I'm having technical difficulties. Please try again in a moment."
    else:
        return "Something went wrong. Please try again."
```

### 3. Context-Aware Error Handling

Use error context to provide specific guidance:

```python
def contextual_error_handling(query: str, user_type: str = "general"):
    """Handle errors based on user context."""
    try:
        response = wrapper.run(query)
        return response.content
    except ClaudeCodeTimeoutError:
        if user_type == "developer":
            return "Query timed out. Try increasing timeout or simplifying the request."
        else:
            return "Your request is taking too long. Please try a shorter question."
    except ClaudeCodeValidationError as e:
        if user_type == "developer":
            return f"Validation failed: {e.field} = {e.value}"
        else:
            return "Please check your input and try again."
```

### 4. Error Recovery

Implement intelligent retry strategies:

```python
def smart_retry(query: str, max_attempts: int = 3):
    """Retry with intelligent backoff and adaptation."""
    timeout = 30.0
    
    for attempt in range(max_attempts):
        try:
            config = ClaudeCodeConfig(timeout=timeout)
            wrapper = ClaudeCodeWrapper(config)
            response = wrapper.run(query)
            
            if not response.is_error:
                return response.content
                
        except ClaudeCodeTimeoutError:
            timeout *= 1.5  # Increase timeout for next attempt
        except ClaudeCodeProcessError:
            time.sleep(2 ** attempt)  # Exponential backoff
        except ClaudeCodeValidationError:
            break  # Don't retry validation errors
    
    return "Unable to process request after multiple attempts."
```

By following these error handling patterns, you can build robust applications that gracefully handle all types of errors while providing excellent user experience.
