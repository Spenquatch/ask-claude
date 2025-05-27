# Usage Examples

Comprehensive examples for using the Claude Code SDK Wrapper in various scenarios.

## Basic Usage

### Simple Queries

```python
from claude_code_wrapper import ask_claude, ClaudeCodeWrapper

# Quick convenience function
response = ask_claude("What is Python?")
print(response.content)

# Using wrapper directly
wrapper = ClaudeCodeWrapper()
response = wrapper.run("Explain machine learning")
print(f"Response: {response.content}")
print(f"Session ID: {response.session_id}")
```

### Model Selection

```python
# Use specific model
response = ask_claude("Complex reasoning task", model="opus")
response = ask_claude("Quick question", model="haiku")

# With temperature control
response = ask_claude(
    "Write creative content",
    model="sonnet",
    temperature=0.8
)

# Full generation control
wrapper = ClaudeCodeWrapper()
response = wrapper.ask(
    "Generate code",
    model="opus",
    temperature=0.1,
    max_tokens=2000,
    top_p=0.95
)
```

### JSON Format Responses

```python
from claude_code_wrapper import ask_claude_json, OutputFormat

# Convenience function
response = ask_claude_json("Write a Python function")
print(f"Content: {response.content}")
print(f"Cost: ${response.metrics.cost_usd:.6f}")
print(f"Duration: {response.metrics.duration_ms}ms")

# Using wrapper
wrapper = ClaudeCodeWrapper()
response = wrapper.run("Analyze this code", output_format=OutputFormat.JSON)
print(f"Metadata: {response.metadata}")
```

## Session Management

### Basic Sessions

```python
# Resume a specific session
wrapper = ClaudeCodeWrapper()
response = wrapper.resume_session(
    session_id="your-session-id",
    query="Continue our previous discussion"
)

# Continue the last session
response = wrapper.continue_last_session("What was my last question?")
```

### Session Context Manager

```python
# Multi-turn conversation with automatic session management
with wrapper.session() as session:
    # First exchange
    response1 = session.ask("I need help with a Python project")
    print(f"Session started: {session.session_id}")
    
    # Follow-up questions use the same session automatically
    response2 = session.ask("How do I read CSV files?")
    response3 = session.ask("Can you show me an example?")
    
    # Get conversation history
    history = session.get_history()
    print(f"Had {len(history)} exchanges")
    
    # Check for any errors in the conversation
    errors = [r for r in history if r.is_error]
    if errors:
        print(f"Encountered {len(errors)} errors")
```

### Session Configuration

```python
# Session with specific configuration
config = ClaudeCodeConfig(
    max_turns=5,
    system_prompt="You are a Python tutor.",
    timeout=30.0
)

with wrapper.session(**config.__dict__) as session:
    response = session.ask("Teach me about list comprehensions")
    response = session.ask("Show me some examples")
    response = session.ask("What are the performance benefits?")
```

## Streaming Responses

### Basic Streaming

```python
from claude_code_wrapper import ask_claude_streaming

# Using convenience function
print("Streaming response:")
for event in ask_claude_streaming("Write a long blog post about AI"):
    if event.get("type") == "message":
        print(event.get("content", ""), end="", flush=True)
    elif event.get("type") == "error":
        print(f"\nError: {event.get('message')}")
        break

print()  # Final newline
```

### Advanced Streaming with Event Handling

```python
def handle_streaming_response(query: str):
    """Handle streaming response with comprehensive event processing."""
    wrapper = ClaudeCodeWrapper()
    
    event_count = 0
    content_parts = []
    
    for event in wrapper.run_streaming(query):
        event_count += 1
        event_type = event.get("type", "unknown")
        
        match event_type:
            case "init":
                session_id = event.get("session_id", "no-session")
                print(f"🚀 Started session: {session_id}")
                
            case "message":
                content = event.get("content", "")
                if content:
                    content_parts.append(content)
                    print(content, end="", flush=True)
                    
            case "tool_use":
                tool = event.get("tool", "unknown")
                action = event.get("action", "unknown")
                print(f"\n🔧 Using {tool}: {action}")
                
            case "tool_result":
                result = event.get("result", "")
                print(f"   Result: {result[:50]}...")
                
            case "result":
                status = event.get("status", "unknown")
                stats = event.get("stats", {})
                print(f"\n✅ Completed: {status}")
                if stats:
                    print(f"   Stats: {stats}")
                    
            case "error":
                error_msg = event.get("message", "Unknown error")
                print(f"\n❌ Error: {error_msg}")
                
            case _:
                print(f"\n📝 Event: {event_type}")
    
    full_content = "".join(content_parts)
    print(f"\n\n📊 Summary: {event_count} events, {len(full_content)} chars")
    return full_content

# Use the function
content = handle_streaming_response("Explain quantum computing in detail")
```

### Streaming in Sessions

```python
# Streaming within a session context
with wrapper.session() as session:
    print("First question (streaming):")
    for event in session.ask_streaming("Explain Python decorators"):
        if event.get("type") == "message":
            print(event.get("content", ""), end="", flush=True)
    
    print("\n\nFollow-up question:")
    response = session.ask("Can you show a practical example?")
    print(response.content)
```

## Error Handling

### Basic Error Handling

```python
from claude_code_wrapper import (
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError,
    ClaudeCodeValidationError
)

def safe_query(query: str) -> str:
    """Execute query with comprehensive error handling."""
    try:
        wrapper = ClaudeCodeWrapper()
        response = wrapper.run(query, timeout=30.0)
        
        # Check response-level errors
        if response.is_error:
            return f"Response error: {response.error_type}"
        
        return response.content
        
    except ClaudeCodeTimeoutError:
        return "Query timed out. Please try a simpler question."
        
    except ClaudeCodeProcessError as e:
        return f"Process failed (code {e.returncode}): {e.stderr}"
        
    except ClaudeCodeValidationError as e:
        return f"Invalid input: {e.field} = {e.value}"
        
    except ClaudeCodeError as e:
        return f"Unexpected error: {e}"

# Use the safe function
result = safe_query("What is machine learning?")
print(result)
```

### Retry Logic with Error Handling

```python
import time
import random

def query_with_custom_retry(query: str, max_attempts: int = 3) -> str:
    """Query with custom retry logic."""
    
    for attempt in range(max_attempts):
        try:
            wrapper = ClaudeCodeWrapper()
            response = wrapper.run(query)
            
            if not response.is_error:
                return response.content
                
            # Handle response errors
            print(f"Attempt {attempt + 1} failed: {response.error_type}")
            
        except ClaudeCodeTimeoutError:
            print(f"Attempt {attempt + 1} timed out")
            
        except ClaudeCodeProcessError as e:
            print(f"Attempt {attempt + 1} process error: {e.returncode}")
            
        # Wait before retry with jitter
        if attempt < max_attempts - 1:
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    return "All attempts failed"

# Use with retry
result = query_with_custom_retry("Complex analysis task")
print(result)
```

## Configuration Examples

### Development Configuration

```python
# Development setup with verbose logging
dev_config = ClaudeCodeConfig(
    timeout=30.0,
    verbose=True,
    max_retries=1,           # Fast failure for development
    enable_metrics=True,
    log_level=10,            # DEBUG level
    system_prompt="You are a helpful coding assistant."
)

wrapper = ClaudeCodeWrapper(dev_config)
```

### Production Configuration

```python
# Production setup with reliability focus
prod_config = ClaudeCodeConfig(
    timeout=60.0,
    max_retries=5,
    retry_delay=2.0,
    retry_backoff_factor=2.0,
    enable_metrics=True,
    log_level=20,            # INFO level
    system_prompt="You are a professional AI assistant providing accurate information.",
    environment_vars={
        "ENVIRONMENT": "production",
        "SERVICE_NAME": "claude-wrapper"
    }
)

wrapper = ClaudeCodeWrapper(prod_config)
```

### Tool-Restricted Configuration

```python
# Security-focused configuration with tool restrictions
secure_config = ClaudeCodeConfig(
    allowed_tools=[
        "Python(import,def,class,print)",    # Safe Python operations
        "Bash(ls,cat,grep,head,tail,wc)"     # Read-only bash commands
    ],
    disallowed_tools=[
        "Bash(rm,del,sudo,chmod,mv)",        # Dangerous file operations
        "Python(exec,eval,__import__)"       # Potentially unsafe Python
    ],
    timeout=30.0,
    max_turns=3,                             # Limit conversation length
    working_directory=Path("./secure_workspace")
)

wrapper = ClaudeCodeWrapper(secure_config)
```

## Advanced Usage Patterns

### Batch Processing

```python
def batch_process_queries(queries: List[str]) -> List[ClaudeCodeResponse]:
    """Process multiple queries efficiently."""
    results = []
    wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(max_retries=2))
    
    # Use session for related queries to maintain context
    with wrapper.session() as session:
        for i, query in enumerate(queries, 1):
            try:
                print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
                response = session.ask(query)
                results.append(response)
                
                # Log progress
                if response.is_error:
                    print(f"  ⚠️  Error: {response.error_type}")
                else:
                    print(f"  ✅ Success ({len(response.content)} chars)")
                    
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                # Create error response instead of failing entire batch
                error_response = ClaudeCodeResponse(
                    content=f"Processing failed: {e}",
                    returncode=1,
                    is_error=True,
                    error_type="batch_processing_error"
                )
                results.append(error_response)
    
    return results

# Example usage
queries = [
    "What is Python?",
    "Explain list comprehensions",
    "Show me a decorator example",
    "How do I handle exceptions?"
]

results = batch_process_queries(queries)
for i, result in enumerate(results):
    print(f"Query {i+1}: {'Success' if not result.is_error else 'Error'}")
```

### Metrics Collection

```python
def run_with_metrics(query: str) -> dict:
    """Run query and collect comprehensive metrics."""
    wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(enable_metrics=True))
    
    start_time = time.time()
    
    try:
        response = wrapper.run(query)
        
        # Get wrapper-level metrics
        wrapper_metrics = wrapper.get_metrics()
        
        # Combine with response metrics
        return {
            "success": not response.is_error,
            "content_length": len(response.content),
            "session_id": response.session_id,
            "execution_time": response.execution_time,
            "cost_usd": response.metrics.cost_usd,
            "duration_ms": response.metrics.duration_ms,
            "num_turns": response.metrics.num_turns,
            "wrapper_metrics": wrapper_metrics,
            "total_time": time.time() - start_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "total_time": time.time() - start_time
        }

# Collect metrics
metrics = run_with_metrics("Explain neural networks")
print(f"Query metrics: {json.dumps(metrics, indent=2)}")
```

### Async-Style Processing (Thread-based)

```python
import concurrent.futures
import threading

def parallel_queries(queries: List[str], max_workers: int = 3) -> List[ClaudeCodeResponse]:
    """Process multiple queries in parallel using threads."""
    
    def process_single_query(query: str) -> ClaudeCodeResponse:
        """Process a single query with its own wrapper instance."""
        wrapper = ClaudeCodeWrapper()
        try:
            return wrapper.run(query)
        except Exception as e:
            return ClaudeCodeResponse(
                content=f"Error: {e}",
                returncode=1,
                is_error=True,
                error_type="parallel_processing_error"
            )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries
        future_to_query = {
            executor.submit(process_single_query, query): query 
            for query in queries
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                response = future.result()
                results.append(response)
                print(f"✅ Completed: {query[:30]}...")
            except Exception as e:
                print(f"❌ Failed: {query[:30]}... - {e}")
                results.append(ClaudeCodeResponse(
                    content=f"Thread error: {e}",
                    returncode=1,
                    is_error=True
                ))
        
        return results

# Example parallel processing
queries = [
    "What is machine learning?",
    "Explain Python decorators",
    "How do neural networks work?",
    "What is functional programming?"
]

results = parallel_queries(queries, max_workers=2)
print(f"Processed {len(results)} queries in parallel")
```

## Production Service Example

```python
class ClaudeService:
    """Production-ready Claude service with comprehensive features."""
    
    def __init__(self, config_path: str = None):
        """Initialize service with configuration."""
        self.config = self._load_config(config_path)
        self.wrapper = ClaudeCodeWrapper(self.config)
        self.logger = logging.getLogger(__name__)
        self._request_count = 0
        self._error_count = 0
    
    def _load_config(self, config_path: str) -> ClaudeCodeConfig:
        """Load configuration from file or environment."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config_data = json.load(f)
            return ClaudeCodeConfig(**config_data)
        else:
            # Use environment variables
            return ClaudeCodeConfig(
                timeout=float(os.getenv("CLAUDE_TIMEOUT", "60")),
                max_retries=int(os.getenv("CLAUDE_MAX_RETRIES", "3")),
                verbose=os.getenv("CLAUDE_VERBOSE", "false").lower() == "true"
            )
    
    def ask(self, query: str, **kwargs) -> dict:
        """Ask Claude with comprehensive error handling and logging."""
        self._request_count += 1
        request_id = f"req_{self._request_count}_{int(time.time())}"
        
        self.logger.info(f"[{request_id}] Processing query: {query[:100]}...")
        
        try:
            response = self.wrapper.run(query, **kwargs)
            
            if response.is_error:
                self._error_count += 1
                self.logger.warning(
                    f"[{request_id}] Response error: {response.error_type}"
                )
                
                return {
                    "success": False,
                    "error": response.error_type,
                    "content": response.content,
                    "request_id": request_id
                }
            
            self.logger.info(
                f"[{request_id}] Success: {len(response.content)} chars, "
                f"${response.metrics.cost_usd:.6f}"
            )
            
            return {
                "success": True,
                "content": response.content,
                "session_id": response.session_id,
                "metrics": {
                    "cost_usd": response.metrics.cost_usd,
                    "duration_ms": response.metrics.duration_ms,
                    "num_turns": response.metrics.num_turns
                },
                "request_id": request_id
            }
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"[{request_id}] Exception: {e}", exc_info=True)
            
            return {
                "success": False,
                "error": "service_error",
                "message": str(e),
                "request_id": request_id
            }
    
    def get_health(self) -> dict:
        """Get service health metrics."""
        wrapper_metrics = self.wrapper.get_metrics()
        error_rate = self._error_count / max(self._request_count, 1)
        
        return {
            "status": "healthy" if error_rate < 0.1 else "degraded",
            "requests_total": self._request_count,
            "errors_total": self._error_count,
            "error_rate": f"{error_rate:.2%}",
            "wrapper_metrics": wrapper_metrics
        }

# Usage
service = ClaudeService("config/production.json")

# Process requests
result = service.ask("What is machine learning?")
print(f"Result: {result}")

# Check health
health = service.get_health()
print(f"Health: {health}")
```

## MCP Auto-Approval Examples

### Basic Auto-Approval

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Auto-approve all MCP tools (development)
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "all"
    }
)
wrapper = ClaudeCodeWrapper(config)

# No manual approval needed!
response = wrapper.ask("Use the filesystem tool to read README.md")
```

### Allowlist-Based Approval

```python
# Only approve specific tools
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "allowlist",
        "allowlist": [
            "mcp__filesystem__read_file",
            "mcp__filesystem__list_directory",
            "mcp__database__query"
        ]
    }
)

# These tools will be auto-approved
response = wrapper.ask("Read all Python files in the src directory")
```

### Pattern-Based Approval

```python
# Approve based on patterns (read-only operations)
config = ClaudeCodeConfig(
    mcp_config_path=Path("mcp-servers.json"),
    mcp_auto_approval={
        "enabled": True,
        "strategy": "patterns",
        "allow_patterns": [
            "mcp__.*__read.*",
            "mcp__.*__list.*",
            "mcp__.*__get.*",
            "mcp__.*__query.*"
        ],
        "deny_patterns": [
            "mcp__.*__write.*",
            "mcp__.*__delete.*",
            "mcp__.*__execute.*",
            "mcp__.*__admin.*"
        ]
    }
)

# Read operations auto-approved, write operations denied
response = wrapper.ask("Analyze the codebase structure")
```

### CLI Usage with Auto-Approval

```bash
# Allow all tools
python cli_tool.py ask "Use sequential thinking to plan a project" \
    --mcp-config mcp-servers.json \
    --approval-strategy all

# Allowlist specific tools
python cli_tool.py stream "Read and summarize the documentation" \
    --mcp-config mcp-servers.json \
    --approval-strategy allowlist \
    --approval-allowlist "mcp__filesystem__read_file" "mcp__filesystem__list_directory"

# Pattern-based approval
python cli_tool.py ask "Query the database for user stats" \
    --mcp-config mcp-servers.json \
    --approval-strategy patterns \
    --approval-allow-patterns "mcp__.*__query.*" "mcp__.*__read.*"
```

This comprehensive set of examples covers most common usage patterns for the Claude Code SDK Wrapper, from basic queries to production-ready services with MCP auto-approval.
