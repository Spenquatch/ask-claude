# Recommended API Improvements for Ask Claude

## Current Problems

1. **Too verbose**: Extensive logging clutters output (but metadata IS valuable for apps)
2. **Inconsistent access**: `response.content` should be clean, but metadata should be easily accessible
3. **JSON confusion**: Should `ask_claude_json()` return JSON string or parsed Python objects?

## Revised Understanding

- **CLI Tool**: Gives plain text output for command-line usage
- **SDK Wrapper**: For Python applications that NEED metadata (costs, tool usage, errors, etc.)
- **JSON Output**: Should be valid JSON strings, not pre-parsed Python objects
- **Metadata Matters**: Applications need to know about tool usage, costs, errors for pipelines/automation

## Proposed API Design

### Core Principle: Clean Content + Easy Metadata Access
```python
# The response object should have clean content but easy metadata access
response = ask_claude("What is Python?")

# Clean content access
print(response.content)  # Just the text, no noise

# Easy metadata access when needed
if response.tools_used:
    log_tool_usage(response.tools_used)

if response.cost > 0.01:
    alert_high_cost(response.cost)

if response.duration > 30000:  # duration is in ms
    log_slow_query(response.duration)

# Cache efficiency tracking
cache_hit_ratio = response.tokens_cache_read / response.tokens_input if response.tokens_input > 0 else 0
if cache_hit_ratio > 0.9:
    log_excellent_caching(cache_hit_ratio)
```

### Quiet Logging by Default
```python
# Current: Always logs verbosely
ask_claude("Hello")  # Prints extensive logging

# Proposed: Quiet by default, verbose on demand
ask_claude("Hello")  # Clean execution, minimal logging
ask_claude("Hello", verbose=True)  # Show detailed logging when debugging
```

## Specific Function Recommendations

### 1. `ask_claude()` - Universal Function with Clean API
```python
def ask_claude(
    query: str,
    *,
    output: Literal["text", "text-stream", "json", "json-stream"] = "text",
    debug: bool = False,
    **kwargs
) -> ClaudeCodeResponse | Iterator[dict[str, Any]]:
    # Note: Full signature includes metadata=True, session=None as defaults
    # Note: ClaudeCodeResponse includes .json() method for parsing JSON responses
    """Ask Claude with configurable output and streaming options.

    Args:
        query: Question to ask
        output: Output format ("text", "text-stream", "json", "json-stream")
        debug: If True, show detailed logging for troubleshooting
        **kwargs: Additional options (timeout, model, etc.)

    Returns:
        ClaudeCodeResponse: Response object (when output is "text" or "json")
        Iterator[dict]: Streaming events (when output ends with "-stream")

    Examples:
        # Text response (default)
        response = ask_claude("What is Python?")
        print(response.content)

        # JSON response
        response = ask_claude("List 3 colors", output="json")
        print(response.content)  # '["red", "blue", "green"]' <- This IS valid JSON!
        colors = response.json()  # ["red", "blue", "green"] <- Parsed to Python list (method defined below)

        # Text streaming (strips JSON, shows just content)
        for event in ask_claude("Tell a story", output="text-stream"):
            if event['type'] == 'content':
                print(event['data'], end='')

        # JSON streaming (full JSON events)
        for event in ask_claude("Get data", output="json-stream"):
            if event['type'] == 'content':
                print(event['data'], end='')

        # Debug mode
        response = ask_claude("Complex query", debug=True)
        # Shows detailed execution logs
    """
```

### 2. NO MORE SEPARATE FUNCTIONS!
```python
# ❌ OLD way (multiple functions - REMOVE ALL OF THESE):
ask_claude_json("List colors")
ask_claude_streaming("Write story")
ask_claude_detailed("Complex query")

# ✅ NEW way (ONE function with parameters):
ask_claude("List colors", output="json")
ask_claude("Write story", output="text-stream")
ask_claude("Complex query", debug=True)
```

## Key Improvements Needed

### 1. Enhanced ClaudeCodeResponse - Flattened for AI/ML Pipelines
```python
@dataclass
class ClaudeCodeResponse:
    # Core response
    content: str                     # Clean response text
    success: bool                    # Direct success indicator

    # Critical metrics (flat for easy pipeline access)
    cost: float                      # Direct: cost_usd from result event
    duration: float                  # Direct: duration_ms from result event
    duration_api: float              # Direct: duration_api_ms (actual API time)
    tokens_input: int               # Direct: input_tokens from usage
    tokens_output: int              # Direct: output_tokens from usage
    tokens_cache_read: int          # Direct: cache_read_input_tokens (cache efficiency)
    tokens_cache_created: int       # Direct: cache_creation_input_tokens (cache building)
    tools_used: list[str]           # Direct: tools from system event
    model: str                      # Direct: model from assistant event
    ttft_ms: float                  # Direct: ttftMs (time to first token)
    num_turns: int                  # Direct: num_turns (conversation length)

    # Session & error tracking
    session_id: str | None = None
    error_type: str | None = None

    # Full metadata for advanced users
    metadata: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse content as JSON if it's JSON format."""
        return json.loads(self.content)

    def __str__(self) -> str:
        """String representation returns just the content."""
        return self.content
```

**Benefits for AI/ML Pipelines:**
- `response.cost` - instant budget tracking and cost monitoring
- `response.duration` / `response.duration_api` - performance and latency monitoring
- `response.tokens_*` - capacity planning and token usage analysis
- `response.tokens_cache_*` - cache efficiency tracking and optimization
- `response.tools_used` - capability analysis and tool usage patterns
- `response.model` - model versioning and A/B testing
- `response.ttft_ms` - user experience monitoring (time to first token)
- `response.num_turns` - conversation complexity analysis
- `response.metadata` - full event details when needed for debugging

### 2. Quiet by Default
- Change default logging level from INFO to WARNING for convenience functions
- Add `debug=True` parameter to restore detailed logging and show all response data
- Keep detailed logging for wrapper methods (since those are more "advanced" usage)

### 3. Better JSON Handling
```python
# Current confusion:
response = ask_claude_json("List 3 colors")
print(response.content)  # Is this JSON string or parsed data?

# Proposed clarity:
response = ask_claude_json("List 3 colors")
print(response.content)    # JSON string: '["red", "blue", "green"]'
print(response.json())     # Parsed: ["red", "blue", "green"] (using .json() method)
```

## Example Usage (Revised)

### Clean and Simple:
```python
from ask_claude import ask_claude

# Text response (default)
response = ask_claude("What is Python?")
print(response.content)  # Just the text response

# Metadata is easily accessible for automation
if response.cost > 0.01:
    alert_high_cost(response.cost)
if response.tools_used:
    log_tool_usage(response.tools_used)

# Cache efficiency monitoring
cache_ratio = response.tokens_cache_read / response.tokens_input if response.tokens_input > 0 else 0
if cache_ratio > 0.8:
    log_cache_efficiency("High cache hit rate", cache_ratio)

# Performance monitoring
if response.ttft_ms > 3000:  # Time to first token > 3s
    alert_slow_response(response.ttft_ms)

# Token usage tracking
total_tokens = response.tokens_input + response.tokens_output
if total_tokens > 100000:
    log_high_token_usage(total_tokens, response.model)
```

### JSON Handling with Unified API:
```python
# One function, explicit format
response = ask_claude("List 5 programming languages", output="json")
print(response.content)  # '["Python", "JavaScript", "Java", "Go", "Rust"]'
languages = response.json()  # Parse to Python list (method defined in ClaudeCodeResponse class)

# Even cleaner with aliasing
from ask_claude import ask_claude as ask
data = ask("Get user data", output="json").json()
```

### JSON Clarification:
```python
# IMPORTANT: These strings ARE valid JSON!
'["red", "blue", "green"]'              # Valid JSON array
'{"name": "Alice", "age": 30}'          # Valid JSON object
'"hello world"'                         # Valid JSON string
'42'                                    # Valid JSON number
'true'                                  # Valid JSON boolean

# Example:
response = ask_claude("List 3 colors as array", output="json")
print(response.content)         # '["red", "blue", "green"]'
print(type(response.content))   # <class 'str'> - It's a JSON string

colors = response.json()        # Parse JSON string to Python (see method definition below)
print(colors)                   # ['red', 'blue', 'green']
print(type(colors))            # <class 'list'> - Now it's Python
```

### Complete Unified API Examples:
```python
# Simple text (default)
response = ask_claude("What is Python?")
print(response.content)

# JSON output
response = ask_claude("List 3 programming languages", output="json")
languages = response.json()  # Parse JSON string to Python object (method shown below)

# Streaming (always JSON format - CLI requirement)
for event in ask_claude("Write a haiku", output="text-stream"):
    if event['type'] == 'content':
        print(event['data'], end='')

# Without metadata for cleaner response
response = ask_claude("Simple query", metadata=False)
print(response)  # Just the content via __str__

# Session handling
response = ask_claude("Add error handling", session="continue")  # Continue most recent conversation
response = ask_claude("Update the tests", session="550e8400-e29b-41d4-a716-446655440000")  # Resume specific session

# Debug mode
response = ask_claude("Complex query", debug=True)

# All options combined
for event in ask_claude("Stream response", output="json-stream", debug=True, session="abc123"):
    print(event)
```

## Claude CLI Compatibility Notes

### Confirmed Feasible:
1. **Output formats**: The CLI supports `text`, `json`, and `stream-json` (we map to `json-stream`)
2. **Non-interactive mode**: Already using `-p` flag with MCP, `--print` otherwise
3. **Metadata extraction**: All proposed fields available in JSON responses
4. **Session handling**: Maps to CLI flags:
   - `session="continue"` → `claude -p --continue "query"`
   - `session="session_id"` → `claude -p --resume session_id "query"`

### Implementation Requirements:
1. **"text-stream" format**: Must be implemented by:
   - Using `--output-format stream-json` internally
   - Extracting text content from assistant message events
   - Yielding simplified `{'type': 'content', 'data': text}` events

2. **Response field extraction**: Must parse JSON events to populate flattened fields:
   - `cost` ← `result.cost_usd`
   - `duration` ← `result.duration_ms`
   - `tokens_*` ← `assistant.message.usage.*`
   - `tools_used` ← `system.tools`
   - `model` ← `assistant.message.model`

## Complete Implementation Plan

### One Function to Rule Them All:
```python
ask_claude(
    query: str,
    *,
    output: Literal["text", "text-stream", "json", "json-stream"] = "text",
    debug: bool = False,
    metadata: bool = True,  # NEW: Control metadata inclusion
    session: str | None = None,
    **kwargs
) -> ClaudeCodeResponse | Iterator[dict[str, Any]]
```

### Functions to REMOVE Entirely:
```python
# Convenience function duplicates:
❌ ask_claude_json()           # → ask_claude(..., output="json")
❌ ask_claude_streaming()      # → ask_claude(..., output="text-stream")

# Session function duplicates:
❌ continue_claude()           # → ask_claude("", session="continue")
❌ resume_claude()             # → ask_claude(..., session="session_id")
❌ ask_claude_with_session()   # → ask_claude(..., session="session_id")

# Wrapper method duplicates:
❌ wrapper.ask()               # → wrapper.run()
❌ wrapper.ask_json()          # → wrapper.run(...).json()
❌ wrapper.ask_streaming()     # → wrapper.run_streaming()
```

### Enhanced ClaudeCodeResponse:
```python
@dataclass
class ClaudeCodeResponse:
    # Core response
    content: str                     # Clean response text
    success: bool                    # Direct success indicator

    # Critical metrics (flat for easy pipeline access)
    cost: float                      # Direct: cost_usd from result event
    duration: float                  # Direct: duration_ms from result event
    duration_api: float              # Direct: duration_api_ms (actual API time)
    tokens_input: int               # Direct: input_tokens from usage
    tokens_output: int              # Direct: output_tokens from usage
    tokens_cache_read: int          # Direct: cache_read_input_tokens (cache efficiency)
    tokens_cache_created: int       # Direct: cache_creation_input_tokens (cache building)
    tools_used: list[str]           # Direct: tools from system event
    model: str                      # Direct: model from assistant event
    ttft_ms: float                  # Direct: ttftMs (time to first token)
    num_turns: int                  # Direct: num_turns (conversation length)

    # Session & error tracking
    session_id: str | None = None
    error_type: str | None = None

    # Full metadata for advanced users
    metadata: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse content as JSON if it's JSON format."""
        return json.loads(self.content)

    def __str__(self) -> str:
        """String representation returns just the content."""
        return self.content
```

## Key Migration Notes

### Old vs New Field Mapping
When updating code that uses ClaudeCodeResponse:
```python
# OLD WAY → NEW WAY
response.returncode == 0 → response.success
response.is_error → not response.success
response.execution_time → response.duration
response.metrics.cost_usd → response.cost
response.metrics.* → response.* (all flattened)
response.exit_code → REMOVED (use success)
response.raw_output → response.content
```

## Complete File-by-File Changes Required

### Core Files:

#### 1. `ask_claude/__init__.py`
```python
# REMOVE these exports:
❌ "ask_claude_json"
❌ "ask_claude_streaming"

# KEEP these exports:
✅ "ask_claude"  # Updated implementation
✅ "ClaudeCodeWrapper"
✅ "ClaudeCodeResponse"  # Enhanced version
```

#### 2. `ask_claude/wrapper.py`
```python
# REMOVE these functions entirely:
❌ ask_claude_json() (lines 1861-1875)
❌ ask_claude_streaming() (lines 1877-1889)
❌ continue_claude() (lines 1812-1815)
❌ resume_claude() (lines 1818-1823)
❌ ask_claude_with_session() (lines 1826-1841)

# REMOVE these wrapper methods:
❌ wrapper.ask() (alias for run)
❌ wrapper.ask_json() (confusing name)
❌ wrapper.ask_streaming() (alias for run_streaming)

# UPDATE ask_claude() to new signature:
✅ def ask_claude(query, *, output="text", stream=False, debug=False, session=None, **kwargs)

# ENHANCE ClaudeCodeResponse with:
✅ .json() method
✅ .cost property (direct access)
✅ .tools_used property
✅ __str__() method
```

### Example Files:

#### 3. `examples/getting_started.py`
```python
# Line 47: CHANGE
❌ response = ask_claude_json("List 3 programming languages")
✅ response = ask_claude("List 3 programming languages", output="json")

# Line 110: CHANGE
❌ for chunk in ask_claude_streaming("Write a haiku"):
✅ for chunk in ask_claude("Write a haiku", output="text-stream"):
```

#### 4. `examples/production_example.py`
```python
# Line 69: CHANGE
❌ json_response = ask_claude_json("List 5 colors")
✅ json_response = ask_claude("List 5 colors", output="json")

# Line 192: CHANGE
❌ for event in ask_claude_streaming("Write a story"):
✅ for event in ask_claude("Write a story", output="text-stream"):
```

#### 5. `examples/mcp_example.py`
```python
# Lines 48, 123, 153, 270, 282: CHANGE
❌ response = wrapper.ask("query")
✅ response = wrapper.run("query")
```

#### 6. `examples/cache_configuration_example.py`
```python
# Line 173: CHANGE
❌ response = wrapper.ask("What is caching?")
✅ response = wrapper.run("What is caching?")
```

### Test Files:

#### 7. `tests/test_claude_code_wrapper.py`
```python
# Multiple locations: UPDATE all usages
❌ ask_claude_json(), wrapper.ask(), wrapper.ask_json()
✅ ask_claude(..., output="json"), wrapper.run(), wrapper.run(...).json()
```

#### 8. `tests/test_session_management.py`
```python
# Lines 312, 320, 325, 333, 340, 349: CHANGE
❌ continue_claude(), resume_claude(), ask_claude_with_session()
✅ ask_claude("", session="continue"), ask_claude(..., session="id")
```

### Documentation Files:

#### 9. `README.md`
```python
# Lines 76-77: UPDATE streaming example
❌ for chunk in ask_claude_streaming("Write a haiku"):
✅ for chunk in ask_claude("Write a haiku", output="text-stream"):
```

#### 10. `docs/api-reference.md`
```python
# Remove documentation for deprecated functions
# Add new unified ask_claude() documentation
```

#### 11. `docs/session-management.md`
```python
# Update all session examples to use unified API
❌ continue_claude(), resume_claude()
✅ ask_claude(..., session="continue"), ask_claude(..., session="id")
```

#### 12. `CHANGELOG.md`
```python
# Update feature references to new API
```

### Test Scripts:

#### 13. `simple_test.py`
```python
# Line 31-33: CHANGE
❌ result = ask_claude_json(query)
✅ result = ask_claude(query, output="json")

# Line 41-43: CHANGE
❌ stream = ask_claude_streaming(query)
✅ stream = ask_claude(query, output="text-stream")

# Line 62-63: CHANGE
❌ json_result = wrapper.ask_json(query)
✅ json_result = wrapper.run(query, output_format=OutputFormat.JSON).json()
```

#### 14. `test_claude_functions.py`
```python
# Update all function imports and calls
❌ from ask_claude import ask_claude_json, ask_claude_streaming
✅ from ask_claude import ask_claude  # Only need one function

# Update all test calls to use parameters instead of separate functions
```

## Detailed Implementation Steps

### Step 1: Update Core Module (`ask_claude/wrapper.py`)

**Remove these entire function definitions:**
- Lines 1861-1875: `ask_claude_json()`
- Lines 1877-1889: `ask_claude_streaming()`
- Lines 1812-1815: `continue_claude()`
- Lines 1818-1823: `resume_claude()`
- Lines 1826-1841: `ask_claude_with_session()`
- Lines 1437-1439: `wrapper.ask()` method
- Lines 1441-1443: `wrapper.ask_streaming()` method
- Lines 1445-1456: `wrapper.ask_json()` method

**Replace `ask_claude()` function (lines 1845-1859) with:**
```python
def ask_claude(
    query: str,
    *,
    output: Literal["text", "text-stream", "json", "json-stream"] = "text",
    debug: bool = False,
    metadata: bool = True,
    session: str | None = None,
    **kwargs: Any
) -> ClaudeCodeResponse | Iterator[dict[str, Any]]:
    """Unified Claude function with all options as parameters.

    Args:
        query: Question to ask Claude
        output: Response format ("text", "text-stream", "json", "json-stream")
        debug: Show detailed logging
        metadata: Include full metadata in response (cost, tools_used, etc.)
        session: Session handling ("continue" for most recent, session_id for specific, or None)
    """
    try:
        # Set up logging level based on debug
        log_level = logging.INFO if debug else logging.WARNING
        config = ClaudeCodeConfig(log_level=log_level, **kwargs)
        wrapper = ClaudeCodeWrapper(config)

        # Handle session parameter
        if session == "continue":
            # Continue most recent conversation (query required for non-interactive)
            return wrapper.continue_conversation(query)
        elif session:
            # Resume specific session by ID (query required for non-interactive)
            return wrapper.resume_specific_session(session, query)

        # Handle streaming outputs
        if output.endswith("-stream"):
            if output == "text-stream":
                # Claude CLI only supports JSON streaming, so we extract text
                for event in wrapper.run_streaming(query):
                    if event.get('type') == 'assistant' and 'message' in event:
                        # Extract text content from assistant messages
                        message = event['message']
                        if isinstance(message.get('content'), list):
                            for content in message['content']:
                                if content.get('type') == 'text':
                                    yield {'type': 'content', 'data': content.get('text', '')}
                        elif isinstance(message.get('content'), str):
                            yield {'type': 'content', 'data': message['content']}
            elif output == "json-stream":
                # Return raw JSON stream events
                yield from wrapper.run_streaming(query)

        # Handle non-streaming outputs
        output_format = OutputFormat.JSON if output == "json" else OutputFormat.TEXT
        response = wrapper.run(query, output_format=output_format)

        # Strip metadata if requested
        if not metadata:
            # Return response without full metadata dict
            response.metadata = {}  # Clear the metadata dict
            return response

        return response

    except Exception as e:
        if debug:
            logger = ClaudeCodeLogger.setup_logger(__name__)
            logger.error(f"Ask Claude failed: {e}")
        return ClaudeCodeResponse(
            content=f"Error: {e}",
            success=False,
            cost=0.0,
            duration=0.0,
            duration_api=0.0,
            tokens_input=0,
            tokens_output=0,
            tokens_cache_read=0,
            tokens_cache_created=0,
            tools_used=[],
            model="unknown",
            ttft_ms=0.0,
            num_turns=0,
            error_type="convenience_function_error",
        )
```

**Replace `ClaudeCodeResponse` class (lines 160-195) with flattened structure:**

**IMPORTANT Field Name Changes (No Backward Compatibility):**
- `returncode` → REMOVED (use `success` boolean instead)
- `is_error` → REMOVED (use `success` boolean instead)
- `execution_time` → `duration` (in milliseconds)
- `metrics.*` → All flattened to top level
- `raw_output` → REMOVED (use `content`)
- `stderr` → REMOVED (included in error metadata if needed)
- `timestamp` → REMOVED (can be in metadata if needed)
- `retries` → REMOVED (can be in metadata if needed)
- `exit_code` property → REMOVED
- `duration` property → Now a direct field (not alias)
```python
@dataclass
class ClaudeCodeResponse:
    # Core response
    content: str                     # Clean response text
    success: bool                    # Direct success indicator

    # Critical metrics (flat for easy pipeline access)
    cost: float                      # Direct: extracted from result.cost_usd
    duration: float                  # Direct: extracted from result.duration_ms
    duration_api: float              # Direct: extracted from result.duration_api_ms
    tokens_input: int               # Direct: extracted from usage.input_tokens
    tokens_output: int              # Direct: extracted from usage.output_tokens
    tokens_cache_read: int          # Direct: extracted from usage.cache_read_input_tokens
    tokens_cache_created: int       # Direct: extracted from usage.cache_creation_input_tokens
    tools_used: list[str]           # Direct: extracted from system.tools
    model: str                      # Direct: extracted from assistant.message.model
    ttft_ms: float                  # Direct: extracted from assistant.message.ttftMs
    num_turns: int                  # Direct: extracted from result.num_turns

    # Session & error tracking
    session_id: str | None = None
    error_type: str | None = None

    # Full metadata for advanced users
    metadata: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse content as JSON."""
        import json
        return json.loads(self.content)

    def __str__(self) -> str:
        """String representation returns just the content."""
        return self.content
```

**Note**: All metrics are now direct fields extracted from the streaming events during response construction, eliminating the need for nested property access.

### Step 2: Update Module Exports (`ask_claude/__init__.py`)

**Remove from __all__ list (lines 27-44):**
```python
❌ "ask_claude_json",
❌ "ask_claude_streaming",
```

**Remove from imports (lines 8-23):**
```python
❌ ask_claude_json,
❌ ask_claude_streaming,
```

### Step 3: Update All Example Files

**`examples/getting_started.py`:**
- Line 47: Change `ask_claude_json()` → `ask_claude(..., output="json")`
- Line 110: Change `ask_claude_streaming()` → `ask_claude(..., output="text-stream")`

**`examples/production_example.py`:**
- Line 69: Change `ask_claude_json()` → `ask_claude(..., output="json")`
- Line 192: Change `ask_claude_streaming()` → `ask_claude(..., output="text-stream")`

**`examples/mcp_example.py`:**
- Lines 48, 123, 153, 270, 282: Change `wrapper.ask()` → `wrapper.run()`

**`examples/cache_configuration_example.py`:**
- Line 173: Change `wrapper.ask()` → `wrapper.run()`

### Step 4: Update All Test Files

**`tests/test_claude_code_wrapper.py`:**
- Lines 539, 547: Change `ask_claude_json()` → `ask_claude(..., output="json")`
- Lines 551, 562: Change `ask_claude_streaming()` → `ask_claude(..., output="text-stream")`
- Lines 385, 397, 484, 488, 605: Change `wrapper.ask()` → `wrapper.run()`
- Lines 418, 429, 890: Change `wrapper.ask_json()` → `wrapper.run(...).json()`

**`tests/test_session_management.py`:**
- Lines 312, 320: Change `continue_claude()` → `ask_claude("", session="continue")`
- Lines 325, 333: Change `resume_claude()` → `ask_claude(..., session="session_id")`
- Lines 340, 349: Change `ask_claude_with_session()` → `ask_claude(..., session="session_id")`

### Step 5: Update Documentation

**`README.md`:**
- Lines 76-77: Update streaming example
- Update all code examples to use unified API

**`docs/api-reference.md`:**
- Remove sections for deprecated functions
- Add comprehensive documentation for new unified `ask_claude()`

**`docs/session-management.md`:**
- Update all session examples to use `session` parameter
- Remove documentation for separate session functions

**`CHANGELOG.md`:**
- Update feature references to reflect new unified API

### Step 6: Update Test Scripts

**`simple_test.py` and `test_claude_functions.py`:**
- Update all function calls to use new parameter-based API
- Update imports to only import `ask_claude`
- Test all parameter combinations (output, stream, debug, session)

## Approval System Analysis & Improvements

### Current State: ✅ Well-Designed
The `ask_claude/approval/` directory is well-architected and doesn't need major changes:

- **Good Integration**: Works seamlessly with wrapper through `mcp_auto_approval` config
- **Proper Patterns**: Uses factory pattern, abstract base classes, clean exports
- **CLI Support**: Full command-line integration

### Minor Improvements Needed:

#### 1. Fix Function Naming (server.py:133)
```python
# CHANGE:
❌ async def permissions__approve(tool_name: str, input: dict, reason: str = "") -> dict:

# TO:
✅ async def permissions_approve(tool_name: str, input: dict, reason: str = "") -> dict:
```

#### 2. Standardize Configuration Keys
```python
# CHANGE in strategies.py:
❌ strategy_type = config.get("type", "allowlist")

# TO:
✅ strategy_type = config.get("strategy", "allowlist")  # Consistent with main config
```

#### 3. Enhanced Factory Function
```python
# IMPROVE create_approval_strategy():
def create_approval_strategy(
    strategy_type: str,
    config: dict,
    logger: logging.Logger | None = None  # NEW: Add logger support
) -> ApprovalStrategy:
```

### API Changes Impact: ✅ No Breaking Changes
The approval system doesn't use any of the functions we're removing:
- Doesn't call `ask_claude_json()` or `ask_claude_streaming()`
- Integrates through config, not function calls
- No changes needed for our API cleanup

### Future Enhancement Opportunity:
Consider adding approval convenience functions (following our new unified pattern):
```python
def approve_tool(tool_name: str, strategy: str = "allowlist", **kwargs) -> bool:
    """Convenience function for single tool approval."""

def create_approval_config(strategy: str, **kwargs) -> dict:
    """Convenience function for creating approval configurations."""
```

## Benefits After Implementation:

1. **Single Import**: `from ask_claude import ask_claude`
2. **Clear Intent**: All parameters are explicit and typed
3. **Consistent Returns**: ClaudeCodeResponse or Iterator, no confusion
4. **Direct Metadata Access**: `response.cost`, `response.tokens_cache_read` instead of nested access
5. **Cache Optimization**: Direct access to `tokens_cache_read/created` for efficiency monitoring
6. **AI/ML Pipeline Ready**: All metrics needed for cost, performance, and capacity monitoring
7. **Future Extensible**: Easy to add new output formats or session types
8. **No Function Proliferation**: One function handles all use cases
9. **Approval System Consistency**: Minor naming fixes for complete consistency
