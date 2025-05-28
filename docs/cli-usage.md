# CLI Usage Guide

Ask Claude - Claude Code SDK Wrapper includes a comprehensive command-line interface (CLI) for easy interaction with Claude Code.

## Quick Start

### Development (with Poetry)
```bash
# Ask a simple question
poetry run python -m ask_claude.cli ask "What is Python?"

# Get JSON response with metadata
poetry run python -m ask_claude.cli ask "Explain machine learning" --format json

# Start interactive session
poetry run python -m ask_claude.cli session --interactive

# Stream a response
poetry run python -m ask_claude.cli stream "Write a tutorial on Python"

# Check system health
poetry run python -m ask_claude.cli health
```

### Production (after installation)
```bash
# Ask a simple question
ask-claude ask "What is Python?"

# Get JSON response with metadata
ask-claude ask "Explain machine learning" --format json

# Start interactive session
ask-claude session --interactive

# Stream a response
ask-claude stream "Write a tutorial on Python"

# Check system health
ask-claude health
```

## Commands

### ask

Execute a single query and return the response.

```bash
# Development
poetry run python -m ask_claude.cli ask <query> [options]

# Production
ask-claude ask <query> [options]
```

**Arguments:**
- `query`: The question or prompt to send to Claude Code

**Options:**
- `--format {text,json}`: Output format (default: text)
- `--timeout SECONDS`: Request timeout in seconds
- `--max-turns NUMBER`: Maximum conversation turns
- `--session-id ID`: Resume specific session
- `--continue`: Continue last session
- `--show-metadata`: Show response metadata

**Examples:**
```bash
# Basic question
poetry run python -m ask_claude.cli ask "What is machine learning?"
# or: ask-claude ask "What is machine learning?"

# JSON format with metadata
poetry run python -m ask_claude.cli ask "Generate Python code" --format json --show-metadata
# or: ask-claude ask "Generate Python code" --format json --show-metadata

# With custom timeout
poetry run python -m ask_claude.cli ask "Complex analysis" --timeout 120
# or: ask-claude ask "Complex analysis" --timeout 120

# Resume specific session
poetry run python -m ask_claude.cli ask "Continue our discussion" --session-id "abc123"
# or: ask-claude ask "Continue our discussion" --session-id "abc123"

# Continue last session
poetry run python -m ask_claude.cli ask "What was my last question?" --continue
# or: ask-claude ask "What was my last question?" --continue
```

### stream

Execute a query with streaming response output.

```bash
# Development
poetry run python -m ask_claude.cli stream <query> [options]

# Production
ask-claude stream <query> [options]
```

**Arguments:**
- `query`: The question or prompt to stream from Claude Code

**Options:**
- `--timeout SECONDS`: Request timeout in seconds
- `--show-stats`: Show streaming statistics at the end

**Examples:**
```bash
# Basic streaming
poetry run python -m ask_claude.cli stream "Write a long story about AI"
# or: ask-claude stream "Write a long story about AI"

# With statistics
poetry run python -m ask_claude.cli stream "Explain quantum computing" --show-stats
# or: ask-claude stream "Explain quantum computing" --show-stats

# Custom timeout
poetry run python -m ask_claude.cli stream "Generate comprehensive tutorial" --timeout 300
# or: ask-claude stream "Generate comprehensive tutorial" --timeout 300
```

### session

Start an interactive session for multi-turn conversations.

```bash
# Development
poetry run python -m ask_claude.cli session [options]

# Production
ask-claude session [options]
```

**Options:**
- `--interactive, -i`: Enable interactive mode (required)
- `--max-turns NUMBER`: Maximum turns in session

**Interactive Commands:**
- `help`: Show session commands
- `history`: Show conversation history
- `clear`: Clear session history
- `exit` or `quit`: End session

**Examples:**
```bash
# Start interactive session
poetry run python -m ask_claude.cli session --interactive
# or: ask-claude session --interactive

# With turn limit
poetry run python -m ask_claude.cli session --interactive --max-turns 10
# or: ask-claude session --interactive --max-turns 10
```

**Interactive Session Example:**
```
$ ask-claude session --interactive
🔄 Starting interactive session...
💡 Type 'exit', 'quit', or Ctrl+C to end session
💡 Type 'help' for commands
--------------------------------------------------

[1] ❓ You: What is Python?
🤖 Claude: Python is a high-level, interpreted programming language...

[2] ❓ You: Can you show me an example?
🤖 Claude: Here's a simple Python example:

def greet(name):
    return f"Hello, {name}!"

print(greet("World"))

[3] ❓ You: history
📚 Session History (2 exchanges):
   1. ✅ What is Python?...
   2. ✅ Can you show me an example?...

[4] ❓ You: exit
👋 Session ended

🏁 Session completed with 2 exchanges
```

### health

Check the health and status of the Claude Code wrapper.

```bash
# Development
poetry run python -m ask_claude.cli health

# Production
ask-claude health
```

**Output includes:**
- Basic functionality test
- Response time measurement
- Error detection
- Streaming capability test
- Overall health status

**Example Output:**
```
🏥 Claude Code Wrapper Health Check
----------------------------------------
✅ Basic functionality: Working
⏱️  Response time: 2.34s
📝 Response: What is 2+2? The answer is 4...
🌊 Testing streaming...
✅ Streaming: 3 events received
📊 Metrics: {'total_requests': 1, 'error_count': 0}

🎯 Overall Status: Healthy
```

### benchmark

Run performance benchmarks to test wrapper performance.

```bash
# Development
poetry run python -m ask_claude.cli benchmark [options]

# Production
ask-claude benchmark [options]
```

**Options:**
- `--queries FILE`: File containing queries to benchmark (one per line)
- `--iterations NUMBER`: Number of iterations per query (default: 3)

**Examples:**
```bash
# Default benchmark
poetry run python -m ask_claude.cli benchmark
# or: ask-claude benchmark

# Custom iterations
poetry run python -m ask_claude.cli benchmark --iterations 5
# or: ask-claude benchmark --iterations 5

# Custom query file
poetry run python -m ask_claude.cli benchmark --queries my_queries.txt --iterations 10
# or: ask-claude benchmark --queries my_queries.txt --iterations 10
```

**Example Output:**
```
🏃 Running performance benchmark (3 iterations)
--------------------------------------------------
🔄 Query 1/4: What is 2+2?...
   ⏱️  Avg: 1.234s, Min: 1.100s, Max: 1.456s
🔄 Query 2/4: Explain Python in one sentence...
   ⏱️  Avg: 2.567s, Min: 2.234s, Max: 2.890s

📊 Benchmark Summary:
------------------------------
Overall Average Time: 1.901s
Overall Error Rate: 0.0%
Fastest Query: 1.234s
Slowest Query: 2.567s
```

## Global Options

Available for all commands:

- `--config, -c FILE`: Configuration file path
- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Quiet mode (minimal output)

**Examples:**
```bash
# Use custom config
poetry run python -m ask_claude.cli --config prod_config.json ask "What is AI?"
# or: ask-claude --config prod_config.json ask "What is AI?"

# Verbose mode
poetry run python -m ask_claude.cli --verbose ask "Debug this query"
# or: ask-claude --verbose ask "Debug this query"

# Quiet mode
poetry run python -m ask_claude.cli --quiet ask "Silent query"
# or: ask-claude --quiet ask "Silent query"
```

## Configuration File

Create a JSON configuration file for consistent settings:

```json
{
  "claude_binary": "claude",
  "timeout": 60.0,
  "max_retries": 3,
  "verbose": false,
  "system_prompt": "You are a helpful assistant.",
  "enable_metrics": true
}
```

Use with CLI:
```bash
# Development
poetry run python -m ask_claude.cli --config config.json ask "What is Python?"

# Production
ask-claude --config config.json ask "What is Python?"
```

## Output Formats

### Text Format (Default)

```bash
$ ask-claude ask "What is 2+2?"
4
```

### JSON Format

```bash
$ ask-claude ask "What is 2+2?" --format json
{
  "content": "4",
  "session_id": "abc123",
  "cost_usd": 0.001234,
  "duration_ms": 1500,
  "is_error": false
}
```

### With Metadata

```bash
$ ask-claude ask "What is 2+2?" --format json --show-metadata

4

📊 Metadata:
   Session ID: abc123
   Is Error: False
   Execution Time: 1.234s
   Cost: $0.001234
   Duration: 1500ms
   Turns: 1
```

## Error Handling

The CLI provides comprehensive error handling and informative error messages.

### Validation Errors
```bash
$ ask-claude ask ""
❌ Error: Query cannot be empty
```

### Timeout Errors
```bash
$ ask-claude ask "Complex query" --timeout 0.1
❌ Timeout Error: Claude Code execution timed out after 0.1s
```

### Process Errors
```bash
$ ask-claude ask "Invalid query"
❌ Process Error: Claude Code process failed with return code 1
   Details: Invalid command syntax
```

### Configuration Errors
```bash
$ ask-claude --config invalid.json ask "Query"
❌ Configuration Error: Config file not found: invalid.json
```

## Streaming Output

The streaming command provides real-time output with event handling:

```bash
$ ask-claude stream "Count from 1 to 5"
🌊 Starting stream...
1
2
3
4
5

📊 Stream Stats:
   Events: 8
   Errors: 0
   Content: 9 chars
```

### Streaming Event Types

- `init`: Stream initialization
- `message`: Content chunks
- `tool_use`: Tool execution
- `tool_result`: Tool results
- `result`: Final completion
- `error`: Error events
- `parse_error`: JSON parsing errors

## Advanced Usage

### Batch Processing with Scripts

Create a query file `queries.txt`:
```
What is Python?
Explain machine learning
Show me a sorting algorithm
What are design patterns?
```

Process all queries:
```bash
# Simple batch processing
for query in $(cat queries.txt); do
    echo "Query: $query"
    ask-claude ask "$query"
    echo "---"
done

# With JSON output for processing
ask-claude benchmark --queries queries.txt --iterations 1
```

### Chaining Commands

```bash
# Get session ID and reuse it
SESSION_ID=$(ask-claude ask "Start conversation" --format json | jq -r '.session_id')
ask-claude ask "Continue conversation" --session-id "$SESSION_ID"
```

### Health Monitoring

```bash
# Simple health check
if ask-claude health >/dev/null 2>&1; then
    echo "Service is healthy"
else
    echo "Service has issues"
fi

# Detailed monitoring
ask-claude health | grep "Overall Status"
```

### Performance Monitoring

```bash
# Regular performance checks
ask-claude benchmark --iterations 1 > performance.log
cat performance.log | grep "Overall Average Time"
```

## Environment Variables

The CLI respects environment variables:

```bash
export CLAUDE_BINARY="/usr/local/bin/claude"
export CLAUDE_TIMEOUT="60"
export CLAUDE_VERBOSE="true"

ask-claude ask "Query with env vars"
```

## Automation and Integration

### CI/CD Integration

```bash
#!/bin/bash
# test_claude_wrapper.sh

echo "Testing Claude wrapper..."

# Health check
if ! ask-claude health; then
    echo "Health check failed"
    exit 1
fi

# Basic functionality test
RESPONSE=$(ask-claude ask "What is 2+2?" --format json)
if echo "$RESPONSE" | jq -e '.content == "4"' > /dev/null; then
    echo "Basic test passed"
else
    echo "Basic test failed"
    exit 1
fi

echo "All tests passed"
```

### Monitoring Scripts

```bash
#!/bin/bash
# monitor_claude.sh

while true; do
    if ask-claude health | grep -q "Healthy"; then
        echo "$(date): Service healthy"
    else
        echo "$(date): Service unhealthy" >&2
        # Send alert
    fi
    sleep 60
done
```

### Log Analysis

```bash
# Analyze CLI usage
ask-claude --verbose ask "Test query" 2>&1 | \
    grep -E "(INFO|ERROR|WARNING)" | \
    tee claude_wrapper.log
```

## Troubleshooting

### Common Issues

1. **"Claude binary not found"**
   ```bash
   # Check Claude installation
   which claude
   claude --version

   # Use full path if needed
   ask-claude --config config.json ask "Query"
   # where config.json contains: {"claude_binary": "/full/path/to/claude"}
   ```

2. **Permission denied**
   ```bash
   # If installed via Poetry, ensure Poetry environment is activated
   poetry shell
   ask-claude ask "Test"
   ```

3. **Import errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"

   # For development, ensure Poetry dependencies are installed
   poetry install
   poetry run python -m ask_claude.cli ask "Test"
   ```

4. **JSON parsing errors**
   ```bash
   # Use text format for debugging
   ask-claude ask "Query" --format text --verbose
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
ask-claude --verbose ask "Debug query"
```

This will show detailed information about:
- Configuration loading
- Command construction
- Process execution
- Response parsing
- Error handling

The CLI tool provides a complete interface to Ask Claude - Claude Code SDK Wrapper with comprehensive error handling, flexible configuration, and powerful features for both interactive use and automation.
