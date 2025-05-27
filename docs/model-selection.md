# Model Selection Guide

The Claude Code Wrapper supports dynamic model selection, allowing you to choose between different Claude models and customize generation parameters for optimal results.

## Available Models

You can specify models using either shorthand names or full model IDs:

### Shorthand Names
- `opus` - Claude 3 Opus (most capable)
- `sonnet` - Claude 3.5 Sonnet (balanced)
- `haiku` - Claude 3 Haiku (fastest)

### Full Model Names
- `claude-3-opus-latest`
- `claude-3-5-sonnet-latest`
- `claude-3-haiku-latest`

## Basic Usage

### Using Model Selection in Code

```python
from claude_code_wrapper import ClaudeCodeWrapper

# Initialize wrapper
wrapper = ClaudeCodeWrapper()

# Use a specific model
response = wrapper.ask("Explain quantum computing", model="haiku")

# Use with temperature control
response = wrapper.ask(
    "Write a creative story about AI",
    model="opus",
    temperature=0.8
)
```

### Configuration-Based Model Selection

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Set model in configuration
config = ClaudeCodeConfig(
    model="sonnet",
    temperature=0.7,
    max_tokens=2000
)
wrapper = ClaudeCodeWrapper(config=config)
```

## Generation Parameters

### Temperature (0.0 - 1.0)
Controls randomness in responses:
- `0.0` - Most deterministic
- `0.7` - Balanced creativity
- `1.0` - Maximum creativity

```python
# Precise, technical response
response = wrapper.ask("Explain TCP/IP", temperature=0.1)

# Creative response
response = wrapper.ask("Write a poem", temperature=0.9)
```

### Max Tokens
Limits response length:

```python
# Short response
response = wrapper.ask("Summarize this", max_tokens=100)

# Detailed response
response = wrapper.ask("Explain in detail", max_tokens=4000)
```

### Top-p (Nucleus Sampling)
Alternative to temperature for controlling randomness:

```python
response = wrapper.ask(
    "Generate ideas",
    top_p=0.9,  # Consider top 90% of probability mass
    temperature=None  # Don't use both temperature and top_p
)
```

### Stop Sequences
Define custom stopping points:

```python
response = wrapper.ask(
    "List items",
    stop_sequences=["\\n\\n", "END", "</list>"]
)
```

## Model Selection Strategies

### 1. Task-Based Selection

```python
def select_model_for_task(task_type):
    """Select optimal model based on task."""
    models = {
        "code_generation": ("opus", 0.1),    # Precise
        "creative_writing": ("opus", 0.8),   # Creative
        "quick_answers": ("haiku", 0.3),     # Fast
        "analysis": ("sonnet", 0.2),         # Balanced
    }
    
    model, temp = models.get(task_type, ("sonnet", 0.5))
    return {"model": model, "temperature": temp}

# Usage
params = select_model_for_task("code_generation")
response = wrapper.ask("Write a Python function", **params)
```

### 2. Cost-Optimized Selection

```python
def cost_optimized_query(query, max_cost_tier="medium"):
    """Select model based on query complexity and cost constraints."""
    query_length = len(query)
    
    if max_cost_tier == "low" or query_length < 100:
        return wrapper.ask(query, model="haiku")
    elif max_cost_tier == "medium" or query_length < 500:
        return wrapper.ask(query, model="sonnet")
    else:
        return wrapper.ask(query, model="opus")
```

### 3. Adaptive Model Selection

```python
class AdaptiveWrapper:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.performance_stats = {}
    
    def ask_adaptive(self, query, quality_threshold=0.8):
        """Start with fast model, upgrade if needed."""
        # Try Haiku first
        response = self.wrapper.ask(query, model="haiku")
        
        # Check if response meets quality threshold
        if self._assess_quality(response) < quality_threshold:
            # Upgrade to Sonnet
            response = self.wrapper.ask(query, model="sonnet")
            
            if self._assess_quality(response) < quality_threshold:
                # Final upgrade to Opus
                response = self.wrapper.ask(query, model="opus")
        
        return response
    
    def _assess_quality(self, response):
        # Implement quality assessment logic
        # e.g., check response length, completeness, etc.
        return 0.9  # Placeholder
```

## Advanced Examples

### Dynamic Parameter Adjustment

```python
def dynamic_query(query, context_size):
    """Adjust parameters based on context."""
    params = {
        "model": "sonnet",  # Default
        "temperature": 0.5
    }
    
    # Large context needs more capable model
    if context_size > 10000:
        params["model"] = "opus"
        params["max_tokens"] = 4000
    elif context_size > 5000:
        params["model"] = "sonnet"
        params["max_tokens"] = 2000
    else:
        params["model"] = "haiku"
        params["max_tokens"] = 1000
    
    return wrapper.ask(query, **params)
```

### Streaming with Model Selection

```python
# Stream with specific model
for chunk in wrapper.stream("Tell me a story", model="opus", temperature=0.8):
    print(chunk, end="", flush=True)
```

### Session-Based Model Switching

```python
# Start session with one model
with wrapper.session(model="haiku") as session:
    # Quick questions with Haiku
    session.ask("What is Python?")
    
    # Switch to Opus for complex task
    session.ask("Write complex algorithm", model="opus")
    
    # Back to Haiku
    session.ask("Summarize the algorithm", model="haiku")
```

## Best Practices

1. **Start with Haiku**: For most queries, try Haiku first
2. **Use Opus for complexity**: Complex reasoning, code generation
3. **Temperature guidelines**:
   - Code/Technical: 0.0 - 0.3
   - General: 0.3 - 0.7
   - Creative: 0.7 - 1.0
4. **Monitor costs**: Track model usage for cost optimization
5. **Test parameters**: Different tasks benefit from different settings

## CLI Integration

Model selection also works with the CLI tool:

```bash
# Using shorthand
claude-code "Explain arrays" --model haiku

# With temperature
claude-code "Write a function" --model opus --temperature 0.1

# Full configuration
claude-code "Complex task" \
  --model sonnet \
  --temperature 0.5 \
  --max-tokens 2000 \
  --top-p 0.9
```

## Error Handling

```python
try:
    response = wrapper.ask("Query", model="invalid-model")
except ClaudeCodeConfigurationError as e:
    print(f"Invalid model: {e}")
    # Fallback to default
    response = wrapper.ask("Query")
```

## Performance Considerations

- **Haiku**: ~2-3x faster than Opus
- **Sonnet**: ~1.5x faster than Opus
- **Opus**: Highest quality, slowest

Choose models based on your latency requirements and quality needs.