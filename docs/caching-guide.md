# Caching Guide

## Overview

The Claude Code Wrapper implements **client-side response caching** to reduce API calls and improve performance. This is different from Anthropic's server-side prompt caching feature.

## Types of Caching

### 1. Client-Side Response Caching (Our Implementation)
- Stores complete responses in local memory
- Prevents duplicate API calls for identical queries
- Configurable TTL (Time To Live)
- Zero additional cost

### 2. Server-Side Prompt Caching (Anthropic's Feature)
- Caches conversation context on Anthropic's servers
- Reduces token costs for repeated contexts
- Automatically handled by Claude API
- May have usage restrictions

## Configuration

### Basic Setup

```python
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Enable caching with default 30-minute TTL
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=1800.0  # 30 minutes
)
wrapper = ClaudeCodeWrapper(config)
```

### TTL Recommendations

Choose your cache TTL based on your use case:

```python
# Short TTL (5 minutes) - For frequently changing data
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=300.0
)

# Medium TTL (30 minutes) - Default, good for most use cases
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=1800.0
)

# Long TTL (60 minutes) - For stable queries/documentation
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=3600.0
)

# Very Long TTL (24 hours) - For reference material
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=86400.0
)
```

## Cache Key Components

The cache key is generated from:
- Query text
- Model selection
- Temperature
- Max tokens
- Top-p
- System prompt
- Output format

This ensures different configurations get different cache entries.

## Usage Examples

### Basic Caching

```python
# First call - cache miss
response1 = wrapper.ask("What is Python?")  # Makes API call

# Second call within TTL - cache hit
response2 = wrapper.ask("What is Python?")  # Returns cached response

# Different query - cache miss
response3 = wrapper.ask("What is JavaScript?")  # Makes API call
```

### Model-Specific Caching

```python
# These will have different cache entries
response1 = wrapper.ask("Explain AI", model="opus")
response2 = wrapper.ask("Explain AI", model="sonnet")  # Different cache key
```

### Clear Cache

```python
# Clear all cached responses
wrapper.clear_cache()

# Or close wrapper to clean up everything
wrapper.close()
```

## Performance Benefits

### API Call Reduction
```python
# Without caching: 100 identical queries = 100 API calls
# With caching: 100 identical queries = 1 API call + 99 cache hits
```

### Cost Savings
```python
# Example with 30-minute cache
# If same query asked 10 times per hour:
# - Without cache: 10 API calls
# - With cache: 2 API calls (one per 30 minutes)
# - Savings: 80% reduction
```

## Monitoring Cache Performance

```python
# Check cache metrics
metrics = wrapper.get_metrics()
print(f"Cache hits: {metrics['cache_hits']}")
print(f"Cache misses: {metrics['cache_misses']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

## Best Practices

### 1. **Enable for Repeated Queries**
```python
# Good: Documentation, explanations, reference queries
wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=3600.0  # 1 hour for stable content
))
```

### 2. **Disable for Dynamic Content**
```python
# Good: Real-time data, personalized responses
wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(
    cache_responses=False  # No caching for dynamic content
))
```

### 3. **Adjust TTL by Use Case**
```python
def create_wrapper_for_use_case(use_case: str) -> ClaudeCodeWrapper:
    """Create wrapper with appropriate cache settings."""
    cache_configs = {
        "documentation": (True, 3600.0),    # 1 hour
        "code_generation": (True, 1800.0),  # 30 minutes
        "real_time": (False, 0),            # No caching
        "analysis": (True, 900.0),          # 15 minutes
    }
    
    enabled, ttl = cache_configs.get(use_case, (True, 1800.0))
    return ClaudeCodeWrapper(ClaudeCodeConfig(
        cache_responses=enabled,
        cache_ttl=ttl
    ))
```

### 4. **Session-Aware Caching**
```python
# Cache is query-specific, not session-specific
# Different sessions with same query will share cache
session1 = wrapper.create_session("session-1")
session2 = wrapper.create_session("session-2")

response1 = session1.ask("What is AI?")  # Cache miss
response2 = session2.ask("What is AI?")  # Cache hit (shared)
```

## Advanced Configuration

### Environment-Based Caching

```python
import os

# Production: Longer cache for stability
# Development: Shorter cache for testing
cache_ttl = float(os.getenv("CLAUDE_CACHE_TTL", "1800"))
cache_enabled = os.getenv("CLAUDE_CACHE_ENABLED", "true").lower() == "true"

config = ClaudeCodeConfig(
    cache_responses=cache_enabled,
    cache_ttl=cache_ttl
)
```

### Conditional Caching

```python
class SmartWrapper:
    def __init__(self):
        self.cached_wrapper = ClaudeCodeWrapper(
            ClaudeCodeConfig(cache_responses=True)
        )
        self.uncached_wrapper = ClaudeCodeWrapper(
            ClaudeCodeConfig(cache_responses=False)
        )
    
    def ask(self, query: str, use_cache: bool = True, **kwargs):
        wrapper = self.cached_wrapper if use_cache else self.uncached_wrapper
        return wrapper.ask(query, **kwargs)
```

## Limitations

1. **Memory-based**: Cache is lost when process ends
2. **Not shared**: Each wrapper instance has its own cache
3. **No persistence**: Cache doesn't survive restarts
4. **Size unbounded**: Cache can grow without limit

## Future Enhancements

Potential improvements for production use:
- Redis/Memcached backend for persistence
- LRU (Least Recently Used) eviction
- Cache size limits
- Cross-process cache sharing
- Cache warming strategies