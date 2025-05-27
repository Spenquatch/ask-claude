# Cache Types Comparison

## Client-Side Response Cache vs Anthropic Prompt Cache

### Our Response Cache (Client-Side)
- **What**: Complete response caching in your application
- **Where**: Local memory
- **Duration**: Configurable (default 30 minutes, up to any duration)
- **Cost**: Free (saves API calls)
- **Use Case**: Avoid duplicate API calls for identical queries
- **Control**: Full control via configuration

### Anthropic's Prompt Cache (Server-Side)
- **What**: Context/prompt caching on Anthropic's infrastructure
- **Where**: Anthropic's servers
- **Duration**: Up to 60 minutes (as of late 2024)
- **Cost**: Reduced token costs for cached portions
- **Use Case**: Reuse large contexts/system prompts efficiently
- **Control**: Automatic (handled by Claude API)

## When to Use Each

### Use Our Response Cache When:
- Making identical queries repeatedly
- Building documentation tools
- Creating reference applications
- Optimizing for response time
- Working with stable content

### Anthropic's Prompt Cache Helps When:
- Using large system prompts repeatedly
- Having long conversation contexts
- Building applications with consistent base context
- Optimizing for token costs

## Example Scenario

```python
# Our cache prevents this redundancy:
for user in users:
    # Without cache: 1000 users = 1000 identical API calls
    # With cache: 1000 users = 1 API call
    response = wrapper.ask("What is our refund policy?")
    send_to_user(user, response)

# Anthropic's cache helps with:
# Large context that's reused across different queries
system_prompt = load_10k_token_documentation()
wrapper.ask("Question 1", system_prompt=system_prompt)  # Full cost
wrapper.ask("Question 2", system_prompt=system_prompt)  # Reduced cost
```

## Complementary Benefits

Both caching mechanisms work together:
1. Anthropic's cache reduces token costs for repeated contexts
2. Our cache eliminates API calls entirely for repeated queries
3. Combined: Maximum cost savings and performance

## Configuration Recommendations

```python
# For maximum efficiency, enable our cache with appropriate TTL
config = ClaudeCodeConfig(
    cache_responses=True,
    cache_ttl=1800.0,  # 30 minutes (or up to 3600.0 for 1 hour)
    # Anthropic's prompt caching is automatic, no config needed
)
```

**Note**: Since Anthropic's 60-minute prompt caching is new (announced late 2024), verify availability for your account type. Our client-side caching works regardless of Anthropic's features.