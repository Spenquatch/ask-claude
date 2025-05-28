#!/usr/bin/env python3
"""
Example: Configuring cache TTL for different use cases.

This example shows how to configure the response cache TTL based on your needs,
from short 5-minute caches to long 60-minute caches.
"""

import time
from typing import Any

from ask_claude import ClaudeCodeConfig, ClaudeCodeWrapper
from ask_claude.wrapper import ClaudeCodeResponse


def demo_cache_configurations() -> dict[str, ClaudeCodeWrapper]:
    """Demonstrate different cache TTL configurations."""

    print("=== Cache Configuration Examples ===\n")

    # Example 1: Short TTL for frequently changing data
    print("1. Short TTL (5 minutes) - For dynamic content:")
    config_short = ClaudeCodeConfig(cache_responses=True, cache_ttl=300.0)  # 5 minutes
    wrapper_short = ClaudeCodeWrapper(config_short)
    print(f"   Cache enabled: {config_short.cache_responses}")
    print(
        f"   Cache TTL: {config_short.cache_ttl} seconds ({config_short.cache_ttl/60:.0f} minutes)\n"
    )

    # Example 2: Medium TTL (default) for balanced usage
    print("2. Medium TTL (30 minutes) - Default configuration:")
    config_medium = ClaudeCodeConfig(
        cache_responses=True
        # cache_ttl defaults to 1800.0 (30 minutes)
    )
    wrapper_medium = ClaudeCodeWrapper(config_medium)
    print(f"   Cache enabled: {config_medium.cache_responses}")
    print(
        f"   Cache TTL: {config_medium.cache_ttl} seconds ({config_medium.cache_ttl/60:.0f} minutes)\n"
    )

    # Example 3: Long TTL for stable content
    print("3. Long TTL (60 minutes) - For stable/reference content:")
    config_long = ClaudeCodeConfig(
        cache_responses=True,
        cache_ttl=3600.0,  # 60 minutes (matching Anthropic's max prompt cache)
    )
    wrapper_long = ClaudeCodeWrapper(config_long)
    print(f"   Cache enabled: {config_long.cache_responses}")
    print(
        f"   Cache TTL: {config_long.cache_ttl} seconds ({config_long.cache_ttl/60:.0f} minutes)\n"
    )

    # Example 4: Very long TTL for documentation
    print("4. Very Long TTL (24 hours) - For documentation/reference:")
    config_docs = ClaudeCodeConfig(cache_responses=True, cache_ttl=86400.0)  # 24 hours
    wrapper_docs = ClaudeCodeWrapper(config_docs)
    print(f"   Cache enabled: {config_docs.cache_responses}")
    print(
        f"   Cache TTL: {config_docs.cache_ttl} seconds ({config_docs.cache_ttl/3600:.0f} hours)\n"
    )

    # Example 5: No caching for real-time data
    print("5. No Caching - For real-time/personalized content:")
    config_realtime = ClaudeCodeConfig(cache_responses=False)
    wrapper_realtime = ClaudeCodeWrapper(config_realtime)
    print(f"   Cache enabled: {config_realtime.cache_responses}")
    print("   Cache TTL: N/A (caching disabled)\n")

    return {
        "dynamic": wrapper_short,
        "balanced": wrapper_medium,
        "stable": wrapper_long,
        "documentation": wrapper_docs,
        "realtime": wrapper_realtime,
    }


def demo_cache_behavior() -> None:
    """Demonstrate cache hit/miss behavior."""
    print("\n=== Cache Behavior Demo ===\n")

    # Create wrapper with 10-second cache for quick demo
    config = ClaudeCodeConfig(
        cache_responses=True,
        cache_ttl=10.0,  # 10 seconds for demo
    )
    wrapper = ClaudeCodeWrapper(config)

    query = "What is 2 + 2?"

    # First call - cache miss
    print("1. First call (cache miss):")
    start = time.time()
    response1 = wrapper.run(query)
    duration1 = time.time() - start
    metrics1 = wrapper.get_metrics()
    print(f"   Duration: {duration1:.3f}s")
    print(
        f"   Cache hits: {metrics1['cache_hits']}, misses: {metrics1['cache_misses']}"
    )
    print(f"   Response: {response1.content[:50]}...\n")

    # Second call - cache hit
    print("2. Second call (cache hit):")
    start = time.time()
    response2 = wrapper.run(query)
    duration2 = time.time() - start
    metrics2 = wrapper.get_metrics()
    print(f"   Duration: {duration2:.3f}s (much faster!)")
    print(
        f"   Cache hits: {metrics2['cache_hits']}, misses: {metrics2['cache_misses']}"
    )
    print(f"   Cache hit rate: {metrics2['cache_hit_rate']:.1%}")
    print(f"   Response identical: {response1.content == response2.content}\n")

    # Wait for cache expiry
    print("3. Waiting 11 seconds for cache to expire...")
    time.sleep(11)

    # Third call - cache miss again
    print("\n4. Third call after expiry (cache miss):")
    start = time.time()
    response3 = wrapper.run(query)
    duration3 = time.time() - start
    metrics3 = wrapper.get_metrics()
    print(f"   Duration: {duration3:.3f}s")
    print(
        f"   Cache hits: {metrics3['cache_hits']}, misses: {metrics3['cache_misses']}"
    )
    print(f"   Cache hit rate: {metrics3['cache_hit_rate']:.1%}\n")


def demo_adaptive_caching() -> None:
    """Demonstrate adaptive caching based on query type."""
    print("\n=== Adaptive Caching Demo ===\n")

    class AdaptiveCacheWrapper:
        """Wrapper that adjusts cache TTL based on query type."""

        def __init__(self) -> None:
            # Different wrappers for different cache strategies
            self.wrappers = {
                "definition": ClaudeCodeWrapper(
                    ClaudeCodeConfig(
                        cache_responses=True,
                        cache_ttl=3600.0,  # 1 hour for definitions
                    )
                ),
                "calculation": ClaudeCodeWrapper(
                    ClaudeCodeConfig(
                        cache_responses=True,
                        cache_ttl=1800.0,  # 30 minutes for calculations
                    )
                ),
                "realtime": ClaudeCodeWrapper(
                    ClaudeCodeConfig(cache_responses=False)  # No cache for real-time
                ),
                "default": ClaudeCodeWrapper(
                    ClaudeCodeConfig(
                        cache_responses=True,
                        cache_ttl=900.0,  # 15 minutes default
                    )
                ),
            }

        def ask(
            self, query: str, query_type: str = "default", **kwargs: Any
        ) -> ClaudeCodeResponse:
            """Route query to appropriate wrapper based on type."""
            wrapper = self.wrappers.get(query_type, self.wrappers["default"])
            print(f"Using {query_type} cache strategy")
            return wrapper.ask(query, **kwargs)

    adaptive = AdaptiveCacheWrapper()

    # Different query types get different cache treatment
    queries = [
        ("What is Python?", "definition"),
        ("Calculate the fibonacci sequence", "calculation"),
        ("What's the current time?", "realtime"),
        ("Explain recursion", "default"),
    ]

    for query, query_type in queries:
        print(f"\nQuery: '{query}'")
        print(f"Type: {query_type}")
        response = adaptive.ask(query, query_type)
        print(f"Response: {response.content[:60]}...")


if __name__ == "__main__":
    # Show different configurations
    wrappers = demo_cache_configurations()

    # Demonstrate cache behavior
    demo_cache_behavior()

    # Show adaptive caching
    demo_adaptive_caching()

    print("\n=== Summary ===")
    print("• Default cache TTL is now 30 minutes (1800 seconds)")
    print("• Can be configured from 0 to any duration")
    print("• Recommended: 5-60 minutes based on use case")
    print("• Complements Anthropic's server-side prompt caching")
    print("• Use cache_responses=False for real-time data")
