#!/usr/bin/env python3
"""
Getting Started with Claude Code SDK Wrapper

Simple test script to verify your setup and demonstrate basic functionality
with proper error handling that won't crash on failures.
"""

import logging

from ask_claude.wrapper import (
    ClaudeCodeConfig,
    ClaudeCodeWrapper,
    OutputFormat,
    ask_claude,
    ask_claude_json,
)

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_basic_functionality() -> None:
    """Test basic wrapper functionality with simple queries."""
    print("🔧 Testing Basic Functionality")
    print("-" * 40)

    # Test 1: Simple text query
    print("1. Testing simple text query...")
    try:
        response = ask_claude("What is 2+2?")
        print(f"   ✅ Success: {response.content}")
        print(f"   📊 Session ID: {response.session_id}")
        print(f"   ⚠️  Is Error: {response.is_error}")

        if response.is_error:
            print(f"   ❌ Error Type: {response.error_type}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print()

    # Test 2: JSON format query
    print("2. Testing JSON format query...")
    try:
        response = ask_claude_json("What is Python? One sentence only.")
        print(f"   ✅ Content: {response.content}")
        print(f"   📊 Session ID: {response.session_id}")
        print(f"   💰 Cost: ${response.metrics.cost_usd:.6f}")
        print(f"   ⏱️  Duration: {response.metrics.duration_ms}ms")
        print(f"   🔄 Turns: {response.metrics.num_turns}")
        print(f"   ⚠️  Is Error: {response.is_error}")

        if response.is_error:
            print(f"   ❌ Error: {response.error_type} - {response.error_subtype}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")


def test_advanced_features() -> None:
    """Test advanced features with proper error handling."""
    print("\n🚀 Testing Advanced Features")
    print("-" * 40)

    try:
        # Create wrapper with specific configuration
        config = ClaudeCodeConfig(
            timeout=30.0,
            max_retries=1,  # Reduced for faster testing
            verbose=True,
            enable_metrics=True,
        )

        wrapper = ClaudeCodeWrapper(config)

        print("3. Testing configured wrapper...")
        response = wrapper.run(
            "Explain what a REST API is in exactly 2 sentences.",
            output_format=OutputFormat.JSON,
        )

        print(f"   ✅ Response length: {len(response.content)} characters")
        print(f"   📊 Execution time: {response.execution_time:.2f}s")
        print(f"   💰 Total cost: ${response.metrics.total_cost:.6f}")

        # Get wrapper metrics
        metrics = wrapper.get_metrics()
        print(f"   📈 Wrapper metrics: {metrics}")

    except Exception as e:
        print(f"   ❌ Advanced features failed: {e}")


def test_streaming_safely() -> None:
    """Test streaming with comprehensive error handling."""
    print("\n🌊 Testing Streaming (Safe Mode)")
    print("-" * 40)

    print("4. Testing streaming response...")
    try:
        from ask_claude import ask_claude_streaming

        events_received = 0
        content_parts = []
        errors_encountered = 0

        # Use a simple query that should stream quickly
        for event in ask_claude_streaming("Count from 1 to 3"):
            events_received += 1

            event_type = event.get("type", "unknown")
            print(f"   📨 Event {events_received}: {event_type}")

            if event_type == "error":
                errors_encountered += 1
                print(f"      ❌ Error: {event.get('message', 'Unknown error')}")

            elif event_type == "message":
                content = event.get("content", "")
                if content:
                    content_parts.append(content)
                    print(f"      💬 Content: {content}")

            elif event_type == "init":
                session_id = event.get("session_id", "no-session")
                print(f"      🚀 Started: {session_id}")

            elif event_type == "result":
                status = event.get("status", "unknown")
                print(f"      🏁 Completed: {status}")

            # Safety limit
            if events_received > 20:
                print("      ⚠️  Safety limit reached, stopping")
                break

        # Summary
        full_content = "".join(content_parts)
        print("   📊 Summary:")
        print(f"      Events: {events_received}")
        print(f"      Errors: {errors_encountered}")
        print(f"      Content: {len(full_content)} chars")

        if full_content:
            print(f"      Preview: {full_content[:100]}...")

    except Exception as e:
        print(f"   ❌ Streaming failed: {e}")
        print("   ℹ️  This is normal if Claude Code isn't configured for streaming")


def test_error_scenarios() -> None:
    """Test error handling scenarios."""
    print("\n⚠️  Testing Error Handling")
    print("-" * 40)

    # Test empty query validation
    print("5. Testing input validation...")
    try:
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)
        response = wrapper.run("")  # Empty query
        print(f"   ❌ Validation failed - got response: {response.content}")
    except Exception as e:
        print(f"   ✅ Validation worked: {type(e).__name__}")

    # Test configuration validation
    print("6. Testing configuration validation...")
    try:
        bad_config = ClaudeCodeConfig(timeout=-1.0)
        print("   ❌ Config validation failed - accepted negative timeout")
    except Exception as e:
        print(f"   ✅ Config validation worked: {type(e).__name__}")


def analyze_claude_code_setup() -> None:
    """Analyze Claude Code binary setup."""
    print("\n🔍 Analyzing Claude Code Setup")
    print("-" * 40)

    import subprocess

    # Test if claude binary exists
    print("7. Checking Claude Code binary...")
    try:
        result = subprocess.run(
            ["claude", "--help"], capture_output=True, text=True, timeout=5
        )
        print(f"   ✅ Claude binary found (exit code: {result.returncode})")

        if result.returncode == 0:
            print("   ✅ Claude binary working correctly")
        else:
            print(f"   ⚠️  Claude binary returned error: {result.stderr}")

    except FileNotFoundError:
        print("   ❌ Claude binary not found in PATH")
        print("   💡 Make sure Claude Code is installed and accessible")

    except subprocess.TimeoutExpired:
        print("   ⚠️  Claude binary timeout (but it exists)")

    except Exception as e:
        print(f"   ❌ Error checking Claude binary: {e}")

    # Test basic claude command
    print("8. Testing basic Claude Code command...")
    try:
        result = subprocess.run(
            ["claude", "--print", "Hello"], capture_output=True, text=True, timeout=10
        )

        print("   📤 Command: claude --print Hello")
        print(f"   📊 Exit code: {result.returncode}")
        print(f"   📝 Output length: {len(result.stdout)} chars")

        if result.stdout:
            print(f"   📄 Output preview: {result.stdout[:100]}...")

        if result.stderr:
            print(f"   ⚠️  Stderr: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("   ⚠️  Command timed out")

    except Exception as e:
        print(f"   ❌ Command failed: {e}")


def main() -> None:
    """Run all tests with comprehensive error handling."""
    print("🧪 Claude Code SDK Wrapper - Getting Started Tests")
    print("=" * 60)

    try:
        # Run all test sections
        analyze_claude_code_setup()
        test_basic_functionality()
        test_advanced_features()
        test_streaming_safely()
        test_error_scenarios()

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")

    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback

        print("📋 Full traceback:")
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print("✅ All tests completed with proper error handling")
    print("⚠️  Any failures are handled gracefully")
    print("🔧 Check the output above for specific issues")
    print("💡 The wrapper will work even if some features fail")
    print("=" * 60)

    # Recommendations
    print("\n💡 NEXT STEPS:")
    print("1. If Claude binary tests fail, ensure Claude Code is installed")
    print("2. If basic queries work, you're ready for production use")
    print("3. If streaming fails, it may need additional Claude Code setup")
    print("4. Run production_example.py for comprehensive demonstrations")
    print("5. Check logs for detailed error information")


if __name__ == "__main__":
    main()
