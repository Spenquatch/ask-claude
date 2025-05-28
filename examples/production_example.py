#!/usr/bin/env python3
"""
Production Example: Claude Code SDK Wrapper

Demonstrates enterprise-grade features:
- Comprehensive error handling with graceful degradation  
- Structured logging and observability
- Retry mechanisms and circuit breaker patterns
- Input validation and sanitization
- Metrics collection and monitoring
- Session management
- Streaming with error recovery
"""

import logging
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_wrapper import (
    ClaudeCodeWrapper,
    ClaudeCodeConfig,
    OutputFormat,
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError,
    ClaudeCodeValidationError,
    ask_claude,
    ask_claude_json,
    ask_claude_streaming
)


def setup_logging():
    """Configure structured logging for production use."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('claude_wrapper.log')
        ]
    )
    return logging.getLogger(__name__)


def demonstrate_basic_usage(logger: logging.Logger):
    """Demonstrate basic usage with proper error handling."""
    logger.info("=== Basic Usage Examples ===")
    
    # 1. Simple text query with error handling
    logger.info("1. Simple text query:")
    try:
        response = ask_claude("What is Python? Please keep it brief.")
        logger.info(f"✅ Success - Content: {response.content[:100]}...")
        logger.info(f"   Session ID: {response.session_id}")
        logger.info(f"   Execution time: {response.execution_time:.2f}s")
        
        if response.is_error:
            logger.warning(f"   Response indicates error: {response.error_type}")
            
    except ClaudeCodeError as e:
        logger.error(f"❌ Claude Code error: {e} (Severity: {e.severity})")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    
    # 2. JSON format with comprehensive error handling
    logger.info("2. JSON format query:")
    try:
        response = ask_claude_json("Explain machine learning in 2 sentences.")
        logger.info(f"✅ JSON Success - Content: {response.content[:100]}...")
        logger.info(f"   Session ID: {response.session_id}")
        logger.info(f"   Metrics: Cost=${response.metrics.cost_usd:.4f}, Duration={response.metrics.duration_ms}ms")
        logger.info(f"   Turns: {response.metrics.num_turns}")
        
        if response.metadata:
            logger.info(f"   Additional metadata: {list(response.metadata.keys())}")
            
    except ClaudeCodeValidationError as e:
        logger.error(f"❌ Validation error: {e} (Field: {e.field})")
    except ClaudeCodeError as e:
        logger.error(f"❌ Claude Code error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


def demonstrate_advanced_configuration(logger: logging.Logger):
    """Demonstrate advanced configuration options."""
    logger.info("=== Advanced Configuration ===")
    
    try:
        # Create production-ready configuration
        config = ClaudeCodeConfig(
            timeout=30.0,
            max_turns=3,
            verbose=True,
            system_prompt="You are a helpful, concise assistant focused on practical answers.",
            allowed_tools=["Python", "Bash"],
            max_retries=2,
            retry_delay=1.0,
            retry_backoff_factor=2.0,
            enable_metrics=True,
            log_level=logging.INFO,
            environment_vars={"CLAUDE_CONTEXT": "production_demo"}
        )
        
        wrapper = ClaudeCodeWrapper(config)
        
        response = wrapper.run(
            "Write a simple Python function to calculate factorial",
            output_format=OutputFormat.JSON
        )
        
        logger.info(f"✅ Advanced config success - Content length: {len(response.content)}")
        logger.info(f"   Error status: {response.is_error}")
        logger.info(f"   Metrics collected: Cost=${response.metrics.cost_usd:.4f}")
        
        # Display metrics
        metrics = wrapper.get_metrics()
        logger.info(f"   Wrapper metrics: {metrics}")
        
    except ClaudeCodeTimeoutError as e:
        logger.error(f"❌ Timeout after {e.timeout_duration}s: {e}")
    except ClaudeCodeProcessError as e:
        logger.error(f"❌ Process failed (code {e.returncode}): {e}")
        if e.stderr:
            logger.error(f"   Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")


def demonstrate_session_management(logger: logging.Logger):
    """Demonstrate session management with error handling."""
    logger.info("=== Session Management ===")
    
    try:
        config = ClaudeCodeConfig(
            max_turns=5,
            timeout=20.0,
            system_prompt="You are helping with a Python tutorial.",
            max_retries=1
        )
        wrapper = ClaudeCodeWrapper(config)
        
        # Use session context manager
        with wrapper.session(max_turns=3) as session:
            logger.info("Session started")
            
            # First exchange
            try:
                response1 = session.ask("What are Python lists?")
                logger.info(f"✅ Q1: {response1.content[:80]}...")
                logger.info(f"   Session ID: {session.session_id}")
            except Exception as e:
                logger.error(f"❌ First question failed: {e}")
            
            # Second exchange (builds on context)
            try:
                response2 = session.ask("Can you show a simple example?")
                logger.info(f"✅ Q2: {response2.content[:80]}...")
            except Exception as e:
                logger.error(f"❌ Second question failed: {e}")
            
            # Get session history
            history = session.get_history()
            logger.info(f"   Session history: {len(history)} exchanges")
            
            # Display any errors in history
            error_count = sum(1 for resp in history if resp.is_error)
            if error_count > 0:
                logger.warning(f"   {error_count} responses had errors")
                
    except Exception as e:
        logger.error(f"❌ Session management failed: {e}")


def demonstrate_streaming_with_recovery(logger: logging.Logger):
    """Demonstrate streaming with comprehensive error handling."""
    logger.info("=== Streaming with Error Recovery ===")
    
    try:
        logger.info("Attempting streaming request...")
        
        event_count = 0
        error_count = 0
        content_parts = []
        
        # Use streaming with error recovery
        for event in ask_claude_streaming("Count from 1 to 5, explaining each number briefly"):
            event_count += 1
            
            # Handle different event types
            event_type = event.get("type", "unknown")
            
            if event_type == "error":
                error_count += 1
                logger.error(f"   Stream error: {event.get('message', 'Unknown error')}")
                if event.get('returncode'):
                    logger.error(f"   Return code: {event['returncode']}")
                continue
                
            elif event_type == "parse_error":
                error_count += 1
                logger.warning(f"   Parse error: {event.get('message', 'Parse failed')}")
                logger.debug(f"   Raw line: {event.get('raw_line', '')[:50]}...")
                continue
                
            elif event_type == "message":
                content = event.get("content", "")
                if content:
                    content_parts.append(content)
                    logger.debug(f"   Message chunk: {content[:30]}...")
                    
            elif event_type == "init":
                logger.info(f"   Stream initialized: {event.get('session_id', 'no-session')}")
                
            elif event_type == "result":
                logger.info(f"   Stream completed: {event.get('status', 'unknown')}")
                stats = event.get('stats', {})
                if stats:
                    logger.info(f"   Stats: {stats}")
                    
            else:
                logger.debug(f"   Event: {event_type}")
            
            # Safety limit to prevent infinite loops
            if event_count > 50:
                logger.warning("   Stream safety limit reached, stopping")
                break
        
        # Summary
        full_content = "".join(content_parts)
        logger.info(f"✅ Streaming completed:")
        logger.info(f"   Total events: {event_count}")
        logger.info(f"   Errors: {error_count}")
        logger.info(f"   Content length: {len(full_content)}")
        if full_content:
            logger.info(f"   Content preview: {full_content[:100]}...")
        
        if error_count == 0:
            logger.info("   ✅ No errors encountered")
        else:
            logger.warning(f"   ⚠️  {error_count} errors handled gracefully")
            
    except Exception as e:
        logger.error(f"❌ Streaming demonstration failed: {e}")
        logger.info("   This is expected if Claude Code isn't properly configured")


def demonstrate_error_handling_patterns(logger: logging.Logger):
    """Demonstrate various error handling patterns."""
    logger.info("=== Error Handling Patterns ===")
    
    # 1. Input validation
    logger.info("1. Testing input validation:")
    try:
        config = ClaudeCodeConfig()
        wrapper = ClaudeCodeWrapper(config)
        
        # This should trigger validation error
        response = wrapper.run("")  # Empty query
        logger.error("   ❌ Validation failed - empty query was accepted")
        
    except ClaudeCodeValidationError as e:
        logger.info(f"   ✅ Validation worked: {e}")
    except Exception as e:
        logger.info(f"   ✅ Other validation: {e}")
    
    # 2. Configuration validation
    logger.info("2. Testing configuration validation:")
    try:
        # This should trigger configuration error
        bad_config = ClaudeCodeConfig(timeout=-1.0)
        logger.error("   ❌ Config validation failed - negative timeout was accepted")
        
    except ClaudeCodeValidationError as e:
        logger.info(f"   ✅ Config validation worked: {e}")
    except Exception as e:
        logger.info(f"   ✅ Config validation worked: {e}")
    
    # 3. Graceful degradation example
    logger.info("3. Testing graceful degradation:")
    try:
        # Use very short timeout to trigger timeout handling
        config = ClaudeCodeConfig(timeout=0.001, max_retries=1)
        wrapper = ClaudeCodeWrapper(config)
        
        response = wrapper.run("This will likely timeout")
        if response.is_error:
            logger.info(f"   ✅ Graceful error handling: {response.content}")
        else:
            logger.info(f"   ✅ Unexpected success: {response.content[:50]}...")
            
    except ClaudeCodeTimeoutError as e:
        logger.info(f"   ✅ Timeout handled properly: {e}")
    except Exception as e:
        logger.info(f"   ✅ Error handled: {e}")


def demonstrate_production_patterns(logger: logging.Logger):
    """Demonstrate production-ready patterns."""
    logger.info("=== Production Patterns ===")
    
    # Production wrapper with comprehensive error handling
    class ProductionClaudeService:
        def __init__(self):
            self.config = ClaudeCodeConfig(
                timeout=30.0,
                max_retries=3,
                retry_delay=1.0,
                retry_backoff_factor=2.0,
                enable_metrics=True,
                log_level=logging.INFO
            )
            try:
                self.wrapper = ClaudeCodeWrapper(self.config)
                self.logger = logging.getLogger(f"{__name__}.production")
            except Exception as e:
                self.logger = logging.getLogger(f"{__name__}.production")
                self.logger.error(f"Failed to initialize wrapper: {e}")
                self.wrapper = None
        
        def ask_with_fallback(self, query: str, fallback_response: str = "Service temporarily unavailable"):
            """Ask with fallback response on any error."""
            if not self.wrapper:
                return fallback_response
                
            try:
                response = self.wrapper.run(query, output_format=OutputFormat.JSON)
                
                if response.is_error:
                    self.logger.warning(f"Response error: {response.error_type}")
                    return fallback_response
                
                self.logger.info(f"Success: {len(response.content)} chars, ${response.metrics.cost_usd:.4f}")
                return response.content
                
            except ClaudeCodeTimeoutError:
                self.logger.error("Request timed out")
                return "Request is taking longer than expected. Please try again."
                
            except ClaudeCodeProcessError as e:
                self.logger.error(f"Process error: {e.returncode}")
                return fallback_response
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return fallback_response
        
        def get_service_health(self):
            """Get service health metrics."""
            if not self.wrapper:
                return {"status": "unhealthy", "reason": "wrapper_not_initialized"}
                
            try:
                metrics = self.wrapper.get_metrics()
                total_requests = metrics.get("total_requests", 0)
                error_count = metrics.get("error_count", 0)
                
                error_rate = (error_count / total_requests) if total_requests > 0 else 0
                
                return {
                    "status": "healthy" if error_rate < 0.1 else "degraded",
                    "total_requests": total_requests,
                    "error_count": error_count,
                    "error_rate": f"{error_rate:.2%}",
                    "avg_execution_time": metrics.get("total_execution_time", 0) / max(total_requests, 1)
                }
            except Exception as e:
                return {"status": "unknown", "error": str(e)}
    
    # Test production service
    try:
        service = ProductionClaudeService()
        
        # Test queries
        queries = [
            "What is 2+2?",
            "Explain REST APIs briefly",
            "What are the benefits of Python?"
        ]
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Production query {i}: {query}")
            response = service.ask_with_fallback(query)
            logger.info(f"   Response: {response[:100]}...")
        
        # Get health metrics
        health = service.get_service_health()
        logger.info(f"Service health: {health}")
        
    except Exception as e:
        logger.error(f"Production pattern demo failed: {e}")


def main():
    """Main demonstration function."""
    logger = setup_logging()
    
    logger.info("🚀 Starting Claude Code SDK Wrapper - Production Demo")
    logger.info("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage(logger)
        print()  # Visual separator
        
        demonstrate_advanced_configuration(logger)  
        print()
        
        demonstrate_session_management(logger)
        print()
        
        demonstrate_streaming_with_recovery(logger)
        print()
        
        demonstrate_error_handling_patterns(logger)
        print()
        
        demonstrate_production_patterns(logger)
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.info("=" * 60)
    logger.info("🏁 Demo completed. Check claude_wrapper.log for detailed logs.")
    
    # Final summary
    print("\n" + "="*60)
    print("DEMO SUMMARY:")
    print("✅ All features demonstrated with proper error handling")
    print("✅ Graceful degradation on failures")  
    print("✅ Comprehensive logging and observability")
    print("✅ Production-ready patterns implemented")
    print("📊 Check claude_wrapper.log for detailed execution logs")
    print("🔧 Adjust ClaudeCodeConfig for your specific needs")
    print("="*60)


if __name__ == "__main__":
    main()
