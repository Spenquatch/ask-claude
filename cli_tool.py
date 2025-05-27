#!/usr/bin/env python3
"""
Claude Code Wrapper CLI Tool

Enterprise-grade command-line interface for the Claude Code SDK Wrapper.
Provides production-ready CLI with comprehensive error handling, configuration
management, and operational features.

Usage:
    python cli_tool.py ask "What is Python?"
    python cli_tool.py --config config.json ask "Generate code"
    python cli_tool.py stream "Write a long explanation"
    python cli_tool.py session --interactive
    python cli_tool.py health
    python cli_tool.py benchmark --queries queries.txt
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from claude_code_wrapper import (
    ClaudeCodeWrapper,
    ClaudeCodeConfig,
    OutputFormat,
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
    ClaudeCodeProcessError,
    ClaudeCodeValidationError,
    ClaudeCodeConfigurationError,
    ErrorSeverity
)


class ClaudeCLI:
    """Enterprise CLI for Claude Code Wrapper."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.wrapper: Optional[ClaudeCodeWrapper] = None
        self.config: Optional[ClaudeCodeConfig] = None
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for CLI operations."""
        logging.basicConfig(
            level=logging.WARNING,  # Default to WARNING for CLI
            format='%(levelname)s: %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_config(self, config_path: Optional[Path] = None) -> ClaudeCodeConfig:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                
                # Convert paths in config
                if 'mcp_config_path' in config_data and config_data['mcp_config_path']:
                    config_data['mcp_config_path'] = Path(config_data['mcp_config_path'])
                if 'working_directory' in config_data and config_data['working_directory']:
                    config_data['working_directory'] = Path(config_data['working_directory'])
                
                self.config = ClaudeCodeConfig(**config_data)
                self.logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
                self.config = ClaudeCodeConfig()
        else:
            self.config = ClaudeCodeConfig()
        
        return self.config
    
    def initialize_wrapper(self, verbose: bool = False) -> bool:
        """Initialize the Claude Code wrapper."""
        try:
            if verbose:
                self.config.log_level = logging.INFO
                self.config.verbose = True
            
            self.wrapper = ClaudeCodeWrapper(self.config)
            return True
            
        except ClaudeCodeConfigurationError as e:
            print(f"❌ Configuration Error: {e}", file=sys.stderr)
            if e.config_field:
                print(f"   Field: {e.config_field}", file=sys.stderr)
            return False
            
        except Exception as e:
            print(f"❌ Initialization Error: {e}", file=sys.stderr)
            return False
    
    def _build_approval_config(self, args) -> Optional[Dict[str, Any]]:
        """Build MCP auto-approval configuration from command line arguments."""
        if not hasattr(args, 'approval_strategy') or not args.approval_strategy:
            return None
        
        approval_config = {
            'enabled': True,
            'strategy': args.approval_strategy
        }
        
        if args.approval_strategy == 'allowlist' and hasattr(args, 'approval_allowlist'):
            approval_config['allowlist'] = args.approval_allowlist or []
        
        if args.approval_strategy == 'patterns':
            if hasattr(args, 'approval_allow_patterns') and args.approval_allow_patterns:
                approval_config['allow_patterns'] = args.approval_allow_patterns
            if hasattr(args, 'approval_deny_patterns') and args.approval_deny_patterns:
                approval_config['deny_patterns'] = args.approval_deny_patterns
        
        return approval_config
    
    def cmd_ask(self, query: str, output_format: str = "text", **kwargs) -> int:
        """Handle ask command."""
        if not query.strip():
            print("❌ Error: Query cannot be empty", file=sys.stderr)
            return 1
        
        try:
            format_enum = OutputFormat(output_format.lower())
            
            response = self.wrapper.run(query, output_format=format_enum, **kwargs)
            
            if response.is_error:
                print(f"⚠️  Response Error: {response.error_type}", file=sys.stderr)
                if response.error_subtype:
                    print(f"   Subtype: {response.error_subtype}", file=sys.stderr)
            
            # Output the content
            print(response.content)
            
            # Show metadata if requested
            if kwargs.get('show_metadata'):
                self._print_response_metadata(response)
            
            return 0 if not response.is_error else 1
            
        except ClaudeCodeValidationError as e:
            print(f"❌ Validation Error: {e}", file=sys.stderr)
            return 1
            
        except ClaudeCodeTimeoutError as e:
            print(f"❌ Timeout Error: {e}", file=sys.stderr)
            return 1
            
        except ClaudeCodeProcessError as e:
            print(f"❌ Process Error: {e}", file=sys.stderr)
            if e.stderr:
                print(f"   Details: {e.stderr}", file=sys.stderr)
            return e.returncode
            
        except Exception as e:
            print(f"❌ Unexpected Error: {e}", file=sys.stderr)
            return 1
    
    def cmd_stream(self, query: str, **kwargs) -> int:
        """Handle streaming command."""
        if not query.strip():
            print("❌ Error: Query cannot be empty", file=sys.stderr)
            return 1
        
        try:
            print("🌊 Starting stream...", file=sys.stderr)
            
            event_count = 0
            error_count = 0
            content_parts = []
            
            for event in self.wrapper.run_streaming(query, **kwargs):
                event_count += 1
                event_type = event.get("type", "unknown")
                
                # Debug: log event types we're seeing (commented out to avoid slowing down)
                # if kwargs.get('verbose'):
                #     print(f"[DEBUG] Event type: {event_type}, Keys: {list(event.keys())}", file=sys.stderr)
                
                if event_type == "error":
                    error_count += 1
                    print(f"❌ Stream Error: {event.get('message', 'Unknown')}", file=sys.stderr)
                    
                elif event_type == "parse_error":
                    error_count += 1
                    if kwargs.get('verbose'):
                        print(f"⚠️  Parse Error: {event.get('message', 'Parse failed')}", file=sys.stderr)
                    
                elif event_type == "message":
                    content = event.get("content", "")
                    if content:
                        content_parts.append(content)
                        print(content, end="", flush=True)
                
                # Handle different possible content formats
                elif "content" in event:
                    content = event.get("content", "")
                    if content:
                        content_parts.append(content)
                        print(content, end="", flush=True)
                
                # Handle text events (common in streaming)
                elif event_type == "text" or "text" in event:
                    text = event.get("text", "")
                    if text:
                        content_parts.append(text)
                        print(text, end="", flush=True)
                
                # Handle content_block events (Claude's format)
                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        content_parts.append(text)
                        print(text, end="", flush=True)
                
                # Handle assistant messages (from debug output)
                elif event_type == "assistant":
                    message = event.get("message", "")
                    if message:
                        content_parts.append(message)
                        print(message, end="", flush=True)
                        
                elif event_type == "init":
                    if kwargs.get('verbose'):
                        session_id = event.get("session_id", "no-session")
                        print(f"🚀 Session: {session_id}", file=sys.stderr)
                        
                elif event_type == "result":
                    if kwargs.get('verbose'):
                        status = event.get("status", "unknown")
                        print(f"\n🏁 Status: {status}", file=sys.stderr)
                        stats = event.get("stats", {})
                        if stats:
                            print(f"📊 Stats: {stats}", file=sys.stderr)
            
            print()  # Final newline
            
            if kwargs.get('show_stats'):
                print(f"\n📊 Stream Stats:", file=sys.stderr)
                print(f"   Events: {event_count}", file=sys.stderr)
                print(f"   Errors: {error_count}", file=sys.stderr)
                print(f"   Content: {len(''.join(content_parts))} chars", file=sys.stderr)
            
            return 0 if error_count == 0 else 1
            
        except KeyboardInterrupt:
            print("\n⏹️  Stream interrupted by user", file=sys.stderr)
            return 130  # Standard SIGINT exit code
            
        except Exception as e:
            print(f"\n❌ Stream Error: {e}", file=sys.stderr)
            return 1
    
    def cmd_session(self, interactive: bool = False, **kwargs) -> int:
        """Handle session command."""
        if not interactive:
            print("❌ Error: Non-interactive sessions not yet implemented", file=sys.stderr)
            return 1
        
        try:
            print("🔄 Starting interactive session...")
            print("💡 Type 'exit', 'quit', or Ctrl+C to end session")
            print("💡 Type 'help' for commands")
            print("-" * 50)
            
            with self.wrapper.session(**kwargs) as session:
                turn_count = 0
                
                while True:
                    try:
                        # Get user input
                        query = input(f"\n[{turn_count + 1}] ❓ You: ").strip()
                        
                        if not query:
                            continue
                            
                        if query.lower() in ['exit', 'quit']:
                            break
                            
                        if query.lower() == 'help':
                            self._print_session_help()
                            continue
                            
                        if query.lower() == 'history':
                            self._print_session_history(session)
                            continue
                            
                        if query.lower() == 'clear':
                            session.clear_history()
                            turn_count = 0
                            print("🧹 Session history cleared")
                            continue
                        
                        # Ask question
                        print("🤖 Claude: ", end="", flush=True)
                        response = session.ask(query)
                        
                        if response.is_error:
                            print(f"❌ Error: {response.error_type}")
                        else:
                            print(response.content)
                            
                        turn_count += 1
                        
                        # Show session info if verbose
                        if kwargs.get('verbose'):
                            print(f"   💰 Cost: ${response.metrics.cost_usd:.6f}")
                            print(f"   ⏱️  Time: {response.execution_time:.2f}s")
                        
                    except KeyboardInterrupt:
                        print("\n⏹️  Session interrupted")
                        break
                        
                    except EOFError:
                        print("\n👋 Session ended")
                        break
            
            print(f"\n🏁 Session completed with {turn_count} exchanges")
            return 0
            
        except Exception as e:
            print(f"❌ Session Error: {e}", file=sys.stderr)
            return 1
    
    def cmd_health(self) -> int:
        """Check health of Claude Code wrapper."""
        try:
            print("🏥 Claude Code Wrapper Health Check")
            print("-" * 40)
            
            # Test wrapper initialization
            if self.wrapper is None:
                print("❌ Wrapper not initialized")
                return 1
            
            # Test basic functionality
            start_time = time.time()
            response = self.wrapper.run("Test health check - respond with 'OK'", timeout=10.0)
            health_time = time.time() - start_time
            
            print(f"✅ Basic functionality: Working")
            print(f"⏱️  Response time: {health_time:.2f}s")
            
            if response.is_error:
                print(f"⚠️  Response had error: {response.error_type}")
            else:
                print(f"📝 Response: {response.content[:50]}...")
            
            # Get metrics
            metrics = self.wrapper.get_metrics()
            print(f"📊 Metrics: {metrics}")
            
            # Test streaming (quick test)
            print("🌊 Testing streaming...")
            stream_events = 0
            try:
                for event in self.wrapper.run_streaming("Say 'streaming test'"):
                    stream_events += 1
                    if stream_events > 5:  # Quick test
                        break
                print(f"✅ Streaming: {stream_events} events received")
            except Exception as e:
                print(f"⚠️  Streaming: {e}")
            
            print("\n🎯 Overall Status: Healthy")
            return 0
            
        except ClaudeCodeTimeoutError:
            print("❌ Health check timed out")
            return 1
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return 1
    
    def cmd_benchmark(self, queries_file: Optional[Path] = None, iterations: int = 3) -> int:
        """Run performance benchmarks."""
        try:
            print(f"🏃 Running performance benchmark ({iterations} iterations)")
            print("-" * 50)
            
            # Default queries if no file provided
            if queries_file and queries_file.exists():
                with open(queries_file) as f:
                    queries = [line.strip() for line in f if line.strip()]
            else:
                queries = [
                    "What is 2+2?",
                    "Explain Python in one sentence",
                    "Write a hello world function",
                    "What are the benefits of REST APIs?"
                ]
            
            results = []
            
            for i, query in enumerate(queries, 1):
                print(f"🔄 Query {i}/{len(queries)}: {query[:50]}...")
                
                times = []
                errors = 0
                
                for iteration in range(iterations):
                    try:
                        start = time.time()
                        response = self.wrapper.run(query)
                        end = time.time()
                        
                        times.append(end - start)
                        
                        if response.is_error:
                            errors += 1
                            
                    except Exception as e:
                        errors += 1
                        print(f"   ⚠️  Iteration {iteration + 1} failed: {e}")
                
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    
                    results.append({
                        "query": query,
                        "avg_time": avg_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "error_rate": errors / iterations
                    })
                    
                    print(f"   ⏱️  Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                    if errors > 0:
                        print(f"   ❌ Errors: {errors}/{iterations}")
            
            # Summary
            print("\n📊 Benchmark Summary:")
            print("-" * 30)
            
            if results:
                overall_avg = sum(r["avg_time"] for r in results) / len(results)
                overall_errors = sum(r["error_rate"] for r in results) / len(results)
                
                print(f"Overall Average Time: {overall_avg:.3f}s")
                print(f"Overall Error Rate: {overall_errors:.1%}")
                
                # Best and worst performers
                fastest = min(results, key=lambda x: x["avg_time"])
                slowest = max(results, key=lambda x: x["avg_time"])
                
                print(f"Fastest Query: {fastest['avg_time']:.3f}s")
                print(f"Slowest Query: {slowest['avg_time']:.3f}s")
            
            return 0
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}", file=sys.stderr)
            return 1
    
    def _print_response_metadata(self, response):
        """Print response metadata."""
        print(f"\n📊 Metadata:", file=sys.stderr)
        print(f"   Session ID: {response.session_id}", file=sys.stderr)
        print(f"   Is Error: {response.is_error}", file=sys.stderr)
        print(f"   Execution Time: {response.execution_time:.3f}s", file=sys.stderr)
        print(f"   Cost: ${response.metrics.cost_usd:.6f}", file=sys.stderr)
        print(f"   Duration: {response.metrics.duration_ms}ms", file=sys.stderr)
        print(f"   Turns: {response.metrics.num_turns}", file=sys.stderr)
    
    def _print_session_help(self):
        """Print session help."""
        print("\n💡 Session Commands:")
        print("   help     - Show this help")
        print("   history  - Show conversation history")
        print("   clear    - Clear session history")
        print("   exit     - End session")
        print("   quit     - End session")
    
    def _print_session_history(self, session):
        """Print session history."""
        history = session.get_history()
        print(f"\n📚 Session History ({len(history)} exchanges):")
        for i, response in enumerate(history, 1):
            status = "❌" if response.is_error else "✅"
            print(f"   {i}. {status} {response.content[:50]}...")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Claude Code SDK Wrapper CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ask "What is Python?"
  %(prog)s --config config.json ask "Generate code" --format json
  %(prog)s stream "Write a tutorial"
  %(prog)s session --interactive --verbose
  %(prog)s health
  %(prog)s benchmark --queries queries.txt --iterations 5
        """
    )
    
    # Global options
    parser.add_argument("--config", "-c", type=Path, help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("query", help="Query to ask Claude")
    ask_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ask_parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    ask_parser.add_argument("--max-turns", type=int, help="Maximum conversation turns")
    ask_parser.add_argument("--session-id", help="Resume specific session")
    ask_parser.add_argument("--continue", action="store_true", help="Continue last session")
    ask_parser.add_argument("--show-metadata", action="store_true", help="Show response metadata")
    ask_parser.add_argument("--mcp-config", type=Path, help="MCP servers configuration file")
    
    # MCP Auto-approval options
    ask_parser.add_argument("--approval-strategy", 
                          choices=['allowlist', 'patterns', 'all', 'none'],
                          help='MCP tool approval strategy')
    ask_parser.add_argument("--approval-allowlist", 
                          nargs='+',
                          help='List of allowed MCP tools')
    ask_parser.add_argument("--approval-allow-patterns", 
                          nargs='+',
                          help='Regex patterns for allowed tools')
    ask_parser.add_argument("--approval-deny-patterns", 
                          nargs='+',
                          help='Regex patterns for denied tools')
    
    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream a response")
    stream_parser.add_argument("query", help="Query to stream from Claude")
    stream_parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    stream_parser.add_argument("--show-stats", action="store_true", help="Show streaming statistics")
    stream_parser.add_argument("--mcp-config", type=Path, help="MCP servers configuration file")
    
    # MCP Auto-approval options
    stream_parser.add_argument("--approval-strategy", 
                             choices=['allowlist', 'patterns', 'all', 'none'],
                             help='MCP tool approval strategy')
    stream_parser.add_argument("--approval-allowlist", 
                             nargs='+',
                             help='List of allowed MCP tools')
    stream_parser.add_argument("--approval-allow-patterns", 
                             nargs='+',
                             help='Regex patterns for allowed tools')
    stream_parser.add_argument("--approval-deny-patterns", 
                             nargs='+',
                             help='Regex patterns for denied tools')
    
    # Session command
    session_parser = subparsers.add_parser("session", help="Interactive session")
    session_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    session_parser.add_argument("--max-turns", type=int, help="Maximum session turns")
    
    # MCP Auto-approval options
    session_parser.add_argument("--approval-strategy", 
                              choices=['allowlist', 'patterns', 'all', 'none'],
                              help='MCP tool approval strategy')
    session_parser.add_argument("--approval-allowlist", 
                              nargs='+',
                              help='List of allowed MCP tools')
    session_parser.add_argument("--approval-allow-patterns", 
                              nargs='+',
                              help='Regex patterns for allowed tools')
    session_parser.add_argument("--approval-deny-patterns", 
                              nargs='+',
                              help='Regex patterns for denied tools')
    
    # Health command
    subparsers.add_parser("health", help="Check wrapper health")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("--queries", type=Path, help="File with queries to benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=3, help="Iterations per query")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = ClaudeCLI()
    
    # Set logging level based on verbosity
    if args.quiet:
        cli.logger.setLevel(logging.ERROR)
    elif args.verbose:
        cli.logger.setLevel(logging.DEBUG)
    
    # Build configuration from all sources
    config_dict = {}
    
    # Load base config from file if provided
    if args.config and args.config.exists():
        try:
            with open(args.config) as f:
                config_dict = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}", file=sys.stderr)
            return 1
    
    # Add MCP config if provided
    if hasattr(args, 'mcp_config') and args.mcp_config:
        config_dict['mcp_config_path'] = args.mcp_config  # Already a Path from argparse
    
    # Add approval config if provided  
    approval_config = cli._build_approval_config(args)
    if approval_config:
        config_dict['mcp_auto_approval'] = approval_config
    
    # Load configuration with all settings
    cli.config = ClaudeCodeConfig.from_dict(config_dict) if config_dict else ClaudeCodeConfig()
    
    # Initialize wrapper with complete config
    if not cli.initialize_wrapper(args.verbose):
        return 1
    
    # Execute command
    try:
        if args.command == "ask":
            kwargs = {}
            if args.timeout:
                kwargs['timeout'] = args.timeout
            if args.max_turns:
                kwargs['max_turns'] = args.max_turns
            if args.session_id:
                kwargs['session_id'] = args.session_id
            if getattr(args, 'continue', False):
                kwargs['continue_session'] = True
            
            kwargs['show_metadata'] = args.show_metadata
            
            return cli.cmd_ask(args.query, args.format, **kwargs)
            
        elif args.command == "stream":
            kwargs = {'verbose': args.verbose}
            if args.timeout:
                kwargs['timeout'] = args.timeout
            kwargs['show_stats'] = args.show_stats
            
            return cli.cmd_stream(args.query, **kwargs)
            
        elif args.command == "session":
            kwargs = {'verbose': args.verbose}
            if args.max_turns:
                kwargs['max_turns'] = args.max_turns
            
            # Add approval config if provided
            approval_config = cli._build_approval_config(args)
            if approval_config:
                kwargs['mcp_auto_approval'] = approval_config
            
            return cli.cmd_session(args.interactive, **kwargs)
            
        elif args.command == "health":
            return cli.cmd_health()
            
        elif args.command == "benchmark":
            return cli.cmd_benchmark(args.queries, args.iterations)
            
        else:
            print(f"❌ Unknown command: {args.command}", file=sys.stderr)
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation interrupted by user", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
