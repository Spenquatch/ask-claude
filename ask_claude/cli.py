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
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .wrapper import (
    ClaudeCodeConfig,
    ClaudeCodeConfigurationError,
    ClaudeCodeProcessError,
    ClaudeCodeTimeoutError,
    ClaudeCodeValidationError,
    ClaudeCodeWrapper,
    OutputFormat,
)


class ClaudeCLI:
    """Enterprise CLI for Claude Code Wrapper."""

    def __init__(self) -> None:
        self.logger = self._setup_logging()
        self.wrapper: ClaudeCodeWrapper | None = None
        self.config: ClaudeCodeConfig | None = None

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for CLI operations."""
        # Suppress all loggers by default for clean CLI output
        logging.getLogger().setLevel(logging.CRITICAL)

        # Create our own logger for CLI-specific messages
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

        # Also suppress the wrapper logger by default
        logging.getLogger("claude_code_wrapper").setLevel(logging.CRITICAL)

        return logger

    def load_config(self, config_path: Path | None = None) -> ClaudeCodeConfig:
        """Load configuration from file or use defaults."""
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)

                # Convert paths in config
                if "mcp_config_path" in config_data and config_data["mcp_config_path"]:
                    config_data["mcp_config_path"] = Path(
                        config_data["mcp_config_path"]
                    )
                if (
                    "working_directory" in config_data
                    and config_data["working_directory"]
                ):
                    config_data["working_directory"] = Path(
                        config_data["working_directory"]
                    )

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
            if verbose and self.config is not None:
                self.config.log_level = logging.INFO
                # Note: Don't set config.verbose = True here as it adds --verbose
                # to Claude commands, which can cause unwanted meta-commentary

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

    def _build_approval_config(self, args: Any) -> dict[str, Any] | None:
        """Build MCP auto-approval configuration from command line arguments."""
        if not hasattr(args, "approval_strategy") or not args.approval_strategy:
            return None

        approval_config = {"enabled": True, "strategy": args.approval_strategy}

        if args.approval_strategy == "allowlist" and hasattr(
            args, "approval_allowlist"
        ):
            approval_config["allowlist"] = args.approval_allowlist or []

        if args.approval_strategy == "patterns":
            if (
                hasattr(args, "approval_allow_patterns")
                and args.approval_allow_patterns
            ):
                approval_config["allow_patterns"] = args.approval_allow_patterns
            if hasattr(args, "approval_deny_patterns") and args.approval_deny_patterns:
                approval_config["deny_patterns"] = args.approval_deny_patterns

        return approval_config

    def cmd_ask(self, query: str, output_format: str = "text", **kwargs: Any) -> int:
        """Handle ask command."""
        if not query.strip():
            print("❌ Error: Query cannot be empty", file=sys.stderr)
            return 1

        try:
            format_enum = OutputFormat(output_format.lower())

            if self.wrapper is None:
                print("❌ Error: Wrapper not initialized", file=sys.stderr)
                return 1
            response = self.wrapper.run(query, output_format=format_enum, **kwargs)

            if response.is_error:
                print(f"⚠️  Response Error: {response.error_type}", file=sys.stderr)
                if response.error_subtype:
                    print(f"   Subtype: {response.error_subtype}", file=sys.stderr)

            # Output the content
            print(response.content)

            # Show metadata if requested
            if kwargs.get("show_metadata"):
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

    def _get_tool_display_info(
        self, tool_name: str, tool_input: dict
    ) -> tuple[str, str, dict]:
        """Get display information for a tool based on its type.

        Returns: (emoji, action_description, display_fields)
        """
        # Tool display registry - easily extensible
        tool_patterns = {
            # Core tools
            "Bash": (
                "🖥️",
                "run Bash command",
                {"command": "Command", "description": "Purpose"},
            ),
            "Read": ("📄", "read file", {"file_path": "File"}),
            "Write": ("📝", "write file", {"file_path": "File"}),
            "Edit": ("✏️", "edit file", {"file_path": "File"}),
            "MultiEdit": ("✏️", "edit multiple files", {"file_path": "File"}),
            "Grep": ("🔍", "search with grep", {"pattern": "Pattern", "path": "Path"}),
            "Glob": ("🔍", "search with glob", {"pattern": "Pattern", "path": "Path"}),
            "LS": ("📁", "list directory", {"path": "Path"}),
            "Task": (
                "🤖",
                "create agent task",
                {"description": "Task", "prompt": "Details"},
            ),
            # Web tools
            "WebSearch": ("🌐", "search the web", {"query": "Query"}),
            "WebFetch": (
                "🌐",
                "fetch web content",
                {"url": "URL", "prompt": "Purpose"},
            ),
            # Todo tools
            "TodoRead": ("📋", "read todos", {}),
            "TodoWrite": ("📋", "update todos", {"todos": "Updates"}),
            # Notebook tools
            "NotebookRead": ("📓", "read notebook", {"notebook_path": "File"}),
            "NotebookEdit": (
                "📓",
                "edit notebook",
                {"notebook_path": "File", "cell_number": "Cell"},
            ),
        }

        # Check for exact match first
        if tool_name in tool_patterns:
            return tool_patterns[tool_name]

        # Special handling for MCP tools
        if "sequential-thinking" in tool_name:
            return (
                "🤔",
                "think",
                {
                    "thought": "Thought",
                    "thoughtNumber": "Step",
                    "totalThoughts": "Total",
                },
            )
        elif "deepwiki" in tool_name:
            return ("📚", "fetch documentation", {"url": "URL", "maxDepth": "Depth"})
        elif "mcp__" in tool_name:
            # Generic MCP tool
            return ("🔧", "use MCP tool", {})

        # Default
        return (
            "🔧",
            "use tool",
            {"description": "Purpose", "query": "Query", "command": "Command"},
        )

    def cmd_stream(self, query: str, **kwargs: Any) -> int:
        """Handle streaming command."""
        if not query.strip():
            print("❌ Error: Query cannot be empty", file=sys.stderr)
            return 1

        try:
            if kwargs.get("verbose"):
                print("🌊 Starting stream...", file=sys.stderr)

            event_count = 0
            error_count = 0
            content_parts = []
            pending_tool_uses: dict[
                str, Any
            ] = {}  # Track tool uses by ID to match with results

            if self.wrapper is None:
                print("❌ Error: Wrapper not initialized", file=sys.stderr)
                return 1
            for event in self.wrapper.run_streaming(query, **kwargs):
                event_count += 1
                event_type = event.get("type", "unknown")

                # DEBUG: Show all events if verbose
                if kwargs.get("verbose"):
                    print(
                        f"\n[DEBUG] Event #{event_count}: {event_type} - {event}\n",
                        file=sys.stderr,
                    )

                if event_type == "error":
                    error_count += 1
                    print(
                        f"❌ Stream Error: {event.get('message', 'Unknown')}",
                        file=sys.stderr,
                    )

                elif event_type == "parse_error":
                    error_count += 1
                    if kwargs.get("verbose"):
                        print(
                            f"⚠️  Parse Error: {event.get('message', 'Parse failed')}",
                            file=sys.stderr,
                        )

                # Handle user messages (tool results/errors)
                elif event_type == "user":
                    message = event.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "tool_result"
                                ):
                                    tool_use_id = item.get("tool_use_id")
                                    is_error = item.get("is_error", False)
                                    result_content = item.get("content", "")

                                    # Get the original tool info
                                    tool_info = (
                                        pending_tool_uses.get(tool_use_id, {})
                                        if tool_use_id
                                        else {}
                                    )
                                    tool_name = tool_info.get("name", "unknown")

                                    if is_error:
                                        # Handle permission errors specially
                                        if (
                                            "permissions" in result_content
                                            and "hasn't granted" in result_content
                                        ):
                                            print(
                                                f"❌ Tool '{tool_name}' not approved",
                                                file=sys.stderr,
                                            )
                                            print(
                                                "   To enable, add: --approval-strategy all",
                                                file=sys.stderr,
                                            )
                                            print(
                                                f"   Or: --approval-allowlist '{tool_name}'",
                                                file=sys.stderr,
                                            )
                                            print(
                                                "   See docs/mcp-integration.md for details\n",
                                                file=sys.stderr,
                                            )
                                        else:
                                            # Other errors
                                            print(
                                                f"❌ Tool error: {result_content}",
                                                file=sys.stderr,
                                            )
                                    else:
                                        # Successful tool result
                                        if "sequential-thinking" in tool_name:
                                            # For sequential thinking, just show completion checkmark
                                            print("✓", file=sys.stderr)
                                        elif kwargs.get("verbose"):
                                            # Show full results in verbose mode
                                            if result_content:
                                                print(
                                                    "✓ Tool completed successfully",
                                                    file=sys.stderr,
                                                )
                                                if len(result_content) > 200:
                                                    print(
                                                        f"   Result: {result_content[:200]}...",
                                                        file=sys.stderr,
                                                    )
                                                else:
                                                    print(
                                                        f"   Result: {result_content}",
                                                        file=sys.stderr,
                                                    )
                                        else:
                                            # For other tools in non-verbose mode, just acknowledge
                                            if tool_name in [
                                                "Bash",
                                                "Read",
                                                "Write",
                                                "Edit",
                                            ]:
                                                print(
                                                    f"✓ {tool_name} completed",
                                                    file=sys.stderr,
                                                )

                # Handle assistant messages according to Claude Code docs
                elif event_type == "assistant":
                    # DEBUG: Show raw event structure
                    if kwargs.get("verbose"):
                        print(f"\n[DEBUG] Assistant event: {event}\n", file=sys.stderr)

                    message = event.get("message", {})
                    if isinstance(message, dict):
                        stop_reason = message.get("stop_reason")
                        content = message.get("content", [])

                        # Process all content items
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    item_type = item.get("type")

                                    if item_type == "text":
                                        text = item.get("text", "")
                                        if text:
                                            # For tool_use stop_reason, show as thinking
                                            if stop_reason == "tool_use":
                                                print(f"💭 {text}", file=sys.stderr)
                                            else:
                                                # Regular content - add to output
                                                content_parts.append(text)
                                                print(text, end="", flush=True)

                                    elif item_type == "tool_use":
                                        # Track tool use for matching with results
                                        tool_id = item.get("id")
                                        tool_name = item.get("name", "unknown")
                                        tool_input = item.get("input", {})

                                        if tool_id is not None:
                                            pending_tool_uses[tool_id] = {
                                                "name": tool_name,
                                                "input": tool_input,
                                            }

                                        # Get display info for this tool
                                        emoji, action, display_fields = (
                                            self._get_tool_display_info(
                                                tool_name, tool_input
                                            )
                                        )

                                        # Special handling for sequential thinking
                                        if (
                                            "sequential-thinking" in tool_name
                                            and "thoughtNumber" in tool_input
                                        ):
                                            thought_num = tool_input.get(
                                                "thoughtNumber", "?"
                                            )
                                            total_thoughts = tool_input.get(
                                                "totalThoughts", "?"
                                            )
                                            print(
                                                f"\n{emoji} Thinking Step {thought_num}/{total_thoughts}:",
                                                file=sys.stderr,
                                            )
                                            if (
                                                "thought" in tool_input
                                                and tool_input["thought"]
                                            ):
                                                print(
                                                    f"   {tool_input['thought']}",
                                                    file=sys.stderr,
                                                )
                                        else:
                                            # Standard tool display
                                            print(
                                                f"\n{emoji} Claude wants to {action}: {tool_name}",
                                                file=sys.stderr,
                                            )

                                            # Display relevant fields
                                            if display_fields:
                                                for (
                                                    field_key,
                                                    field_label,
                                                ) in display_fields.items():
                                                    if (
                                                        field_key in tool_input
                                                        and tool_input[field_key]
                                                    ):
                                                        value = tool_input[field_key]
                                                        # Truncate long values
                                                        if (
                                                            isinstance(value, str)
                                                            and len(value) > 100
                                                        ):
                                                            value = value[:100] + "..."
                                                        elif isinstance(
                                                            value, list | dict
                                                        ):
                                                            value = f"[{type(value).__name__} with {len(value)} items]"
                                                        print(
                                                            f"   {field_label}: {value}",
                                                            file=sys.stderr,
                                                        )
                                            else:
                                                # If no specific fields defined, show all non-empty fields
                                                for key, value in tool_input.items():
                                                    if value and key not in [
                                                        "tool_name"
                                                    ]:
                                                        if (
                                                            isinstance(value, str)
                                                            and len(value) > 100
                                                        ):
                                                            value = value[:100] + "..."
                                                        elif isinstance(
                                                            value, list | dict
                                                        ):
                                                            value = f"[{type(value).__name__} with {len(value)} items]"
                                                        print(
                                                            f"   {key}: {value}",
                                                            file=sys.stderr,
                                                        )

                        elif isinstance(content, str):
                            # String content - display normally
                            if kwargs.get("verbose"):
                                print(
                                    f"[DEBUG] String content: {repr(content)}\n",
                                    file=sys.stderr,
                                )
                            content_parts.append(content)
                            print(content, end="", flush=True)

                elif event_type == "system":
                    subtype = event.get("subtype", "")
                    if subtype == "init":
                        if kwargs.get("verbose"):
                            session_id = event.get("session_id", "no-session")
                            tools = event.get("tools", [])
                            mcp_servers = event.get("mcp_servers", [])
                            print(f"🚀 Session: {session_id}", file=sys.stderr)
                            if tools:
                                print(
                                    f"   Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}",
                                    file=sys.stderr,
                                )
                            if mcp_servers:
                                print(
                                    f"   MCP Servers: {', '.join([s['name'] for s in mcp_servers])}",
                                    file=sys.stderr,
                                )

                elif event_type == "result":
                    subtype = event.get("subtype", "")
                    if subtype == "error_max_turns":
                        print("\n⚠️  Maximum turns reached", file=sys.stderr)
                    elif kwargs.get("verbose"):
                        status = event.get("subtype", "unknown")
                        print(f"\n🏁 Status: {status}", file=sys.stderr)
                        if "cost_usd" in event:
                            print(f"   Cost: ${event['cost_usd']:.4f}", file=sys.stderr)
                        if "duration_ms" in event:
                            print(
                                f"   Duration: {event['duration_ms']/1000:.2f}s",
                                file=sys.stderr,
                            )
                        if "num_turns" in event:
                            print(f"   Turns: {event['num_turns']}", file=sys.stderr)

                else:
                    # Catch unhandled event types to prevent raw JSON output
                    # Claude CLI sometimes outputs raw events that we need to suppress
                    # Suppress the raw event by not printing anything
                    pass

            print()  # Final newline

            if kwargs.get("show_stats"):
                print("\n📊 Stream Stats:", file=sys.stderr)
                print(f"   Events: {event_count}", file=sys.stderr)
                print(f"   Errors: {error_count}", file=sys.stderr)
                # Safely calculate content length by filtering only strings
                string_parts = [part for part in content_parts if isinstance(part, str)]
                print(
                    f"   Content: {len(''.join(string_parts))} chars", file=sys.stderr
                )

            return 0 if error_count == 0 else 1

        except KeyboardInterrupt:
            print("\n⏹️  Stream interrupted by user", file=sys.stderr)
            return 130  # Standard SIGINT exit code

        except Exception as e:
            print(f"\n❌ Stream Error: {e}", file=sys.stderr)
            return 1

    def cmd_session(self, interactive: bool = False, **kwargs: Any) -> int:
        """Handle session command."""
        if not interactive:
            print(
                "❌ Error: Non-interactive sessions not yet implemented",
                file=sys.stderr,
            )
            return 1

        if not self.initialize_wrapper():
            return 1
        assert self.wrapper is not None

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

                        if query.lower() in ["exit", "quit"]:
                            break

                        if query.lower() == "help":
                            self._print_session_help()
                            continue

                        if query.lower() == "history":
                            self._print_session_history(session)
                            continue

                        if query.lower() == "clear":
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
                        if kwargs.get("verbose"):
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
            response = self.wrapper.run(
                "Test health check - respond with 'OK'", timeout=10.0
            )
            health_time = time.time() - start_time

            print("✅ Basic functionality: Working")
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
                for _event in self.wrapper.run_streaming("Say 'streaming test'"):
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

    def cmd_benchmark(
        self, queries_file: Path | None = None, iterations: int = 3
    ) -> int:
        """Run performance benchmarks."""
        if not self.initialize_wrapper():
            return 1
        assert self.wrapper is not None

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
                    "What are the benefits of REST APIs?",
                ]

            results: list[dict[str, Any]] = []

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

                    results.append(
                        {
                            "query": query,
                            "avg_time": avg_time,
                            "min_time": min_time,
                            "max_time": max_time,
                            "error_rate": errors / iterations,
                        }
                    )

                    print(
                        f"   ⏱️  Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s"
                    )
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

    def _print_response_metadata(self, response: Any) -> None:
        """Print response metadata."""
        print("\n📊 Metadata:", file=sys.stderr)
        print(f"   Session ID: {response.session_id}", file=sys.stderr)
        print(f"   Is Error: {response.is_error}", file=sys.stderr)
        print(f"   Execution Time: {response.execution_time:.3f}s", file=sys.stderr)
        print(f"   Cost: ${response.metrics.cost_usd:.6f}", file=sys.stderr)
        print(f"   Duration: {response.metrics.duration_ms}ms", file=sys.stderr)
        print(f"   Turns: {response.metrics.num_turns}", file=sys.stderr)

    def _print_session_help(self) -> None:
        """Print session help."""
        print("\n💡 Session Commands:")
        print("   help     - Show this help")
        print("   history  - Show conversation history")
        print("   clear    - Clear session history")
        print("   exit     - End session")
        print("   quit     - End session")

    def _print_session_history(self, session: Any) -> None:
        """Print session history."""
        history = session.get_history()
        print(f"\n📚 Session History ({len(history)} exchanges):")
        for i, response in enumerate(history, 1):
            status = "❌" if response.is_error else "✅"
            print(f"   {i}. {status} {response.content[:50]}...")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Ask Claude - Claude Code CLI Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ask "What is Python?"
  %(prog)s --config config.json ask "Generate code" --format json
  %(prog)s stream "Write a tutorial"
  %(prog)s session --interactive --verbose
  %(prog)s health
  %(prog)s benchmark --queries queries.txt --iterations 5
        """,
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
    ask_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    ask_parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    ask_parser.add_argument("--max-turns", type=int, help="Maximum conversation turns")
    ask_parser.add_argument("--session-id", help="Resume specific session")
    ask_parser.add_argument(
        "--continue", action="store_true", help="Continue last session"
    )
    ask_parser.add_argument(
        "--show-metadata", action="store_true", help="Show response metadata"
    )
    ask_parser.add_argument(
        "--mcp-config", type=Path, help="MCP servers configuration file"
    )

    # MCP Auto-approval options
    ask_parser.add_argument(
        "--approval-strategy",
        choices=["allowlist", "patterns", "all", "none"],
        help="MCP tool approval strategy",
    )
    ask_parser.add_argument(
        "--approval-allowlist", nargs="+", help="List of allowed MCP tools"
    )
    ask_parser.add_argument(
        "--approval-allow-patterns", nargs="+", help="Regex patterns for allowed tools"
    )
    ask_parser.add_argument(
        "--approval-deny-patterns", nargs="+", help="Regex patterns for denied tools"
    )

    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream a response")
    stream_parser.add_argument("query", help="Query to stream from Claude")
    stream_parser.add_argument("--timeout", type=float, help="Timeout in seconds")
    stream_parser.add_argument(
        "--show-stats", action="store_true", help="Show streaming statistics"
    )
    stream_parser.add_argument(
        "--mcp-config", type=Path, help="MCP servers configuration file"
    )

    # MCP Auto-approval options
    stream_parser.add_argument(
        "--approval-strategy",
        choices=["allowlist", "patterns", "all", "none"],
        help="MCP tool approval strategy",
    )
    stream_parser.add_argument(
        "--approval-allowlist", nargs="+", help="List of allowed MCP tools"
    )
    stream_parser.add_argument(
        "--approval-allow-patterns", nargs="+", help="Regex patterns for allowed tools"
    )
    stream_parser.add_argument(
        "--approval-deny-patterns", nargs="+", help="Regex patterns for denied tools"
    )

    # Session command
    session_parser = subparsers.add_parser("session", help="Interactive session")
    session_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    session_parser.add_argument("--max-turns", type=int, help="Maximum session turns")

    # MCP Auto-approval options
    session_parser.add_argument(
        "--approval-strategy",
        choices=["allowlist", "patterns", "all", "none"],
        help="MCP tool approval strategy",
    )
    session_parser.add_argument(
        "--approval-allowlist", nargs="+", help="List of allowed MCP tools"
    )
    session_parser.add_argument(
        "--approval-allow-patterns", nargs="+", help="Regex patterns for allowed tools"
    )
    session_parser.add_argument(
        "--approval-deny-patterns", nargs="+", help="Regex patterns for denied tools"
    )

    # Health command
    subparsers.add_parser("health", help="Check wrapper health")

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run performance benchmarks"
    )
    benchmark_parser.add_argument(
        "--queries", type=Path, help="File with queries to benchmark"
    )
    benchmark_parser.add_argument(
        "--iterations", type=int, default=3, help="Iterations per query"
    )

    return parser


def main() -> int:
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
        # Suppress almost everything
        logging.getLogger().setLevel(logging.CRITICAL)
        cli.logger.setLevel(logging.CRITICAL)
    elif args.verbose:
        # Enable verbose logging
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("claude_code_wrapper").setLevel(logging.INFO)
        cli.logger.setLevel(logging.DEBUG)
        # Set up a proper format for verbose mode
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            force=True,
        )

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
    if hasattr(args, "mcp_config") and args.mcp_config:
        config_dict["mcp_config_path"] = args.mcp_config  # Already a Path from argparse

    # Add approval config if provided
    approval_config = cli._build_approval_config(args)
    if approval_config:
        config_dict["mcp_auto_approval"] = approval_config

    # Disable logging by default unless verbose is set
    if not args.verbose:
        config_dict["enable_logging"] = False
        config_dict["log_level"] = logging.CRITICAL

    # Load configuration with all settings
    cli.config = (
        ClaudeCodeConfig.from_dict(config_dict) if config_dict else ClaudeCodeConfig()
    )

    # Initialize wrapper with complete config
    if not cli.initialize_wrapper(args.verbose):
        return 1

    # Execute command
    try:
        if args.command == "ask":
            kwargs = {}
            if args.timeout:
                kwargs["timeout"] = args.timeout
            if args.max_turns:
                kwargs["max_turns"] = args.max_turns
            if args.session_id:
                kwargs["session_id"] = args.session_id
            if getattr(args, "continue", False):
                kwargs["continue_session"] = True

            kwargs["show_metadata"] = args.show_metadata

            return cli.cmd_ask(args.query, args.format, **kwargs)

        elif args.command == "stream":
            kwargs = {"verbose": args.verbose}
            if args.timeout:
                kwargs["timeout"] = args.timeout
            kwargs["show_stats"] = args.show_stats

            return cli.cmd_stream(args.query, **kwargs)

        elif args.command == "session":
            kwargs = {"verbose": args.verbose}
            if args.max_turns:
                kwargs["max_turns"] = args.max_turns

            # Add approval config if provided
            approval_config = cli._build_approval_config(args)
            if approval_config:
                kwargs["mcp_auto_approval"] = approval_config

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
