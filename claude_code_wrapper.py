"""
Claude Code SDK Wrapper - Production Ready

Enterprise-grade Python wrapper around the Claude Code SDK with comprehensive
error handling, observability, resilience patterns, and industry best practices.
"""

import json
import subprocess
import time
import logging
import signal
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Callable
from contextlib import contextmanager
from abc import ABC, abstractmethod
import functools
import os


# Logging configuration
class ClaudeCodeLogger:
    """Centralized logging configuration for Claude Code operations."""
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Set up structured logging with consistent format."""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '[%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(level)
        
        return logger


# Configuration and Constants
class OutputFormat(Enum):
    """Supported output formats for Claude Code."""
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Custom Exceptions with proper inheritance hierarchy
class ClaudeCodeError(Exception):
    """Base exception for Claude Code wrapper errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Raised when Claude Code execution times out."""
    
    def __init__(self, timeout_duration: float, context: Optional[Dict[str, Any]] = None):
        message = f"Claude Code execution timed out after {timeout_duration}s"
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.timeout_duration = timeout_duration


class ClaudeCodeProcessError(ClaudeCodeError):
    """Raised when Claude Code process fails."""
    
    def __init__(self, message: str, returncode: int, stderr: str = "", 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.returncode = returncode
        self.stderr = stderr


class ClaudeCodeValidationError(ClaudeCodeError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = "", value: Any = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, context)
        self.field = field
        self.value = value


class ClaudeCodeConfigurationError(ClaudeCodeError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_field: str = "", 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.config_field = config_field


# Response Models
@dataclass
class ClaudeCodeMetrics:
    """Metrics and telemetry data from Claude Code execution."""
    cost_usd: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    total_cost: float = 0.0
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None


@dataclass
class ClaudeCodeResponse:
    """Structured response from Claude Code execution with comprehensive metadata."""
    content: str
    returncode: int
    session_id: Optional[str] = None
    is_error: bool = False
    error_type: Optional[str] = None
    error_subtype: Optional[str] = None
    metrics: ClaudeCodeMetrics = field(default_factory=ClaudeCodeMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


# Configuration with validation
@dataclass
class ClaudeCodeConfig:
    """Production-ready configuration with validation and defaults."""
    
    # Core settings
    claude_binary: str = "claude"
    timeout: Optional[float] = 60.0
    max_turns: Optional[int] = None
    verbose: bool = False
    
    # Session management
    session_id: Optional[str] = None
    continue_session: bool = False
    
    # System prompts
    system_prompt: Optional[str] = None
    append_system_prompt: Optional[str] = None
    
    # Tool configuration
    allowed_tools: List[str] = field(default_factory=list)
    disallowed_tools: List[str] = field(default_factory=list)
    
    # MCP configuration
    mcp_config_path: Optional[Path] = None
    permission_prompt_tool: Optional[str] = None
    
    # Environment
    working_directory: Optional[Path] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    # Resilience settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0
    
    # Observability
    enable_metrics: bool = True
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.timeout is not None and self.timeout <= 0:
            raise ClaudeCodeConfigurationError(
                "Timeout must be positive", "timeout", {"value": self.timeout}
            )
        
        if self.max_turns is not None and self.max_turns <= 0:
            raise ClaudeCodeConfigurationError(
                "Max turns must be positive", "max_turns", {"value": self.max_turns}
            )
        
        if self.max_retries < 0:
            raise ClaudeCodeConfigurationError(
                "Max retries cannot be negative", "max_retries", {"value": self.max_retries}
            )
        
        if self.retry_delay < 0:
            raise ClaudeCodeConfigurationError(
                "Retry delay cannot be negative", "retry_delay", {"value": self.retry_delay}
            )
        
        if self.mcp_config_path and not self.mcp_config_path.exists():
            raise ClaudeCodeConfigurationError(
                f"MCP config file not found: {self.mcp_config_path}", 
                "mcp_config_path"
            )
        
        if self.working_directory and not self.working_directory.exists():
            raise ClaudeCodeConfigurationError(
                f"Working directory not found: {self.working_directory}", 
                "working_directory"
            )


# Response Parser Interface
class ResponseParser(ABC):
    """Abstract base class for response parsers."""
    
    @abstractmethod
    def parse(self, raw_output: str, output_format: OutputFormat) -> ClaudeCodeResponse:
        """Parse raw output into structured response."""
        pass


class ClaudeCodeResponseParser(ResponseParser):
    """Production parser for Claude Code responses with comprehensive error handling."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def parse(self, raw_output: str, output_format: OutputFormat) -> ClaudeCodeResponse:
        """Parse Claude Code response with proper error handling."""
        start_time = time.time()
        
        try:
            if output_format == OutputFormat.JSON:
                return self._parse_json_response(raw_output)
            else:
                return self._parse_text_response(raw_output)
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}", exc_info=True)
            # Return error response instead of raising
            return ClaudeCodeResponse(
                content=f"Failed to parse response: {e}",
                returncode=1,
                is_error=True,
                error_type="parsing_error",
                raw_output=raw_output,
                execution_time=time.time() - start_time
            )
    
    def _parse_json_response(self, raw_output: str) -> ClaudeCodeResponse:
        """Parse JSON response from Claude Code."""
        try:
            data = json.loads(raw_output.strip())
            self.logger.debug(f"Parsed JSON structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
            # Handle JSON arrays by extracting the most relevant response
            if isinstance(data, list):
                self.logger.debug(f"Received JSON array with {len(data)} items")
                data = self._extract_from_json_array(data)
            elif not isinstance(data, dict):
                # Handle other non-dict types by converting to dict
                self.logger.debug(f"Converting {type(data).__name__} to dict format")
                data = {"result": str(data)}
            
            # Extract content from the 'result' field (based on your output)
            content = ""
            if "result" in data and data["result"]:
                result = data["result"]
                if isinstance(result, str):
                    content = result
                elif isinstance(result, dict):
                    # Handle nested result structure
                    content = result.get("content", str(result))
                else:
                    content = str(result)
            
            # Fallback content extraction
            if not content:
                for field in ["content", "response", "text", "message", "output"]:
                    if field in data and data[field]:
                        content = str(data[field])
                        break
            
            # If still no content but there's an error, use error message
            if not content and data.get("is_error"):
                content = f"Error occurred: {data.get('error_message', 'Unknown error')}"
            
            # Extract metrics
            metrics = ClaudeCodeMetrics(
                cost_usd=float(data.get("cost_usd", 0)),
                duration_ms=int(data.get("duration_ms", 0)),
                duration_api_ms=int(data.get("duration_api_ms", 0)),
                num_turns=int(data.get("num_turns", 0)),
                total_cost=float(data.get("total_cost", 0))
            )
            
            return ClaudeCodeResponse(
                content=content,
                returncode=0,
                session_id=data.get("session_id"),
                is_error=bool(data.get("is_error", False)),
                error_type=data.get("type") if data.get("is_error") else None,
                error_subtype=data.get("subtype") if data.get("is_error") else None,
                metrics=metrics,
                metadata={k: v for k, v in data.items() if k not in [
                    "result", "session_id", "is_error", "type", "subtype",
                    "cost_usd", "duration_ms", "duration_api_ms", "num_turns", "total_cost"
                ]},
                raw_output=raw_output
            )
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON response: {e}")
            # Graceful fallback to text parsing
            return self._parse_text_response(raw_output)
    
    def _extract_from_json_array(self, data: list) -> dict:
        """Extract the most relevant response from a JSON array."""
        if not data:
            return {"result": "Empty response array"}
        
        # Strategy 1: Look for the last non-system message
        for item in reversed(data):
            if isinstance(item, dict):
                # Skip system/metadata messages
                if item.get("type") not in ["init", "system", "metadata"]:
                    return item
        
        # Strategy 2: Look for items with content
        for item in data:
            if isinstance(item, dict) and any(key in item for key in ["result", "content", "response", "message"]):
                return item
        
        # Strategy 3: Use the last item if it's a dict
        if isinstance(data[-1], dict):
            return data[-1]
        
        # Strategy 4: Create a synthetic response from the array
        return {
            "result": str(data[-1]) if data else "No content",
            "_array_length": len(data),
            "_original_array": data
        }
    
    def _parse_text_response(self, raw_output: str) -> ClaudeCodeResponse:
        """Parse text response from Claude Code."""
        return ClaudeCodeResponse(
            content=raw_output.strip(),
            returncode=0,
            raw_output=raw_output
        )


# Circuit Breaker Pattern
class CircuitBreaker:
    """Circuit breaker pattern for resilient external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise ClaudeCodeError(
                        "Circuit breaker is OPEN - service unavailable",
                        ErrorSeverity.HIGH
                    )
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise


# Retry Decorator with Exponential Backoff
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      backoff_factor: float = 2.0, max_delay: float = 60.0):
    """Retry decorator with exponential backoff."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ClaudeCodeTimeoutError, ClaudeCodeProcessError) as e:
                    if attempt == max_retries:
                        raise
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logging.getLogger(__name__).warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
        return wrapper
    return decorator


# Main Wrapper Class
class ClaudeCodeWrapper:
    """
    Production-ready wrapper around Claude Code SDK.
    
    Features:
    - Comprehensive error handling and graceful degradation
    - Circuit breaker pattern for resilience
    - Retry mechanisms with exponential backoff
    - Structured logging and observability
    - Input validation and sanitization
    - Session management and state tracking
    - Metrics collection and monitoring
    """
    
    def __init__(self, config: Optional[ClaudeCodeConfig] = None):
        """Initialize wrapper with production-ready defaults."""
        self.config = config or ClaudeCodeConfig()
        self.logger = ClaudeCodeLogger.setup_logger(__name__, self.config.log_level)
        self.parser = ClaudeCodeResponseParser(self.logger)
        self.circuit_breaker = CircuitBreaker()
        self._session_state: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        
        # Validate binary availability
        self._validate_binary()
        
        self.logger.info(f"Claude Code Wrapper initialized with config: {self.config}")
    
    def _validate_binary(self):
        """Validate that Claude Code binary is available."""
        try:
            result = subprocess.run(
                [self.config.claude_binary, "--help"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ClaudeCodeConfigurationError(
                    f"Claude binary not working properly: {result.stderr.decode()}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise ClaudeCodeConfigurationError(
                f"Claude binary not found or not executable: {self.config.claude_binary}"
            ) from e
    
    def run(self, query: str, output_format: OutputFormat = OutputFormat.TEXT, 
            **kwargs) -> ClaudeCodeResponse:
        """
        Execute Claude Code with comprehensive error handling.
        
        Args:
            query: The query/prompt to send to Claude Code
            output_format: Output format (text, json, or stream-json)
            **kwargs: Additional configuration overrides
            
        Returns:
            ClaudeCodeResponse with structured result and metadata
        """
        # Input validation
        if not query or not query.strip():
            raise ClaudeCodeValidationError("Query cannot be empty", "query", query)
        
        if len(query) > 100000:  # Reasonable limit
            raise ClaudeCodeValidationError(
                "Query too long (max 100k characters)", "query", len(query)
            )
        
        # Merge configuration
        config = self._merge_config(**kwargs)
        
        # Apply retry logic
        @retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=config.retry_delay,
            backoff_factor=config.retry_backoff_factor
        )
        def _execute():
            return self.circuit_breaker.call(self._execute_single, query, output_format, config)
        
        return _execute()
    
    def _execute_single(self, query: str, output_format: OutputFormat, 
                       config: ClaudeCodeConfig) -> ClaudeCodeResponse:
        """Execute single Claude Code call."""
        start_time = time.time()
        
        try:
            # Build and validate command
            cmd = self._build_command(query, output_format, config)
            self.logger.debug(f"Executing command: {' '.join(cmd[:3])}... (truncated)")
            
            # Execute with timeout handling
            result = self._execute_command(cmd, config)
            
            # Parse response
            response = self.parser.parse(result.stdout, output_format)
            response.returncode = result.returncode
            response.stderr = result.stderr
            response.execution_time = time.time() - start_time
            
            # Update metrics
            if config.enable_metrics:
                self._update_metrics(response)
            
            self.logger.info(f"Command executed successfully in {response.execution_time:.2f}s")
            return response
            
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Command timed out after {config.timeout}s")
            raise ClaudeCodeTimeoutError(
                config.timeout or 0,
                {"query_length": len(query), "format": output_format.value}
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
            self.logger.error(f"Command failed with code {e.returncode}: {stderr}")
            raise ClaudeCodeProcessError(
                f"Claude Code process failed with return code {e.returncode}",
                e.returncode,
                stderr,
                {"query_length": len(query), "format": output_format.value}
            )
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ClaudeCodeError(
                f"Unexpected error during execution: {e}",
                ErrorSeverity.HIGH,
                {"query_length": len(query), "format": output_format.value}
            )
    
    def run_streaming(self, query: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Execute Claude Code with streaming output and graceful error handling.
        
        Args:
            query: The query/prompt to send to Claude Code
            **kwargs: Additional configuration overrides
            
        Yields:
            Dict: Each message/event from the streaming response
        """
        if not query or not query.strip():
            self.logger.error("Empty query provided for streaming")
            yield {"type": "error", "message": "Query cannot be empty"}
            return
        
        config = self._merge_config(**kwargs)
        cmd = self._build_command(query, OutputFormat.STREAM_JSON, config)
        
        process = None
        try:
            self.logger.info("Starting streaming execution")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=config.working_directory,
                env=self._build_env(config)
            )
            
            # Set up timeout handling if configured
            if config.timeout:
                def timeout_handler():
                    time.sleep(config.timeout)
                    if process and process.poll() is None:
                        self.logger.warning("Streaming process timed out, terminating")
                        process.terminate()
                
                timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                timeout_thread.start()
            
            # Stream output with error handling
            line_count = 0
            for line in process.stdout:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        line_count += 1
                        yield event
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse streaming line: {line[:100]}...")
                        yield {
                            "type": "parse_error",
                            "message": f"Invalid JSON in stream: {e}",
                            "raw_line": line
                        }
            
            # Wait for process completion
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read() if process.stderr else ""
                self.logger.error(f"Streaming process failed with code {process.returncode}: {stderr}")
                yield {
                    "type": "error",
                    "message": f"Process failed with return code {process.returncode}",
                    "stderr": stderr,
                    "returncode": process.returncode
                }
            else:
                self.logger.info(f"Streaming completed successfully with {line_count} events")
                
        except Exception as e:
            self.logger.error(f"Streaming execution failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Streaming failed: {e}",
                "error_type": type(e).__name__
            }
        finally:
            # Ensure process cleanup
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
    
    def _merge_config(self, **kwargs) -> ClaudeCodeConfig:
        """Merge base config with overrides."""
        merged_dict = {
            'claude_binary': kwargs.get('claude_binary', self.config.claude_binary),
            'timeout': kwargs.get('timeout', self.config.timeout),
            'max_turns': kwargs.get('max_turns', self.config.max_turns),
            'verbose': kwargs.get('verbose', self.config.verbose),
            'session_id': kwargs.get('session_id', self.config.session_id),
            'continue_session': kwargs.get('continue_session', self.config.continue_session),
            'system_prompt': kwargs.get('system_prompt', self.config.system_prompt),
            'append_system_prompt': kwargs.get('append_system_prompt', self.config.append_system_prompt),
            'allowed_tools': kwargs.get('allowed_tools', self.config.allowed_tools.copy()),
            'disallowed_tools': kwargs.get('disallowed_tools', self.config.disallowed_tools.copy()),
            'mcp_config_path': kwargs.get('mcp_config_path', self.config.mcp_config_path),
            'permission_prompt_tool': kwargs.get('permission_prompt_tool', self.config.permission_prompt_tool),
            'working_directory': kwargs.get('working_directory', self.config.working_directory),
            'environment_vars': kwargs.get('environment_vars', self.config.environment_vars.copy()),
            'max_retries': kwargs.get('max_retries', self.config.max_retries),
            'retry_delay': kwargs.get('retry_delay', self.config.retry_delay),
            'retry_backoff_factor': kwargs.get('retry_backoff_factor', self.config.retry_backoff_factor),
            'enable_metrics': kwargs.get('enable_metrics', self.config.enable_metrics),
            'log_level': kwargs.get('log_level', self.config.log_level)
        }
        return ClaudeCodeConfig(**merged_dict)
    
    def _build_command(self, query: str, output_format: OutputFormat, 
                      config: ClaudeCodeConfig) -> List[str]:
        """Build Claude Code command with validation."""
        cmd = [config.claude_binary, "--print", query]
        
        if output_format == OutputFormat.STREAM_JSON:
            cmd.extend(["--output-format", output_format.value])
            cmd.append("--verbose")  # Required by Claude Code for streaming JSON
        elif output_format != OutputFormat.TEXT:
            cmd.extend(["--output-format", output_format.value])
        
        if config.session_id:
            cmd.extend(["--resume", config.session_id])
        elif config.continue_session:
            cmd.append("--continue")
        
        if config.system_prompt:
            cmd.extend(["--system-prompt", config.system_prompt])
        if config.append_system_prompt:
            cmd.extend(["--append-system-prompt", config.append_system_prompt])
        
        if config.max_turns:
            cmd.extend(["--max-turns", str(config.max_turns)])
        
        if config.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(config.allowed_tools)])
        if config.disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(config.disallowed_tools)])
        
        if config.mcp_config_path:
            cmd.extend(["--mcp-config", str(config.mcp_config_path)])
        if config.permission_prompt_tool:
            cmd.extend(["--permission-prompt-tool", config.permission_prompt_tool])
        
        if config.verbose:
            cmd.append("--verbose")
        
        return cmd
    
    def _execute_command(self, cmd: List[str], config: ClaudeCodeConfig) -> subprocess.CompletedProcess:
        """Execute command with proper error handling."""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            cwd=config.working_directory,
            env=self._build_env(config),
            check=True
        )
    
    def _build_env(self, config: ClaudeCodeConfig) -> Optional[Dict[str, str]]:
        """Build environment variables."""
        if not config.environment_vars:
            return None
        
        env = os.environ.copy()
        env.update(config.environment_vars)
        return env
    
    def _update_metrics(self, response: ClaudeCodeResponse):
        """Update internal metrics."""
        self._metrics.setdefault("total_requests", 0)
        self._metrics["total_requests"] += 1
        
        if response.is_error:
            self._metrics.setdefault("error_count", 0)
            self._metrics["error_count"] += 1
        
        self._metrics.setdefault("total_execution_time", 0)
        self._metrics["total_execution_time"] += response.execution_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self._metrics.copy()
    
    def resume_session(self, session_id: str, query: str, **kwargs) -> ClaudeCodeResponse:
        """Resume a specific session."""
        return self.run(query, session_id=session_id, **kwargs)
    
    def continue_last_session(self, query: str, **kwargs) -> ClaudeCodeResponse:
        """Continue the most recent session."""
        return self.run(query, continue_session=True, **kwargs)
    
    @contextmanager
    def session(self, **session_config):
        """Context manager for session-based conversations."""
        session = ClaudeCodeSession(self, **session_config)
        try:
            yield session
        finally:
            session.cleanup()


class ClaudeCodeSession:
    """Session wrapper for multi-turn conversations with state management."""
    
    def __init__(self, wrapper: ClaudeCodeWrapper, **config):
        self.wrapper = wrapper
        self.config = config
        self.session_id: Optional[str] = None
        self.history: List[ClaudeCodeResponse] = []
        self.logger = ClaudeCodeLogger.setup_logger(f"{__name__}.session")
    
    def ask(self, query: str, **kwargs) -> ClaudeCodeResponse:
        """Ask a question in the current session with error handling."""
        try:
            merged_config = {**self.config, **kwargs}
            
            if self.session_id:
                merged_config['session_id'] = self.session_id
            elif self.history:
                merged_config['continue_session'] = True
            
            response = self.wrapper.run(query, **merged_config)
            
            if response.session_id:
                self.session_id = response.session_id
            
            self.history.append(response)
            self.logger.info(f"Session query completed. Total exchanges: {len(self.history)}")
            return response
            
        except Exception as e:
            self.logger.error(f"Session query failed: {e}")
            # Create error response instead of failing
            error_response = ClaudeCodeResponse(
                content=f"Session error: {e}",
                returncode=1,
                is_error=True,
                error_type="session_error"
            )
            self.history.append(error_response)
            return error_response
    
    def ask_streaming(self, query: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Ask with streaming response in session context."""
        merged_config = {**self.config, **kwargs}
        
        if self.session_id:
            merged_config['session_id'] = self.session_id
        elif self.history:
            merged_config['continue_session'] = True
        
        yield from self.wrapper.run_streaming(query, **merged_config)
    
    def get_history(self) -> List[ClaudeCodeResponse]:
        """Get conversation history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
        self.session_id = None
        self.logger.info("Session history cleared")
    
    def cleanup(self):
        """Clean up session resources."""
        self.logger.info(f"Session cleanup completed. Total exchanges: {len(self.history)}")


# Convenience functions with error handling
def ask_claude(query: str, **kwargs) -> ClaudeCodeResponse:
    """Quick function to ask Claude with error handling."""
    try:
        wrapper = ClaudeCodeWrapper()
        return wrapper.run(query, **kwargs)
    except Exception as e:
        logger = ClaudeCodeLogger.setup_logger(__name__)
        logger.error(f"Quick ask failed: {e}")
        return ClaudeCodeResponse(
            content=f"Error: {e}",
            returncode=1,
            is_error=True,
            error_type="convenience_function_error"
        )


def ask_claude_json(query: str, **kwargs) -> ClaudeCodeResponse:
    """Quick function to ask Claude with JSON output."""
    try:
        wrapper = ClaudeCodeWrapper()
        return wrapper.run(query, output_format=OutputFormat.JSON, **kwargs)
    except Exception as e:
        logger = ClaudeCodeLogger.setup_logger(__name__)
        logger.error(f"Quick JSON ask failed: {e}")
        return ClaudeCodeResponse(
            content=f"Error: {e}",
            returncode=1,
            is_error=True,
            error_type="convenience_function_error"
        )


def ask_claude_streaming(query: str, **kwargs) -> Iterator[Dict[str, Any]]:
    """Quick function for streaming with comprehensive error handling."""
    try:
        wrapper = ClaudeCodeWrapper()
        yield from wrapper.run_streaming(query, **kwargs)
    except Exception as e:
        logger = ClaudeCodeLogger.setup_logger(__name__)
        logger.error(f"Quick streaming ask failed: {e}")
        yield {
            "type": "error",
            "message": f"Streaming failed: {e}",
            "error_type": type(e).__name__
        }
