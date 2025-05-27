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
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Callable
from contextlib import contextmanager
from abc import ABC, abstractmethod
import functools
import os
import sys
import tempfile
import shutil

# Import approval system components
try:
    from approval_strategies import ApprovalStrategy, create_strategy
    HAS_APPROVAL_SYSTEM = True
except ImportError:
    HAS_APPROVAL_SYSTEM = False


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
    retries: int = 0
    
    @property
    def success(self) -> bool:
        """Check if the response was successful."""
        return self.returncode == 0 and not self.is_error
    
    @property
    def duration(self) -> float:
        """Alias for execution_time for backwards compatibility."""
        return self.execution_time
    
    @property
    def exit_code(self) -> int:
        """Alias for returncode for backwards compatibility."""
        return self.returncode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return asdict(self)


# MCP Configuration Models
@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # stdio or sse
    url: Optional[str] = None  # For SSE transport
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP config format."""
        config = {
            "command": self.command,
            "args": self.args
        }
        if self.env:
            config["env"] = self.env
        return config


@dataclass
class MCPConfig:
    """Complete MCP configuration."""
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'MCPConfig':
        """Load MCP configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        servers = {}
        for name, server_data in data.get("mcpServers", {}).items():
            servers[name] = MCPServerConfig(
                name=name,
                command=server_data["command"],
                args=server_data.get("args", []),
                env=server_data.get("env", {})
            )
        
        return cls(servers=servers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP JSON format."""
        return {
            "mcpServers": {
                name: server.to_dict()
                for name, server in self.servers.items()
            }
        }
    
    def save(self, file_path: Union[str, Path]):
        """Save MCP configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Configuration with validation
@dataclass
class ClaudeCodeConfig:
    """Production-ready configuration with validation and defaults."""
    
    # Core settings
    claude_binary: str = "claude"
    timeout: Optional[float] = 300.0  # 5 minutes for long-running MCP tools
    max_turns: Optional[int] = None
    verbose: bool = False
    
    # Streaming timeout settings (industry best practices)
    streaming_idle_timeout: Optional[float] = 30.0  # Reset on each event
    streaming_max_timeout: Optional[float] = 600.0  # 10 min absolute max
    streaming_initial_timeout: Optional[float] = 60.0  # Time to first event
    
    # Model selection
    model: Optional[str] = None  # opus, sonnet, haiku, or full model name
    
    # Generation parameters
    temperature: Optional[float] = None  # 0.0-1.0, controls randomness
    max_tokens: Optional[int] = None     # Maximum response length
    top_p: Optional[float] = None        # Nucleus sampling parameter
    stop_sequences: Optional[List[str]] = field(default_factory=list)
    
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
    mcp_config: Optional[MCPConfig] = field(default=None, init=False)  # Loaded MCP config
    permission_prompt_tool: Optional[str] = None
    mcp_allowed_servers: List[str] = field(default_factory=list)
    mcp_scope: Optional[str] = None  # 'local', 'project', 'user'
    use_existing_mcp_servers: bool = True  # Use pre-configured servers  # Specific MCP servers to allow
    
    # MCP Auto-approval configuration
    mcp_auto_approval: Dict[str, Any] = field(default_factory=dict)
    # Example: {
    #   "enabled": true,
    #   "strategy": "allowlist",  # "all", "none", "allowlist", "patterns"
    #   "allowlist": ["mcp__tool__*"],
    #   "allow_patterns": ["mcp__.*__read.*"],
    #   "deny_patterns": ["mcp__.*__write.*"]
    # }
    
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
    
    # Caching
    cache_responses: bool = False
    cache_ttl: float = 1800.0  # 30 minutes default (balanced between freshness and efficiency)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Load MCP config if path provided
        if self.mcp_config_path:
            try:
                self.mcp_config = MCPConfig.from_file(self.mcp_config_path)
            except Exception as e:
                raise ClaudeCodeConfigurationError(
                    f"Failed to load MCP config: {e}",
                    "mcp_config_path"
                )
        self._validate()
    
    def validate(self):
        """Public method to validate configuration."""
        self._validate()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClaudeCodeConfig':
        """Create configuration from dictionary."""
        # Filter out any keys that aren't valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        # Convert path strings to Path objects
        if 'mcp_config_path' in filtered_dict and filtered_dict['mcp_config_path'] is not None:
            filtered_dict['mcp_config_path'] = Path(filtered_dict['mcp_config_path'])
        if 'working_directory' in filtered_dict and filtered_dict['working_directory'] is not None:
            filtered_dict['working_directory'] = Path(filtered_dict['working_directory'])
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'ClaudeCodeConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Remove the loaded mcp_config object (keep only the path)
        if 'mcp_config' in data:
            del data['mcp_config']
        return data
    
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
        
        if self.cache_ttl <= 0:
            raise ClaudeCodeConfigurationError(
                "Cache TTL must be positive", "cache_ttl", {"value": self.cache_ttl}
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
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"


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
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_retries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._sessions: Dict[str, ClaudeCodeSession] = {}
        self._cache: Dict[str, Any] = {}
        self._temp_mcp_config_path: Optional[str] = None
        
        # Validate binary availability
        self._validate_binary()
        
        self.logger.info(f"Claude Code Wrapper initialized with config: {self.config}")
    
    def validate_prompt(self, prompt: Any) -> None:
        """Validate prompt input."""
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            raise ClaudeCodeValidationError("Query cannot be empty", "prompt", prompt)
        
        if isinstance(prompt, str) and len(prompt) > 100000:
            raise ClaudeCodeValidationError(
                "Query too long (max 100k characters)", "prompt", len(prompt)
            )
    
    # Compatibility alias
    _validate_prompt = validate_prompt
    
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
        
        # Check cache if enabled
        if config.cache_responses:
            cache_key = self._generate_cache_key(query, kwargs)
            if cache_key in self._cache:
                entry, timestamp = self._cache[cache_key]
                if time.time() - timestamp < config.cache_ttl:
                    self._metrics["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return entry
            self._metrics["cache_misses"] += 1
            self.logger.debug(f"Cache miss for query: {query[:50]}...")
        
        # Apply retry logic
        @retry_with_backoff(
            max_retries=config.max_retries,
            base_delay=config.retry_delay,
            backoff_factor=config.retry_backoff_factor
        )
        def _execute():
            return self.circuit_breaker.call(self._execute_single, query, output_format, config)
        
        response = _execute()
        
        # Cache successful response if caching is enabled
        if config.cache_responses and response.success:
            cache_key = self._generate_cache_key(query, kwargs)
            self._cache[cache_key] = (response, time.time())
            self.logger.debug(f"Cached response for query: {query[:50]}...")
        
        return response
    
    def _execute_single(self, query: str, output_format: OutputFormat, 
                       config: ClaudeCodeConfig) -> ClaudeCodeResponse:
        """Execute single Claude Code call."""
        start_time = time.time()
        temp_mcp_config = None
        
        try:
            # Setup approval server if needed
            if config.mcp_auto_approval.get('enabled', False):
                temp_mcp_config = self._setup_approval_server(config)
                if temp_mcp_config:
                    # Get the allowed tools from approval config
                    allowed_tools = config.allowed_tools.copy() if config.allowed_tools else []
                    approval_config = config.mcp_auto_approval
                    
                    # Add tools based on strategy
                    if approval_config.get('strategy') == 'allowlist':
                        allowed_tools.extend(approval_config.get('allowlist', []))
                    elif approval_config.get('strategy') == 'all':
                        # For 'all' strategy, we need to allow all MCP tools
                        allowed_tools.append('mcp__*')
                    
                    # Override config with combined MCP config and allowed tools
                    config = ClaudeCodeConfig.from_dict({
                        **config.to_dict(),
                        'mcp_config_path': temp_mcp_config,
                        'permission_prompt_tool': 'mcp__approval-server__permissions__approve',
                        'allowed_tools': allowed_tools
                    })
            
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
            stderr = e.stderr if e.stderr else ""
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
        finally:
            # Clean up approval server if it was started
            if temp_mcp_config:
                self._cleanup_approval_server()
    
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
        temp_mcp_config = None
        
        # Setup approval server if needed (same as in _execute_single)
        if config.mcp_auto_approval.get('enabled', False):
            temp_mcp_config = self._setup_approval_server(config)
            if temp_mcp_config:
                # Get the allowed tools from approval config
                allowed_tools = config.allowed_tools.copy() if config.allowed_tools else []
                approval_config = config.mcp_auto_approval
                
                # Add tools based on strategy
                if approval_config.get('strategy') == 'allowlist':
                    allowed_tools.extend(approval_config.get('allowlist', []))
                elif approval_config.get('strategy') == 'all':
                    # For 'all' strategy, we need to allow all MCP tools
                    allowed_tools.append('mcp__*')
                
                # Override config with combined MCP config and allowed tools
                config = ClaudeCodeConfig.from_dict({
                    **config.to_dict(),
                    'mcp_config_path': temp_mcp_config,
                    'permission_prompt_tool': 'mcp__approval-server__permissions__approve',
                    'allowed_tools': allowed_tools
                })
        
        cmd = self._build_command(query, OutputFormat.STREAM_JSON, config)
        # self.logger.info(f"Executing streaming command: {' '.join(cmd)}")  # Commented to avoid slowdown
        
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
            
            # Set up activity-based timeout handling (industry best practice)
            last_activity = time.time()
            start_time = time.time()
            timeout_thread = None
            timeout_lock = threading.Lock()
            
            def activity_timeout_handler():
                """Industry-standard activity-based timeout with three phases"""
                nonlocal last_activity, process
                
                while True:
                    with timeout_lock:
                        current_time = time.time()
                        time_since_start = current_time - start_time
                        time_since_activity = current_time - last_activity
                        
                        # Phase 1: Initial timeout (time to first event)
                        if time_since_start < config.streaming_initial_timeout and time_since_activity < config.streaming_initial_timeout:
                            time.sleep(1)
                            continue
                        
                        # Phase 2: Idle timeout (reset on each event)
                        if time_since_activity > config.streaming_idle_timeout:
                            if process and process.poll() is None:
                                self.logger.warning(f"Streaming idle timeout after {time_since_activity:.1f}s, terminating")
                                process.terminate()
                                return
                        
                        # Phase 3: Absolute maximum timeout
                        if time_since_start > config.streaming_max_timeout:
                            if process and process.poll() is None:
                                self.logger.warning(f"Streaming maximum timeout after {time_since_start:.1f}s, terminating")
                                process.terminate()
                                return
                        
                        # Check if process has completed
                        if not process or process.poll() is not None:
                            return
                    
                    time.sleep(1)  # Check every second
            
            if config.streaming_idle_timeout:
                timeout_thread = threading.Thread(target=activity_timeout_handler, daemon=True)
                timeout_thread.start()
            
            # Stream output with error handling
            line_count = 0
            for line in process.stdout:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        line_count += 1
                        
                        # Reset activity timer on each event (industry best practice)
                        with timeout_lock:
                            last_activity = time.time()
                        
                        yield event
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse streaming line: {line[:100]}...")
                        
                        # Reset activity timer even for parse errors
                        with timeout_lock:
                            last_activity = time.time()
                        
                        yield {
                            "type": "parse_error",
                            "message": f"Invalid JSON in stream: {e}",
                            "raw_line": line
                        }
            
            # Check if process is still running
            returncode = process.poll()
            if returncode is None:
                # Process is still running, wait a bit more
                process.wait(timeout=5)
                returncode = process.returncode
            
            # Read any remaining stderr
            stderr = process.stderr.read() if process.stderr else ""
            
            if returncode != 0:
                self.logger.error(f"Streaming process failed with code {returncode}: {stderr}")
                yield {
                    "type": "error",
                    "message": f"Process failed with return code {returncode}",
                    "stderr": stderr,
                    "returncode": returncode
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
            
            # Clean up approval server if it was started
            if temp_mcp_config:
                self._cleanup_approval_server()
    
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
            'mcp_auto_approval': kwargs.get('mcp_auto_approval', self.config.mcp_auto_approval.copy()),
            'working_directory': kwargs.get('working_directory', self.config.working_directory),
            'environment_vars': kwargs.get('environment_vars', self.config.environment_vars.copy()),
            'max_retries': kwargs.get('max_retries', self.config.max_retries),
            'retry_delay': kwargs.get('retry_delay', self.config.retry_delay),
            'retry_backoff_factor': kwargs.get('retry_backoff_factor', self.config.retry_backoff_factor),
            'enable_metrics': kwargs.get('enable_metrics', self.config.enable_metrics),
            'log_level': kwargs.get('log_level', self.config.log_level),
            # Model selection and generation parameters
            'model': kwargs.get('model', self.config.model),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'stop_sequences': kwargs.get('stop_sequences', self.config.stop_sequences.copy() if self.config.stop_sequences else []),
            # Caching parameters
            'cache_responses': kwargs.get('cache_responses', self.config.cache_responses),
            'cache_ttl': kwargs.get('cache_ttl', self.config.cache_ttl)
        }
        return ClaudeCodeConfig(**merged_dict)
    
    def _build_command(self, query: str, output_format: OutputFormat, 
                      config: ClaudeCodeConfig) -> List[str]:
        """Build Claude Code command with validation."""
        # Use -p flag when permission_prompt_tool is set (for MCP non-interactive mode)
        if config.permission_prompt_tool:
            cmd = [config.claude_binary, "-p", query]
        else:
            cmd = [config.claude_binary, "--print", query]
        
        if output_format == OutputFormat.STREAM_JSON:
            cmd.extend(["--output-format", output_format.value])
            cmd.append("--verbose")  # Required by Claude Code for streaming JSON
        elif output_format != OutputFormat.TEXT:
            cmd.extend(["--output-format", output_format.value])
        
        # Model selection
        if config.model:
            cmd.extend(["--model", config.model])
        
        # Generation parameters
        if config.temperature is not None:
            cmd.extend(["--temperature", str(config.temperature)])
        if config.max_tokens:
            cmd.extend(["--max-tokens", str(config.max_tokens)])
        if config.top_p is not None:
            cmd.extend(["--top-p", str(config.top_p)])
        if config.stop_sequences:
            for seq in config.stop_sequences:
                cmd.extend(["--stop-sequence", seq])
        
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
    
    def _setup_approval_server(self, config: ClaudeCodeConfig) -> Optional[Path]:
        """
        Setup MCP approval server if configured.
        
        Returns:
            Path to temporary MCP config file if approval server is setup, None otherwise
        """
        if not HAS_APPROVAL_SYSTEM:
            if config.mcp_auto_approval.get('enabled', False):
                self.logger.warning("MCP auto-approval requested but approval system not available")
            return None
        
        if not config.mcp_auto_approval.get('enabled', False):
            return None
        
        try:
            # Create strategy configuration
            strategy_config = {
                'type': config.mcp_auto_approval.get('strategy', 'allowlist'),
                'allowlist': config.mcp_auto_approval.get('allowlist', []),
                'allow_patterns': config.mcp_auto_approval.get('allow_patterns', []),
                'deny_patterns': config.mcp_auto_approval.get('deny_patterns', [])
            }
            
            # Create combined MCP config
            combined_config = {"mcpServers": {}}
            
            # Add existing MCP servers
            if config.mcp_config_path:
                existing_config = MCPConfig.from_file(config.mcp_config_path)
                combined_config["mcpServers"].update(existing_config.to_dict()["mcpServers"])
            
            # Add configurable approval server with environment variable
            combined_config["mcpServers"]["approval-server"] = {
                "command": sys.executable,
                "args": [str(Path(__file__).parent / "configurable_approval_server.py")],
                "env": {
                    "APPROVAL_STRATEGY_CONFIG": json.dumps(strategy_config),
                    "APPROVAL_LOG_PATH": str(Path(tempfile.gettempdir()) / f"approval_log_{os.getpid()}.txt")
                }
            }
            
            # Write to temporary file
            fd, self._temp_mcp_config_path = tempfile.mkstemp(suffix='.json', prefix='mcp_config_')
            with os.fdopen(fd, 'w') as f:
                json.dump(combined_config, f, indent=2)
            
            # Debug: log the config
            self.logger.debug(f"Combined MCP config: {json.dumps(combined_config, indent=2)}")
            self.logger.info(f"MCP auto-approval configured with strategy: {strategy_config['type']}")
            self.logger.info(f"Temporary MCP config written to: {self._temp_mcp_config_path}")
            
            return Path(self._temp_mcp_config_path)
            
        except Exception as e:
            self.logger.error(f"Failed to setup approval server: {e}")
            return None
    
    def _cleanup_approval_server(self):
        """Clean up temporary files."""
        if self._temp_mcp_config_path and os.path.exists(self._temp_mcp_config_path):
            try:
                os.unlink(self._temp_mcp_config_path)
                self._temp_mcp_config_path = None
                self.logger.debug("Removed temporary MCP config file")
            except Exception as e:
                self.logger.error(f"Error removing temporary MCP config: {e}")
    
    def _update_metrics(self, response: ClaudeCodeResponse):
        """Update internal metrics."""
        self._metrics["total_requests"] += 1
        
        if response.success:
            self._metrics["successful_requests"] += 1
        else:
            self._metrics["failed_requests"] += 1
        
        if response.is_error:
            self._metrics.setdefault("error_count", 0)
            self._metrics["error_count"] += 1
        
        # Track retries
        if hasattr(response, 'retries') and response.retries > 0:
            self._metrics["total_retries"] += response.retries
        
        self._metrics.setdefault("total_execution_time", 0)
        self._metrics["total_execution_time"] += response.execution_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics including calculated values."""
        metrics = self._metrics.copy()
        
        # Calculate derived metrics
        total = metrics.get("total_requests", 0)
        if total > 0:
            # Success rate
            successful = metrics.get("successful_requests", 0)
            metrics["success_rate"] = successful / total
            
            # Average retries per request
            total_retries = metrics.get("total_retries", 0)
            metrics["average_retries_per_request"] = total_retries / total
            
            # Average execution time
            total_time = metrics.get("total_execution_time", 0)
            metrics["average_execution_time"] = total_time / total
        else:
            # Set defaults when no requests
            metrics["success_rate"] = 0.0
            metrics["average_retries_per_request"] = 0.0
            metrics["average_execution_time"] = 0.0
        
        # Cache hit rate
        cache_hits = metrics.get("cache_hits", 0)
        cache_misses = metrics.get("cache_misses", 0)
        cache_total = cache_hits + cache_misses
        
        if cache_total > 0:
            metrics["cache_hit_rate"] = cache_hits / cache_total
        else:
            metrics["cache_hit_rate"] = 0.0
        
        return metrics
    
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
    
    # Session continuation methods
    def continue_conversation(self, query: str = "") -> ClaudeCodeResponse:
        """Continue the most recent conversation using -c flag."""
        # Set continue flag and run
        original_continue = self.config.continue_session
        self.config.continue_session = True
        try:
            response = self.run(query)
            # Update session ID if returned
            if response.session_id:
                self._session_state["last_session_id"] = response.session_id
            return response
        finally:
            self.config.continue_session = original_continue
    
    def resume_specific_session(self, session_id: str, query: str = "") -> ClaudeCodeResponse:
        """Resume a specific session using --resume flag."""
        # Set session ID and run
        original_session_id = self.config.session_id
        self.config.session_id = session_id
        try:
            response = self.run(query)
            # Track this session
            self._session_state["last_session_id"] = session_id
            return response
        finally:
            self.config.session_id = original_session_id
    
    def get_last_session_id(self) -> Optional[str]:
        """Get the ID of the last session used."""
        return self._session_state.get("last_session_id")
    
    # Additional methods for test compatibility
    def ask(self, query: str, **kwargs) -> ClaudeCodeResponse:
        """Ask Claude a question (alias for run)."""
        return self.run(query, **kwargs)
    
    def ask_streaming(self, query: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Ask Claude with streaming response (alias for run_streaming)."""
        return self.run_streaming(query, **kwargs)
    
    def ask_json(self, query: str, **kwargs) -> Any:
        """Ask Claude and parse JSON response."""
        response = self.ask(query, output_format=OutputFormat.JSON, **kwargs)
        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ClaudeCodeError(f"Failed to parse JSON response: {e}")
        else:
            raise ClaudeCodeProcessError(
                f"Command failed with code {response.returncode}",
                response.returncode
            )
    
    def stream(self, query: str, **kwargs) -> Iterator[str]:
        """Stream response chunks."""
        for event in self.ask_streaming(query, **kwargs):
            if event.get('type') == 'content':
                yield event.get('content', '')
    
    def create_session(self, session_id: Optional[str] = None) -> 'ClaudeCodeSession':
        """Create a new session."""
        import uuid
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        config = {'session_id': session_id}
        session = ClaudeCodeSession(self, **config)
        self._sessions[session_id] = session
        return session
    
    def ask_in_session(self, session_id: str, query: str, **kwargs) -> ClaudeCodeResponse:
        """Ask within a specific session."""
        return self.run(query, session_id=session_id, **kwargs)
    
    def get_sessions(self) -> Dict[str, 'ClaudeCodeSession']:
        """Get active sessions."""
        return self._sessions.copy()
    
    def _generate_cache_key(self, query: str, config_kwargs: Dict[str, Any]) -> str:
        """Generate a unique cache key for the query and configuration."""
        # Include relevant configuration parameters that affect the response
        cache_params = {
            'query': query,
            'model': config_kwargs.get('model', self.config.model),
            'temperature': config_kwargs.get('temperature', self.config.temperature),
            'max_tokens': config_kwargs.get('max_tokens', self.config.max_tokens),
            'top_p': config_kwargs.get('top_p', self.config.top_p),
            'system_prompt': config_kwargs.get('system_prompt', self.config.system_prompt),
            'output_format': config_kwargs.get('output_format', 'text'),
            # Include MCP context to avoid cache collisions
            'allowed_tools': sorted(config_kwargs.get('allowed_tools', self.config.allowed_tools or [])),
            'mcp_config_path': str(config_kwargs.get('mcp_config_path', self.config.mcp_config_path or '')),
            'session_id': config_kwargs.get('session_id', self.config.session_id),
            'timestamp': int(time.time() / 300)  # 5-minute time buckets for session context
        }
        # Create a stable string representation
        cache_str = json.dumps(cache_params, sort_keys=True)
        # Use hash for a shorter key
        import hashlib
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear response cache."""
        if hasattr(self, '_cache'):
            self._cache.clear()
            self.logger.info("Cache cleared")
    
    def close(self):
        """Clean up resources."""
        self.logger.info(f"Closing wrapper - sessions before: {len(self._sessions)}, cache before: {len(self._cache)}")
        # Clear sessions
        self._sessions.clear()
        # Clear cache
        if hasattr(self, '_cache'):
            self._cache.clear()
        # Reset circuit breaker
        self.circuit_breaker.reset()
        self.logger.info(f"Wrapper closed - sessions after: {len(self._sessions)}, cache after: {len(self._cache)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            result = subprocess.run(
                [self.config.claude_binary, "--version"],
                capture_output=True,
                timeout=5
            )
            return {
                "status": "healthy" if result.returncode == 0 else "unhealthy",
                "claude_available": result.returncode == 0,
                "version": result.stdout.decode().strip() if result.returncode == 0 else None,
                "error": result.stderr.decode().strip() if result.returncode != 0 else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "claude_available": False,
                "error": str(e)
            }
    
    # MCP Management Methods
    def get_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """Get configured MCP servers."""
        if not self.config.mcp_config:
            return {}
        return self.config.mcp_config.servers.copy()
    
    def list_available_mcp_servers(self) -> ClaudeCodeResponse:
        """List MCP servers configured in Claude Code."""
        try:
            # Use claude mcp list command
            cmd = [self.config.claude_binary, "mcp", "list"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            metrics = ClaudeCodeMetrics()
            metrics.execution_time = 0.0
            
            return ClaudeCodeResponse(
                content=result.stdout if result.returncode == 0 else result.stderr,
                returncode=result.returncode,
                is_error=result.returncode != 0,
                error_type="MCP_LIST_ERROR" if result.returncode != 0 else None,
                metrics=metrics,
                raw_output=result.stdout + result.stderr
            )
        except Exception as e:
            metrics = ClaudeCodeMetrics()
            metrics.execution_time = 0.0
            
            return ClaudeCodeResponse(
                content=str(e),
                returncode=-1,
                is_error=True,
                error_type="MCP_LIST_EXCEPTION",
                error_subtype=type(e).__name__,
                metrics=metrics
            )
    
    def get_mcp_tools(self, server_name: Optional[str] = None) -> List[str]:
        """
        Get MCP tool names in the format expected by Claude.
        
        Args:
            server_name: Optional specific server name. If None, returns all tools.
            
        Returns:
            List of tool names in format: mcp__<serverName>__<toolName>
        """
        if not self.config.mcp_config:
            return []
        
        # This is a placeholder - in reality, we'd need to query the MCP server
        # to get its available tools. For now, return common tool patterns.
        tools = []
        servers = [server_name] if server_name else self.config.mcp_config.servers.keys()
        
        for server in servers:
            if server not in self.config.mcp_config.servers:
                continue
                
            # Common MCP tools based on server type
            if "filesystem" in server.lower():
                tools.extend([
                    f"mcp__{server}__read_file",
                    f"mcp__{server}__write_file",
                    f"mcp__{server}__list_directory",
                    f"mcp__{server}__create_directory",
                    f"mcp__{server}__delete_file"
                ])
            elif "github" in server.lower():
                tools.extend([
                    f"mcp__{server}__get_repository",
                    f"mcp__{server}__list_repositories",
                    f"mcp__{server}__get_file_contents",
                    f"mcp__{server}__create_issue",
                    f"mcp__{server}__list_issues"
                ])
            else:
                # Generic tools
                tools.extend([
                    f"mcp__{server}__execute",
                    f"mcp__{server}__query",
                    f"mcp__{server}__list"
                ])
        
        return tools
    
    def allow_mcp_tools(self, server_name: str, tool_names: Optional[List[str]] = None):
        """
        Add MCP tools to allowed tools list.
        
        Args:
            server_name: MCP server name
            tool_names: Optional specific tool names. If None, allows all tools from server.
        """
        if tool_names:
            # Add specific tools
            for tool in tool_names:
                tool_id = f"mcp__{server_name}__{tool}"
                if tool_id not in self.config.allowed_tools:
                    self.config.allowed_tools.append(tool_id)
        else:
            # Add all tools from server
            tools = self.get_mcp_tools(server_name)
            for tool in tools:
                if tool not in self.config.allowed_tools:
                    self.config.allowed_tools.append(tool)
    
    def create_mcp_config(self, servers: Dict[str, MCPServerConfig]) -> MCPConfig:
        """Create a new MCP configuration."""
        return MCPConfig(servers=servers)
    
    def save_mcp_config(self, config: MCPConfig, file_path: Union[str, Path]):
        """Save MCP configuration to file."""
        config.save(file_path)


class ClaudeCodeSession:
    """Session wrapper for multi-turn conversations with state management."""
    
    def __init__(self, wrapper: ClaudeCodeWrapper, **config):
        self.wrapper = wrapper
        self.config = config
        self.session_id: Optional[str] = config.get('session_id')
        self.history: List[ClaudeCodeResponse] = []
        self.messages: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.total_duration = 0.0
        self.total_retries = 0
        self.metadata: Dict[str, Any] = {}
        self.logger = ClaudeCodeLogger.setup_logger(f"{__name__}.session")
    
    def ask(self, query: str, **kwargs) -> ClaudeCodeResponse:
        """Ask a question in the current session with error handling."""
        try:
            merged_config = {**self.config, **kwargs}
            
            if self.session_id:
                merged_config['session_id'] = self.session_id
            elif self.history:
                merged_config['continue_session'] = True
            
            # Add user message
            self.add_message("user", query)
            
            response = self.wrapper.run(query, **merged_config)
            
            if response.session_id:
                self.session_id = response.session_id
            
            # Add assistant response
            self.add_message("assistant", response.content, 
                           metadata={"returncode": response.returncode})
            
            # Update metrics
            self.update_metrics(duration=response.execution_time, 
                              retries=getattr(response, 'retries', 0))
            
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
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        if metadata:
            message["metadata"] = metadata
        self.messages.append(message)
    
    def update_metrics(self, duration: float = 0, retries: int = 0):
        """Update session metrics."""
        self.total_duration += duration
        self.total_retries += retries
    
    def get_context(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation context."""
        if max_messages is None:
            return self.messages.copy()
        return self.messages[-max_messages:] if max_messages > 0 else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        from datetime import datetime
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "total_duration": self.total_duration,
            "total_retries": self.total_retries,
            "metadata": self.metadata
        }


# Session-aware convenience functions
def continue_claude(**kwargs) -> ClaudeCodeResponse:
    """Continue the most recent Claude conversation."""
    wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(**kwargs))
    return wrapper.continue_conversation()


def resume_claude(session_id: str, query: str = "", **kwargs) -> ClaudeCodeResponse:
    """Resume a specific Claude session."""
    wrapper = ClaudeCodeWrapper(ClaudeCodeConfig(**kwargs))
    return wrapper.resume_specific_session(session_id, query)


def ask_claude_with_session(query: str, session_id: Optional[str] = None, 
                           continue_last: bool = False, **kwargs) -> ClaudeCodeResponse:
    """Ask Claude with automatic session management."""
    config = ClaudeCodeConfig(**kwargs)
    
    if session_id:
        config.session_id = session_id
    elif continue_last:
        config.continue_session = True
        
    wrapper = ClaudeCodeWrapper(config)
    return wrapper.run(query)


# Original convenience functions with error handling
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
