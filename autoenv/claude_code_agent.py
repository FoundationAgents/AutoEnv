from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator, PrivateAttr

from base.agent.base_agent import BaseAgent


try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        ClaudeSDKError,
        CLINotFoundError,
        ProcessError,
        CLIJSONDecodeError,
    )
    CLAUDE_AGENT_AVAILABLE = True
except ImportError:
    CLAUDE_AGENT_AVAILABLE = False
    # Fallback exception definitions that maintain proper inheritance
    # These match the SDK's exception hierarchy for consistent error handling
    class ClaudeSDKError(Exception):
        """Base exception for Claude SDK errors."""
        pass
    
    class CLINotFoundError(ClaudeSDKError):
        """Exception raised when Claude CLI is not found."""
        pass
    
    class ProcessError(ClaudeSDKError):
        """Exception raised when process execution fails."""
        def __init__(self, message="", exit_code=-1):
            super().__init__(message)
            self.exit_code = exit_code
    
    class CLIJSONDecodeError(ClaudeSDKError):
        """Exception raised when JSON decoding fails."""
        pass
    
    # Stub for query and ClaudeAgentOptions when SDK is unavailable
    query = None
    ClaudeAgentOptions = None


class ClaudeCodeAgent(BaseAgent):
    """Claude Code Agent for code generation using Claude Agent SDK."""
    
    name: str = Field(default="claude_code", description="Agent name")
    description: str = Field(
        default="Claude Code agent for code generation and execution",
        description="Agent description"
    )
    
    # Claude Code specific settings
    # NOTE: Uses `max_turns` for conversation turns with Claude Code CLI (vs BaseAgent's generic `max_steps`)
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    max_messages: int = Field(
        default=1000,
        description="Maximum number of messages to keep in history. Older messages are discarded."
    )
    cwd: Optional[Path] = Field(default=None, description="Working directory")
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="Allowed tools (e.g., ['Read', 'Write', 'Bash'])"
    )
    permission_mode: str = Field(
        default="acceptEdits",
        description=(
            "Permission mode controlling agent's action execution behavior.\n"
            "- 'default': Interactive mode, requests confirmation before actions\n"
            "- 'acceptEdits': Auto-accepts code edits, still confirms dangerous operations\n"
            "- 'bypassPermissions': ⚠️ DANGEROUS - Bypasses all permission checks and executes "
            "actions without confirmation. Only use in fully trusted, isolated environments.\n"
            "- 'plan': Planning mode, generates action plans without executing"
        )
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Override system prompt (only for non-interactive mode)"
    )
    append_system_prompt: Optional[str] = Field(
        default=None,
        description="Append to system prompt (only for non-interactive mode)"
    )
    api_base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (e.g., for proxies or API gateways)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If not provided, uses ANTHROPIC_API_KEY env var"
    )
    env_vars: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional environment variables to pass to Claude CLI"
    )
    
    class Config:
        arbitrary_types_allowed = True

    # Private attributes for internal state
    _messages: List[Any] = PrivateAttr(default_factory=list)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _total_cost_usd: float = PrivateAttr(default=0.0)
    _current_prompt: Optional[str] = PrivateAttr(default=None)
    _max_messages: int = PrivateAttr(default=1000)  # Limit message history to prevent unbounded growth

    @model_validator(mode="after")
    def validate_claude_agent_available(self) -> "ClaudeCodeAgent":
        """Validate that Claude Agent SDK is available."""
        if not CLAUDE_AGENT_AVAILABLE:
            raise ImportError(
                "Claude Agent SDK is not installed. "
                "Install it with: pip install claude-agent-sdk\n"
                "Note: The Claude Code CLI is automatically bundled - no separate installation needed!"
            )

        # Set default cwd if not provided
        if self.cwd is None:
            self.cwd = Path.cwd()
        else:
            self.cwd = Path(self.cwd)

        # Validate working directory exists
        self._validate_cwd()

        # Validate permission mode
        valid_modes = ["default", "acceptEdits", "bypassPermissions", "plan"]
        if self.permission_mode not in valid_modes:
            raise ValueError(
                f"Invalid permission_mode: {self.permission_mode}. "
                f"Must be one of: {valid_modes}"
            )

        return self

    def _validate_cwd(self) -> None:
        """Validate working directory exists and is a directory."""
        if not self.cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {self.cwd}")
        if not self.cwd.is_dir():
            raise NotADirectoryError(f"Working directory path is not a directory: {self.cwd}")

    def _create_options(self) -> ClaudeAgentOptions:
        """Create ClaudeAgentOptions from agent settings."""
        options_dict = {
            "max_turns": self.max_turns,
            "cwd": str(self.cwd),
            "permission_mode": self.permission_mode,
        }

        if self.allowed_tools:
            options_dict["allowed_tools"] = self.allowed_tools
        if self.system_prompt_override:
            options_dict["system_prompt"] = self.system_prompt_override
        elif self.append_system_prompt:
            options_dict["append_system_prompt"] = self.append_system_prompt
        
        # Build environment variables for Claude CLI
        env_dict = {}
        if self.env_vars:
            env_dict.update(self.env_vars)
        
        # Set API key if provided
        if self.api_key:
            env_dict["ANTHROPIC_API_KEY"] = self.api_key
        
        # Set base URL if provided (Claude CLI respects ANTHROPIC_BASE_URL)
        if self.api_base_url:
            env_dict["ANTHROPIC_BASE_URL"] = self.api_base_url
        
        if env_dict:
            options_dict["env"] = env_dict

        return ClaudeAgentOptions(**options_dict)

    def _handle_sdk_error(self, e: Exception) -> str:
        """Handle Claude SDK errors with consistent formatting.
        
        Only converts expected SDK errors to user-friendly strings.
        Re-raises unexpected exceptions (programming errors) for proper debugging.
        
        Args:
            e: The exception to handle
            
        Returns:
            Formatted error message string for SDK errors
            
        Raises:
            Exception: Re-raises non-SDK exceptions for proper error tracking
        """
        if isinstance(e, CLINotFoundError):
            return f"Error: Claude Code CLI not found: {str(e)}"
        elif isinstance(e, ProcessError):
            exit_code = getattr(e, 'exit_code', 'unknown')
            return f"Error: Process failed with exit code {exit_code}: {str(e)}"
        elif isinstance(e, CLIJSONDecodeError):
            return f"Error: Failed to parse Claude response: {str(e)}"
        elif isinstance(e, ClaudeSDKError):
            return f"Error: Claude SDK error: {str(e)}"
        else:
            # Re-raise unexpected exceptions (programming errors, etc.) for proper debugging
            raise

    async def _process_query_stream(self, prompt: str, options: ClaudeAgentOptions) -> str:
        """Process the query stream and extract result.
        
        Handles message iteration, session tracking, cost accumulation, and result extraction.
        This is shared logic between step() and run() methods.
        
        Args:
            prompt: The prompt to send to Claude
            options: Claude agent options
            
        Returns:
            Result text from the query
        """
        result_text = ""
        async for message in query(prompt=prompt, options=options):
            # Enforce max_messages limit to prevent unbounded memory growth
            if len(self._messages) >= self._max_messages:
                # Remove oldest message to maintain sliding window
                self._messages.pop(0)
            self._messages.append(message)

            if hasattr(message, 'session_id'):
                self._session_id = message.session_id

            # Skip non-result messages
            if not (hasattr(message, 'type') and message.type == "result"):
                continue

            # Track costs for result messages
            if hasattr(message, 'total_cost_usd'):
                self._total_cost_usd += message.total_cost_usd

            # Capture result if available
            if hasattr(message, 'result'):
                result_text = message.result
            # Handle error or completion subtypes
            elif hasattr(message, 'subtype'):
                if message.subtype == "error_max_turns":
                    result_text = f"Error: Reached maximum turns ({self.max_turns})"
                elif message.subtype == "error_during_execution":
                    result_text = "Error: Execution failed"
                else:
                    result_text = f"Completed with status: {message.subtype}"
            else:
                result_text = "Execution completed"

        return result_text if result_text else "No result received"

    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            options = self._create_options()
            return await self._process_query_stream(self._current_prompt, options)
        except Exception as e:
            return self._handle_sdk_error(e)

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """Execute the agent's main loop asynchronously.
        
        Args:
            request: The task or prompt to execute
            **kwargs: Temporary attribute overrides (max_turns, cwd, permission_mode, etc.)
                     
        Returns:
            Result string from execution or error message
            
        Warning:
            This agent is NOT safe for concurrent use. Do not call run() from multiple
            coroutines simultaneously on the same agent instance, as attribute modifications
            will interfere with each other.
            
        Note:
            Attributes modified via kwargs are restored after execution on a "best effort" basis.
            In rare cases, restoration may fail to avoid masking the primary execution error.
        """
        if not request:
            return "Error: No request provided"

        self._current_prompt = request
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0

        # Whitelist of attributes that can be modified via kwargs
        modifiable_attrs = {
            'max_turns', 'cwd', 'permission_mode', 'allowed_tools',
            'system_prompt_override', 'append_system_prompt',
            'api_base_url', 'api_key', 'env_vars'
        }

        # Safely modify attributes with validation
        original_values = {}
        for key, value in kwargs.items():
            if not hasattr(self, key):
                return f"Error: Unknown attribute '{key}'"
            
            if key not in modifiable_attrs:
                return f"Error: Attribute '{key}' cannot be modified via kwargs. Allowed: {sorted(modifiable_attrs)}"
                
            try:
                original_values[key] = getattr(self, key)
                setattr(self, key, value)
                
                # Validate critical attributes after modification
                if key == 'cwd':
                    # Security: Validate and sanitize cwd to prevent directory traversal attacks
                    if isinstance(value, str):
                        new_cwd = Path(value)
                    elif isinstance(value, Path):
                        new_cwd = value
                    else:
                        raise ValueError("cwd must be a string or pathlib.Path")
                    
                    # Security: Prevent absolute paths to sensitive directories
                    if new_cwd.is_absolute():
                        raise ValueError(
                            "For security reasons, cwd cannot be set to an absolute path via kwargs. "
                            "Set cwd during agent initialization instead."
                        )
                    
                    # Security: Prevent directory traversal via '..' 
                    if ".." in new_cwd.parts:
                        raise ValueError("cwd cannot contain parent directory references ('..')")
                    
                    # Resolve relative to current cwd and validate
                    self.cwd = (self.cwd / new_cwd).resolve()
                    self._validate_cwd()
                elif key == 'permission_mode':
                    valid_modes = ["default", "acceptEdits", "bypassPermissions", "plan"]
                    if value not in valid_modes:
                        raise ValueError(f"Invalid permission_mode: {value}. Must be one of: {valid_modes}")
                        
            except Exception as e:
                # If validation fails, restore any attributes set so far
                for restore_key, restore_value in original_values.items():
                    try:
                        setattr(self, restore_key, restore_value)
                    except Exception:
                        pass  # Best effort restoration
                return f"Error: Failed to set attribute '{key}': {str(e)}"

        try:
            options = self._create_options()
            return await self._process_query_stream(request, options)
        except Exception as e:
            return self._handle_sdk_error(e)
        finally:
            # Restore original values (best effort)
            restoration_failures = []
            for key, value in original_values.items():
                try:
                    setattr(self, key, value)
                except Exception as e:
                    # Track restoration failures for potential debugging
                    restoration_failures.append(f"{key}: {e}")
            
            # Note: We don't raise restoration errors to avoid masking the primary execution result.
            # In production, consider logging restoration_failures for debugging.

    async def __call__(self, **kwargs) -> str:
        """Execute the agent with given parameters."""
        request = kwargs.pop('request', None) or kwargs.pop('task', None) or kwargs.pop('prompt', None)
        return await self.run(request=request, **kwargs)

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "num_messages": len(self._messages),
            "cwd": str(self.cwd),
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

    def get_messages(self) -> List[Any]:
        """Get all messages from the current session."""
        return self._messages.copy()

    def reset(self) -> None:
        """Reset the agent state for a new session."""
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0
        self._current_prompt = None

