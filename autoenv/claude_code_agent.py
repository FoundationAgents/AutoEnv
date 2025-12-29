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
    # Fallback types
    AssistantMessage = UserMessage = ResultMessage = Any
    TextBlock = ToolUseBlock = ToolResultBlock = Any
    ClaudeSDKError = CLINotFoundError = ProcessError = CLIJSONDecodeError = Exception


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
    cwd: Optional[Path] = Field(default=None, description="Working directory")
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="Allowed tools (e.g., ['Read', 'Write', 'Bash'])"
    )
    permission_mode: str = Field(
        default="acceptEdits",
        description="Permission mode: default|acceptEdits|bypassPermissions|plan"
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Override system prompt (only for non-interactive mode)"
    )
    append_system_prompt: Optional[str] = Field(
        default=None,
        description="Append to system prompt (only for non-interactive mode)"
    )
    
    class Config:
        arbitrary_types_allowed = True

    # Private attributes for internal state
    _messages: List[Any] = PrivateAttr(default_factory=list)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _total_cost_usd: float = PrivateAttr(default=0.0)
    _current_prompt: Optional[str] = PrivateAttr(default=None)

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

        # Validate permission mode
        valid_modes = ["default", "acceptEdits", "bypassPermissions", "plan"]
        if self.permission_mode not in valid_modes:
            raise ValueError(
                f"Invalid permission_mode: {self.permission_mode}. "
                f"Must be one of: {valid_modes}"
            )

        return self

    def _create_options(self) -> ClaudeAgentOptions:
        """Create ClaudeAgentOptions from agent settings."""
        # Validate working directory exists
        if not self.cwd.exists():
            raise FileNotFoundError(f"Working directory does not exist: {self.cwd}")
        if not self.cwd.is_dir():
            raise NotADirectoryError(f"Working directory path is not a directory: {self.cwd}")
        
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

        return ClaudeAgentOptions(**options_dict)

    def _handle_sdk_error(self, e: Exception) -> str:
        """Handle Claude SDK errors with consistent formatting.
        
        Args:
            e: The exception to handle
            
        Returns:
            Formatted error message string
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
            return f"Error: Unexpected error: {str(e)}"

    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            options = self._create_options()

            async for message in query(
                prompt=self._current_prompt,
                options=options
            ):
                self._messages.append(message)

                if hasattr(message, 'session_id'):
                    self._session_id = message.session_id

                # Skip non-result messages
                if not (hasattr(message, 'type') and message.type == "result"):
                    continue

                # Track costs for result messages
                if hasattr(message, 'total_cost_usd'):
                    self._total_cost_usd += message.total_cost_usd

                # Return result if available
                if hasattr(message, 'result'):
                    return message.result

                # Handle error or completion subtypes
                if hasattr(message, 'subtype'):
                    if message.subtype == "error_max_turns":
                        return f"Error: Reached maximum turns ({self.max_turns})"
                    elif message.subtype == "error_during_execution":
                        return "Error: Execution failed"

                return "Execution completed"

            return "No result received"

        except Exception as e:
            return self._handle_sdk_error(e)

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """Execute the agent's main loop asynchronously."""
        if not request:
            return "Error: No request provided"

        self._current_prompt = request
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0

        # Safely modify attributes, only tracking successful changes
        original_values = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                try:
                    original_values[key] = getattr(self, key)
                    setattr(self, key, value)
                except Exception as e:
                    # If setting fails, restore any attributes set so far
                    for restore_key, restore_value in original_values.items():
                        try:
                            setattr(self, restore_key, restore_value)
                        except Exception:
                            pass  # Best effort restoration
                    return f"Error: Failed to set attribute '{key}': {str(e)}"

        try:
            options = self._create_options()

            result_text = ""
            async for message in query(prompt=request, options=options):
                self._messages.append(message)

                if hasattr(message, 'session_id'):
                    self._session_id = message.session_id

                # Skip non-result messages
                if not (hasattr(message, 'type') and message.type == "result"):
                    continue

                # Track costs for result messages
                if hasattr(message, 'total_cost_usd'):
                    self._total_cost_usd = message.total_cost_usd

                # Return result if available
                if hasattr(message, 'result'):
                    result_text = message.result
                # Handle error or completion subtypes
                elif hasattr(message, 'subtype'):
                    if message.subtype == "error_max_turns":
                        result_text = f"Error: Reached maximum turns ({self.max_turns})"
                    elif message.subtype == "error_during_execution":
                        result_text = "Error: Execution failed during run"
                    else:
                        result_text = f"Completed with status: {message.subtype}"
                else:
                    result_text = "Execution completed"

            return result_text if result_text else "No result received"

        except Exception as e:
            return self._handle_sdk_error(e)
        finally:
            # Restore original values (best effort)
            for key, value in original_values.items():
                try:
                    setattr(self, key, value)
                except Exception:
                    pass  # Ignore errors during restoration

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

