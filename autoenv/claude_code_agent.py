"""
Claude Code Agent for AutoEnv

This module provides a ClaudeCodeAgent class that integrates Claude Agent SDK
into the AutoEnv project. It inherits from BaseAgent and provides a unified
interface for code generation and execution tasks.

Installation:
    pip install claude-agent-sdk

    Note: The Claude Code CLI is automatically bundled with the SDK -
    no separate installation required!

Prerequisites:
    - Python 3.10+
    - ANTHROPIC_API_KEY environment variable

Example:
    from autoenv.claude_code_agent import ClaudeCodeAgent
    from base.engine.async_llm import AsyncLLM

    agent = ClaudeCodeAgent(
        llm=AsyncLLM("claude-sonnet-4-5"),
        max_turns=5,
        cwd="/path/to/project"
    )

    result = await agent.run(request="Write a function to calculate Fibonacci numbers")
    print(result)

References:
    - Claude Agent SDK: https://github.com/anthropics/claude-agent-sdk-python
    - Documentation: https://platform.claude.com/docs/en/agent-sdk/python
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator, PrivateAttr

from base.agent.base_agent import BaseAgent
from base.engine.async_llm import AsyncLLM

# Import Claude Agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions, ClaudeSDKClient
    from claude_agent_sdk import AssistantMessage, UserMessage, ResultMessage
    from claude_agent_sdk import TextBlock, ToolUseBlock, ToolResultBlock
    from claude_agent_sdk import ClaudeSDKError, CLINotFoundError, ProcessError, CLIJSONDecodeError
    CLAUDE_AGENT_AVAILABLE = True
except ImportError:
    CLAUDE_AGENT_AVAILABLE = False
    # Fallback types
    AssistantMessage = UserMessage = ResultMessage = Any
    TextBlock = ToolUseBlock = ToolResultBlock = Any
    ClaudeSDKError = CLINotFoundError = ProcessError = CLIJSONDecodeError = Exception


class ClaudeCodeAgent(BaseAgent):
    """
    Claude Code Agent that integrates Claude Code SDK for code generation tasks.
    
    This agent wraps the Claude Code SDK to provide a unified interface compatible
    with AutoEnv's BaseAgent architecture. It supports both single-step execution
    and multi-turn conversations.
    
    Attributes:
        name: Agent name (default: "claude_code")
        description: Agent description
        llm: AsyncLLM instance (optional, for compatibility)
        max_turns: Maximum number of conversation turns (default: 10)
        cwd: Current working directory for code execution
        allowed_tools: List of allowed tools (e.g., ["Read", "Write", "Bash"])
        permission_mode: Permission mode ("default", "acceptEdits", "bypassPermissions", "plan")
        system_prompt_override: Optional system prompt override
        append_system_prompt: Optional text to append to system prompt
        
        # Internal state
        _messages: List of messages from the current session
        _session_id: Current session ID
        _total_cost_usd: Total cost in USD
        _current_prompt: Current prompt being processed
    """
    
    name: str = Field(default="claude_code", description="Agent name")
    description: str = Field(
        default="Claude Code agent for code generation and execution",
        description="Agent description"
    )
    
    # Claude Code specific settings
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
        options_dict = {
            "max_turns": self.max_turns,
            "cwd": str(self.cwd),  # Ensure cwd is a string
            "permission_mode": self.permission_mode,
        }

        if self.allowed_tools:
            options_dict["allowed_tools"] = self.allowed_tools

        # Official SDK uses system_prompt instead of system_prompt_override
        if self.system_prompt_override:
            options_dict["system_prompt"] = self.system_prompt_override
        elif self.append_system_prompt:
            options_dict["append_system_prompt"] = self.append_system_prompt

        return ClaudeAgentOptions(**options_dict)

    async def step(self) -> str:
        """
        Execute a single step in the agent's workflow.

        This method processes one turn of the conversation with Claude Code.
        It uses the current prompt stored in _current_prompt and collects
        messages until the conversation completes or reaches max_turns.

        Returns:
            str: The result text from the current step

        Note:
            This method is designed to work with the run() method, which sets
            up the _current_prompt. For standalone use, call run() instead.
        """
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            options = self._create_options()

            # Execute one turn using Claude Agent SDK
            async for message in query(
                prompt=self._current_prompt,
                options=options
            ):
                self._messages.append(message)

                # Extract session_id if available
                if hasattr(message, 'session_id'):
                    self._session_id = message.session_id

                # Check if this is a result message (final message)
                if hasattr(message, 'type') and message.type == "result":
                    if hasattr(message, 'total_cost_usd'):
                        self._total_cost_usd += message.total_cost_usd

                    # Return the result text
                    if hasattr(message, 'result'):
                        return message.result
                    elif hasattr(message, 'subtype'):
                        # Handle error cases
                        if message.subtype == "error_max_turns":
                            return f"Error: Reached maximum turns ({self.max_turns})"
                        elif message.subtype == "error_during_execution":
                            return "Error: Execution failed"

                    return "Execution completed"

            return "No result received"

        except CLINotFoundError:
            return "Error: Claude Code CLI not found. Please install: pip install claude-agent-sdk"
        except ProcessError as e:
            return f"Error: Process failed with exit code {getattr(e, 'exit_code', 'unknown')}"
        except CLIJSONDecodeError as e:
            return f"Error: Failed to parse Claude response: {str(e)}"
        except ClaudeSDKError as e:
            return f"Error: Claude SDK error: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error: {str(e)}"

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """
        Execute the agent's main loop asynchronously.

        This method runs a complete task using Claude Code SDK. It handles
        the full conversation lifecycle, including multiple turns if needed.

        Args:
            request: The task/prompt to execute
            **kwargs: Additional keyword arguments:
                - cwd: Override working directory
                - max_turns: Override max turns
                - allowed_tools: Override allowed tools
                - permission_mode: Override permission mode
                - system_prompt: Override system prompt
                - append_system_prompt: Append to system prompt

        Returns:
            str: The final result from Claude Code execution

        Example:
            result = await agent.run(
                request="Write a Python function to read CSV files",
                cwd="/path/to/project",
                max_turns=5
            )
        """
        if not request:
            return "Error: No request provided"

        # Store the prompt for step() method
        self._current_prompt = request

        # Reset state for new run
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0

        # Apply kwargs overrides
        original_values = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                original_values[key] = getattr(self, key)
                setattr(self, key, value)

        try:
            # Create options with current settings
            options = self._create_options()

            # Execute the full conversation
            result_text = ""
            async for message in query(prompt=request, options=options):
                self._messages.append(message)

                # Extract session_id
                if hasattr(message, 'session_id'):
                    self._session_id = message.session_id

                # Handle result message
                if hasattr(message, 'type') and message.type == "result":
                    if hasattr(message, 'total_cost_usd'):
                        self._total_cost_usd = message.total_cost_usd

                    if hasattr(message, 'result'):
                        result_text = message.result
                    elif hasattr(message, 'subtype'):
                        if message.subtype == "error_max_turns":
                            result_text = f"Error: Reached maximum turns ({self.max_turns})"
                        elif message.subtype == "error_during_execution":
                            result_text = "Error: Execution failed during run"
                        else:
                            result_text = f"Completed with status: {message.subtype}"

            return result_text if result_text else "No result received"

        except CLINotFoundError:
            return "Error: Claude Code CLI not found. Please install: pip install claude-agent-sdk"
        except ProcessError as e:
            return f"Error: Process failed with exit code {getattr(e, 'exit_code', 'unknown')}"
        except CLIJSONDecodeError as e:
            return f"Error: Failed to parse Claude response: {str(e)}"
        except ClaudeSDKError as e:
            return f"Error: Claude SDK error: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error: {str(e)}"
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(self, key, value)

    async def __call__(self, **kwargs) -> str:
        """
        Execute the agent with given parameters.

        This method provides a callable interface compatible with BaseAgent.

        Args:
            **kwargs: Keyword arguments passed to run()
                - request: The task/prompt (can also be 'task' or 'prompt')
                - Other kwargs are passed to run()

        Returns:
            str: The result from run()
        """
        # Extract request from various possible keys
        request = kwargs.pop('request', None) or kwargs.pop('task', None) or kwargs.pop('prompt', None)
        return await self.run(request=request, **kwargs)

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            dict: Session information including:
                - session_id: Current session ID
                - total_cost_usd: Total cost in USD
                - num_messages: Number of messages in current session
                - cwd: Current working directory
        """
        return {
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "num_messages": len(self._messages),
            "cwd": str(self.cwd),
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

    def get_messages(self) -> List[Any]:
        """
        Get all messages from the current session.

        Returns:
            list: List of Message objects from Claude Code SDK
        """
        return self._messages.copy()

    def reset(self) -> None:
        """Reset the agent state for a new session."""
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0
        self._current_prompt = None
        self.current_step = 0

