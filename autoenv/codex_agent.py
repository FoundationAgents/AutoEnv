"""
Codex Agent for AutoEnv

This module provides a CodexAgent class that integrates OpenAI Codex CLI
into the AutoEnv project. It inherits from BaseAgent and provides a unified
interface for code generation and execution tasks using Codex.

Installation:
    npm install -g @openai/codex
    # Or using Homebrew:
    brew install --cask codex

Prerequisites:
    - Python 3.10+
    - Node.js
    - OpenAI Codex CLI
    - OPENAI_API_KEY environment variable

Example:
    from autoenv.codex_agent import CodexAgent
    from base.engine.async_llm import AsyncLLM

    agent = CodexAgent(
        llm=AsyncLLM("gpt-4"),
        max_turns=5,
        cwd="/path/to/project"
    )

    result = await agent.run(request="Write a function to calculate Fibonacci numbers")
    print(result)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator, PrivateAttr

from base.agent.base_agent import BaseAgent
from base.engine.async_llm import AsyncLLM


# Check if Codex CLI is available
def _check_codex_cli() -> bool:
    """Check if Codex CLI (codex command) is installed."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


CODEX_CLI_AVAILABLE = _check_codex_cli()


class CodexAgent(BaseAgent):
    """
    Codex Agent that integrates OpenAI Codex CLI for code generation tasks.

    This agent wraps the OpenAI Codex CLI to provide a unified interface
    compatible with AutoEnv's BaseAgent architecture. It supports both single-step
    execution and multi-turn conversations.

    Codex CLI is a coding agent from OpenAI that runs locally on your computer.
    It can read, write, and execute code in your project directory.
    Learn more: https://github.com/openai/codex

    Attributes:
        name: Agent name (default: "codex")
        description: Agent description
        llm: AsyncLLM instance (optional, for compatibility)
        max_turns: Maximum number of conversation turns (default: 10)
        cwd: Current working directory for code execution
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        permission_mode: Permission mode ("default", "acceptEdits", "bypassPermissions", "plan")
        allowed_tools: List of allowed tools (e.g., ["Read", "Write", "Bash"])
        system_prompt_override: Optional system prompt override
        append_system_prompt: Optional text to append to system prompt

        # Internal state
        _messages: List of messages from the current session
        _session_id: Current session ID
        _total_cost_usd: Total cost in USD
        _current_prompt: Current prompt being processed
        _process: Subprocess handle for CLI interaction
    """
    
    name: str = Field(default="codex", description="Agent name")
    description: str = Field(
        default="Codex agent for code generation and execution",
        description="Agent description"
    )
    
    # Codex specific settings
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    cwd: Optional[Path] = Field(default=None, description="Working directory")
    model: str = Field(
        default="gpt-4",
        description="Model name (gpt-4, gpt-3.5-turbo, etc.)"
    )
    permission_mode: str = Field(
        default="acceptEdits",
        description="Permission mode: default|acceptEdits|bypassPermissions|plan"
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="Allowed tools (e.g., ['Read', 'Write', 'Bash'])"
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Override system prompt (only for non-interactive mode)"
    )
    append_system_prompt: Optional[str] = Field(
        default=None,
        description="Append to system prompt (only for non-interactive mode)"
    )
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for CLI commands"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    # Private attributes for internal state
    _messages: List[Any] = PrivateAttr(default_factory=list)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _total_cost_usd: float = PrivateAttr(default=0.0)
    _current_prompt: Optional[str] = PrivateAttr(default=None)
    _process: Optional[subprocess.Popen] = PrivateAttr(default=None)
    
    @model_validator(mode="after")
    def validate_codex_cli_available(self) -> "CodexAgent":
        """Validate that Codex CLI is available."""
        if not CODEX_CLI_AVAILABLE:
            raise ImportError(
                "Codex CLI is not installed. "
                "Install it with: npm install -g @openai/codex\n"
                "Or using Homebrew: brew install --cask codex\n"
                "Learn more: https://github.com/openai/codex"
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

    def _build_cli_command(self, prompt: str) -> List[str]:
        """Build the Codex CLI command with all options.

        Uses 'codex exec' for non-interactive execution.
        See: https://github.com/openai/codex for available options.

        Note: The permission_mode parameter is mapped to sandbox and approval settings:
        - "default": read-only sandbox (safe, no auto-approval)
        - "acceptEdits": workspace-write sandbox (auto-approve on-request)
        - "bypassPermissions": danger-full-access sandbox (skip all approvals)
        - "plan": read-only sandbox (planning mode)
        """
        # Use 'codex exec' for non-interactive execution
        cmd = ["codex", "exec"]

        # Add model if specified
        if self.model:
            cmd.extend(["-m", self.model])

        # Add working directory
        if self.cwd:
            cmd.extend(["-C", str(self.cwd)])

        # Map permission_mode to sandbox mode
        sandbox_map = {
            "default": "read-only",
            "acceptEdits": "workspace-write",
            "bypassPermissions": "danger-full-access",
            "plan": "read-only"
        }
        sandbox_mode = sandbox_map.get(self.permission_mode, "read-only")
        cmd.extend(["-s", sandbox_mode])

        # Handle special permission modes
        if self.permission_mode == "bypassPermissions":
            # Use the dangerous flag to skip all approvals
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.permission_mode == "acceptEdits":
            # Use full-auto for automatic execution with workspace write access
            cmd.append("--full-auto")

        # Add JSON output format for structured parsing
        cmd.append("--json")

        # Add the prompt as the final argument
        cmd.append(prompt)

        return cmd

    async def _run_cli_command(self, prompt: str) -> Dict[str, Any]:
        """
        Run Codex CLI command and parse JSON output.

        Args:
            prompt: The prompt to send to Codex

        Returns:
            dict: Parsed JSON response from Codex CLI

        Raises:
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If command fails
            json.JSONDecodeError: If output is not valid JSON
        """
        cmd = self._build_cli_command(prompt)

        # Run command asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd)
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Codex CLI command timed out after {self.timeout} seconds"
            )

        # Check return code
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            raise RuntimeError(
                f"Codex CLI command failed with code {process.returncode}: {error_msg}"
            )

        # Parse JSONL output (each line is a JSON object)
        output = stdout.decode('utf-8').strip()

        if not output:
            # Empty output, return empty result
            return {"result": "", "session_id": None}

        try:
            # Parse as JSONL (multiple JSON objects, one per line)
            lines = output.split('\n')
            result = {"result": ""}
            thread_id = None

            # Process each line as a JSON object
            for line in lines:
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)

                    # Extract thread_id
                    if obj.get("type") == "thread.started":
                        thread_id = obj.get("thread_id")

                    # Extract agent messages
                    elif obj.get("type") == "item.completed":
                        item = obj.get("item", {})
                        if item.get("type") == "agent_message":
                            # Append agent message to result
                            text = item.get("text", "")
                            if text:
                                if result["result"]:
                                    result["result"] += "\n\n"
                                result["result"] += text

                    # Extract usage information
                    elif obj.get("type") == "turn.completed":
                        usage = obj.get("usage", {})
                        if usage:
                            result["usage"] = usage

                except json.JSONDecodeError:
                    # If a line is not valid JSON, skip it
                    pass

            # Add session info
            if thread_id:
                result["session_id"] = thread_id

            # Clean up result text
            if result["result"]:
                result["result"] = result["result"].strip()
            else:
                result["result"] = "No result received"

            return result

        except Exception as e:
            raise json.JSONDecodeError(
                f"Failed to parse Codex CLI output as JSON: {str(e)}",
                output,
                0
            )

    async def step(self) -> str:
        """
        Execute a single step in the agent's workflow.

        This method processes one turn of the conversation with Codex CLI.
        It uses the current prompt stored in _current_prompt.

        Returns:
            str: The result text from the current step

        Note:
            This method is designed to work with the run() method, which sets
            up the _current_prompt. For standalone use, call run() instead.
        """
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            result = await self._run_cli_command(self._current_prompt)

            # Extract session_id if available
            if 'session_id' in result:
                self._session_id = result['session_id']

            # Extract cost if available
            if 'total_cost_usd' in result:
                self._total_cost_usd = result['total_cost_usd']

            # Store result for message history
            self._messages.append(result)

            # Check result type
            if result.get('type') == 'result':
                subtype = result.get('subtype', 'success')

                if subtype == 'success':
                    return result.get('result', 'Execution completed')
                elif subtype == 'error_max_turns':
                    return f"Error: Reached maximum turns ({self.max_turns})"
                elif subtype == 'error_during_execution':
                    return "Error: Execution failed"
                else:
                    return f"Completed with status: {subtype}"

            # Fallback: return result field if available
            return result.get('result', 'No result received')

        except Exception as e:
            return f"Error during execution: {str(e)}"

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """
        Execute the agent's main loop asynchronously.

        This method runs a complete task using Codex CLI. It handles
        the full conversation lifecycle, including multiple turns if needed.

        Args:
            request: The task/prompt to execute
            **kwargs: Additional keyword arguments:
                - cwd: Override working directory
                - max_turns: Override max turns
                - model: Override model name
                - allowed_tools: Override allowed tools
                - permission_mode: Override permission mode
                - system_prompt: Override system prompt
                - append_system_prompt: Append to system prompt
                - timeout: Override timeout

        Returns:
            str: The final result from Codex CLI execution

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
            # Run the CLI command
            result = await self._run_cli_command(request)

            # Extract session info
            if 'session_id' in result:
                self._session_id = result['session_id']

            if 'total_cost_usd' in result:
                self._total_cost_usd = result['total_cost_usd']

            # Store result
            self._messages.append(result)

            # Extract result text
            if result.get('type') == 'result':
                subtype = result.get('subtype', 'success')

                if subtype == 'success':
                    result_text = result.get('result', 'Execution completed')
                elif subtype == 'error_max_turns':
                    result_text = f"Error: Reached maximum turns ({self.max_turns})"
                elif subtype == 'error_during_execution':
                    result_text = "Error: Execution failed during run"
                else:
                    result_text = f"Completed with status: {subtype}"
            else:
                result_text = result.get('result', 'No result received')

            return result_text

        except Exception as e:
            return f"Error during execution: {str(e)}"

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
                - model: Model name
                - max_turns: Maximum turns
                - permission_mode: Permission mode
        """
        return {
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "num_messages": len(self._messages),
            "cwd": str(self.cwd),
            "model": self.model,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

    def get_messages(self) -> List[Any]:
        """
        Get all messages from the current session.

        Returns:
            list: List of message dictionaries from Codex CLI
        """
        return self._messages.copy()

    def reset(self) -> None:
        """Reset the agent state for a new session."""
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0
        self._current_prompt = None
        self.current_step = 0
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass

