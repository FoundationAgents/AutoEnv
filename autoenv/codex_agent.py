from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator, PrivateAttr

from base.agent.base_agent import BaseAgent


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


# Lazily-evaluated flag indicating whether the Codex CLI is available.
CODEX_CLI_AVAILABLE: Optional[bool] = None


def is_codex_cli_available(force_recheck: bool = False) -> bool:
    """Check if Codex CLI is available (cached after first call).
    
    Args:
        force_recheck: If True, bypass cache and recheck CLI availability
        
    Returns:
        True if Codex CLI is available, False otherwise
    """
    global CODEX_CLI_AVAILABLE
    if CODEX_CLI_AVAILABLE is None or force_recheck:
        CODEX_CLI_AVAILABLE = _check_codex_cli()
    return CODEX_CLI_AVAILABLE


class CodexAgent(BaseAgent):
    """Codex agent for code generation using a Codex CLI tool."""
    
    name: str = Field(default="codex", description="Agent name")
    description: str = Field(
        default="Codex agent for code generation and execution",
        description="Agent description"
    )
    
    # Codex specific settings
    # NOTE: Uses `max_turns` for conversation turns with Codex CLI (vs BaseAgent's generic `max_steps`).
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    cwd: Optional[Path] = Field(default=None, description="Working directory")
    model: Optional[str] = Field(
        default=None,
        description="Model identifier for the Codex CLI. If not specified, uses the CLI's default model."
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
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for CLI commands"
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional API key for Codex CLI. If set, passed via OPENAI_API_KEY environment variable. "
            "If not set, CLI uses its own login state (codex login)."
        )
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    # Private attributes for internal state
    _messages: List[Any] = PrivateAttr(default_factory=list)
    _session_id: Optional[str] = PrivateAttr(default=None)
    _total_cost_usd: float = PrivateAttr(default=0.0)
    _current_prompt: Optional[str] = PrivateAttr(default=None)
    
    @model_validator(mode="after")
    def validate_codex_cli_available(self) -> "CodexAgent":
        """Validate that Codex CLI is available."""
        # Force recheck to handle case where CLI was installed after module import
        if not is_codex_cli_available(force_recheck=True):
            raise ImportError(
                "Codex CLI is not installed or not available on PATH. "
                "Please install the 'codex' command-line tool required for this project "
                "and ensure it is accessible in your PATH, then try again. "
                "Refer to your project documentation for the correct installation instructions."
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

    def _build_cli_command(self, prompt: str) -> List[str]:
        """Build Codex CLI command."""
        cmd = ["codex", "exec"]

        if self.model:
            cmd.extend(["-m", self.model])
        if self.cwd:
            cmd.extend(["-C", str(self.cwd)])

        sandbox_map = {
            "default": "read-only",
            "acceptEdits": "workspace-write",
            "bypassPermissions": "danger-full-access",
            "plan": "read-only"
        }
        cmd.extend(["-s", sandbox_map.get(self.permission_mode, "read-only")])

        if self.permission_mode == "bypassPermissions":
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.permission_mode == "acceptEdits":
            cmd.append("--full-auto")

        cmd.append("--json")
        # Use "--" so prompt content (even starting with "--") is treated as positional, not CLI flags
        cmd.append("--")
        cmd.append(prompt)

        return cmd

    async def _run_cli_command(self, prompt: str) -> Dict[str, Any]:
        """Run Codex CLI command and parse JSONL output."""
        # Validate cwd before use (handles case where cwd was modified via kwargs)
        self._validate_cwd()
        
        cmd = self._build_cli_command(prompt)

        env = os.environ.copy()
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd),
            env=env
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

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            raise RuntimeError(
                f"Codex CLI command failed with code {process.returncode}: {error_msg}"
            )

        output = stdout.decode('utf-8').strip()

        if not output:
            return {"result": "No output received", "session_id": None}

        try:
            lines = output.split('\n')
            result = {"result": ""}
            thread_id = None

            for line in lines:
                if not line.strip():
                    continue

                try:
                    obj = json.loads(line)

                    if obj.get("type") == "thread.started":
                        thread_id = obj.get("thread_id")
                    elif obj.get("type") == "item.completed":
                        item = obj.get("item", {})
                        if item.get("type") == "agent_message":
                            text = item.get("text", "")
                            if text:
                                if result["result"]:
                                    result["result"] += "\n\n"
                                result["result"] += text
                    elif obj.get("type") == "turn.completed":
                        usage = obj.get("usage", {})
                        if usage:
                            result["usage"] = usage

                except json.JSONDecodeError:
                    # Skip invalid JSON lines (e.g., empty/partial) and continue parsing the stream
                    pass

            if thread_id:
                result["session_id"] = thread_id

            if result["result"]:
                result["result"] = result["result"].strip()
            else:
                result["result"] = "No result received"

            return result

        except Exception as e:
            raise ValueError(
                f"Failed to parse Codex CLI output as JSON: {e}. Raw output: {output!r}"
            ) from e

    def _process_result(self, result: Dict[str, Any]) -> str:
        """Process CLI command result and update internal state.
        
        Args:
            result: Result dictionary from _run_cli_command
            
        Returns:
            Formatted result string
        """
        # Update session info
        if 'session_id' in result:
            self._session_id = result['session_id']
        if 'total_cost_usd' in result:
            self._total_cost_usd = result['total_cost_usd']

        # Append to message history
        self._messages.append(result)

        # Parse result based on type and subtype
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
        
        return result.get('result', 'No result received')

    async def step(self) -> str:
        """Execute a single step in the agent's workflow."""
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            result = await self._run_cli_command(self._current_prompt)
            return self._process_result(result)
        except (TimeoutError, RuntimeError, ValueError, FileNotFoundError, NotADirectoryError) as e:
            # Known CLI execution errors - return as error strings
            return f"Error during execution: {str(e)}"
        except Exception:
            # Re-raise unexpected errors (programming errors) for proper debugging
            raise

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """Execute the agent's main loop asynchronously.
        
        Args:
            request: The task or prompt to execute
            **kwargs: Temporary attribute overrides (max_turns, timeout, cwd, model, etc.)
                     
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
            'max_turns', 'timeout', 'cwd', 'model', 'permission_mode'
        }

        # Safely modify attributes with validation
        original_values = {}
        for key, value in kwargs.items():
            if key not in modifiable_attrs:
                return f"Error: Attribute '{key}' cannot be modified via kwargs. Allowed: {sorted(modifiable_attrs)}"
            
            if not hasattr(self, key):
                return f"Error: Unknown attribute '{key}'"
                
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
            result = await self._run_cli_command(request)
            return self._process_result(result)

        except (TimeoutError, RuntimeError, ValueError, FileNotFoundError, NotADirectoryError) as e:
            # Known CLI execution errors - return as error strings
            return f"Error during execution: {str(e)}"
        except Exception:
            # Re-raise unexpected errors (programming errors) for proper debugging
            raise

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
            "model": self.model,
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

