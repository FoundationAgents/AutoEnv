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
    """Codex Agent for code generation using OpenAI Codex CLI."""
    
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
        cmd.append(prompt)

        return cmd

    async def _run_cli_command(self, prompt: str) -> Dict[str, Any]:
        """Run Codex CLI command and parse JSONL output."""
        cmd = self._build_cli_command(prompt)

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

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            raise RuntimeError(
                f"Codex CLI command failed with code {process.returncode}: {error_msg}"
            )

        output = stdout.decode('utf-8').strip()

        if not output:
            return {"result": "", "session_id": None}

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
                    pass

            if thread_id:
                result["session_id"] = thread_id

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
        """Execute a single step in the agent's workflow."""
        if not self._current_prompt:
            return "No prompt provided. Use run() method to execute tasks."

        try:
            result = await self._run_cli_command(self._current_prompt)

            if 'session_id' in result:
                self._session_id = result['session_id']
            if 'total_cost_usd' in result:
                self._total_cost_usd = result['total_cost_usd']

            self._messages.append(result)

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

        except Exception as e:
            return f"Error during execution: {str(e)}"

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        """Execute the agent's main loop asynchronously."""
        if not request:
            return "Error: No request provided"

        self._current_prompt = request
        self._messages = []
        self._session_id = None
        self._total_cost_usd = 0.0

        original_values = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                original_values[key] = getattr(self, key)
                setattr(self, key, value)

        try:
            result = await self._run_cli_command(request)

            if 'session_id' in result:
                self._session_id = result['session_id']
            if 'total_cost_usd' in result:
                self._total_cost_usd = result['total_cost_usd']

            self._messages.append(result)

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

