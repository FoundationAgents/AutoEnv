from typing import Optional, Literal
import os

from base.agent.base_agent import BaseAgent
from autoenv.miniswe_agent import MiniSWEAutoEnvAgent


class ECodeAgent(BaseAgent):
    """Agent for code generation tasks in AutoEnv.
    
    Supports multiple backends via Strategy Pattern:
        - miniswe (default): Uses MiniSWEAutoEnvAgent, works with any LLM
        - codex: Uses OpenAI Codex CLI (requires OPENAI_API_KEY)
        - claude: Uses Claude Code CLI (requires ANTHROPIC_API_KEY)
    
    All backends implement the same BaseAgent.run(request=...) interface.
    """

    name: str = "coder"
    desc: str = "A minimal coder for AutoEnv-generated environments"
    
    # Backend selection: "miniswe" (default), "codex", or "claude"
    backend: Literal["miniswe", "codex", "claude"] = "miniswe"

    def _create_agent(self, cwds: str, environment_type: str = "local") -> BaseAgent:
        """Factory method to create the appropriate agent based on backend."""
        if self.backend == "codex":
            from autoenv.codex_agent import CodexAgent
            agent = CodexAgent(cwd=cwds)
        
        elif self.backend == "claude":
            from autoenv.claude_code_agent import ClaudeCodeAgent
            agent = ClaudeCodeAgent(cwd=cwds)
        
        else:  # miniswe (default)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            base_env = {"PYTHONPATH": os.pathsep.join([repo_root, os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)}
            
            if environment_type == "docker":
                agent = MiniSWEAutoEnvAgent(
                    llm=self.llm,  # Pass the LLM instance from BaseAgent
                    mode="yolo",
                    step_limit=50,
                    environment_type="docker",
                    cwd = cwds,
                    env = base_env,
                    timeout = 900,
                    docker_image="python:3.11-slim",
                )
            elif environment_type == "local":
                agent = MiniSWEAutoEnvAgent(
                    llm=self.llm,  # Pass the LLM instance from BaseAgent
                    mode="yolo",
                    step_limit=100,
                    environment_type="local",
                    cwd = cwds,
                    env = base_env,
                    timeout = 900,
                )
            else:
                raise ValueError(f"Unsupported environment_type: {environment_type}")
        
        return agent

    async def __call__(self, requirements: Optional[str] = None, cwds: Optional[str] = None, environment_type: Optional[str] = "local") -> str:
        """Execute code task with configured backend."""
        agent = self._create_agent(cwds, environment_type)
        return await agent.run(request=requirements)

    # BaseAgent abstract methods
    async def step(self) -> str:
        return "noop"

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        return await self.__call__(requirements=request, cwds=kwargs.get("cwds"))
