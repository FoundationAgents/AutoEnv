from typing import Optional
import os

from base.agent.base_agent import BaseAgent
from autoenv.miniswe_agent import MiniSWEAutoEnvAgent

class ECodeAgent(BaseAgent):
    """Agent that asks mini-swe-agent to generate levels inside Docker.

    - Mounts the target env folder at /workspace.
    - Mounts repo root at /repo and sets PYTHONPATH for imports.
    - Uses a persistent pip cache to speed up runs.
    - Instructs the assistant to fix scripts when needed, generate 100 levels to ./levels, validate, then finish.
    """

    name: str = "coder"
    desc: str = "A minimal coder for AutoEnv-generated environments"

    async def __call__(self, requirements: Optional[str] = None, cwds: Optional[str] = None, environment_type: Optional[str] = "local") -> str:
        # Resolve paths
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

        return await agent.run(task=requirements)

    # BaseAgent abstract methods
    async def step(self) -> str:
        return "noop"

    async def run(self, request: Optional[str] = None, **kwargs) -> str:
        return await self.__call__(requirements=request, cwds=kwargs.get("cwds"))
