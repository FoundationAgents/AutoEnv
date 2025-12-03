"""
Modifier for exist environment, main focus on skin modification.
"""

from base.agent.base_agent import BaseAgent

class Modifier(BaseAgent):
    """
    Define attribute here.
    """

    async def copy_workspace(env_path):
        pass

    async def analysis_environment(workspace_path):
        pass

    async def propose_modification(analysis_result, workspace_path):
        pass

    async def apply_modification(modification_result, workspace_path):
        pass

    async def run(env_path, requirement, mode):
        pass


