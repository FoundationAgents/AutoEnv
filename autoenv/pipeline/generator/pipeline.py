"""
Generator Pipeline

DAG:
  EnvDesc → EnvYaml → EnvCode → EnvValidator → CodeFix → LevelGen → MaxReward → Archive
"""

from pathlib import Path

from autoenv.pipeline.generator.nodes import (
    ArchiveNode,
    CodeFixNode,
    EnvCodeNode,
    EnvDescNode,
    EnvValidatorNode,
    EnvYamlNode,
    GeneratorContext,
    LevelGenNode,
    MaxRewardNode,
)
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_pipeline import BasePipeline


class GeneratorPipeline(BasePipeline):
    """Environment generation pipeline."""

    @classmethod
    def create_default(
        cls,
        llm_name: str = "gemini-3-pro-preview",
        reasoning_llm_name: str | None = None,
    ) -> "GeneratorPipeline":
        """
        Create default Generator Pipeline.

        Args:
            llm_name: Main LLM model name
            reasoning_llm_name: Reasoning LLM model (for desc/yaml), defaults to llm_name
        """
        llm = AsyncLLM(llm_name)
        reasoning_llm = AsyncLLM(reasoning_llm_name) if reasoning_llm_name else llm

        # Create nodes
        env_desc = EnvDescNode(node_id="env_desc", llm=reasoning_llm)
        env_yaml = EnvYamlNode(node_id="env_yaml", llm=reasoning_llm)
        env_code = EnvCodeNode(node_id="env_code", llm=llm)
        env_validator = EnvValidatorNode(node_id="env_validator", llm=llm)
        code_fix = CodeFixNode(node_id="code_fix", llm=llm)
        level_gen = LevelGenNode(node_id="level_gen", llm=llm)
        max_reward = MaxRewardNode(node_id="max_reward", llm=llm)
        archive = ArchiveNode(node_id="archive")

        # Build DAG
        (
            env_desc
            >> env_yaml
            >> env_code
            >> env_validator
            >> code_fix
            >> level_gen
            >> max_reward
            >> archive
        )

        return cls(root=env_desc)

    async def run(
        self,
        requirements: str,
        output_dir: Path | str | None = None,
        env_theme: str = "random",
    ) -> GeneratorContext:
        """
        Run generation pipeline.

        Args:
            requirements: Environment requirements (string or .txt file path)
            output_dir: Output root directory, defaults to workspace/envs
            env_theme: Environment theme name

        Returns:
            GeneratorContext: Context containing generation results
        """
        # Handle file path input
        if isinstance(requirements, str) and requirements.endswith(".txt"):
            req_path = Path(requirements)
            if req_path.exists():
                env_theme = req_path.stem
                requirements = req_path.read_text(encoding="utf-8")

        ctx = GeneratorContext(
            requirements=requirements,
            env_theme=env_theme,
        )
        if output_dir:
            ctx.envs_root_path = Path(output_dir)

        return await super().run(ctx)

