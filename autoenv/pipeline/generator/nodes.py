"""
Generator Pipeline Nodes

Flow:
1. EnvDescNode: Generate env_desc.txt
2. EnvYamlNode: Generate config.yaml
3. EnvCodeNode: Generate code files + agent_instruction + action_space
4. EnvValidatorNode: Generate validator
5. CodeFixNode: Fix code with ECodeAgent
6. LevelGenNode: Generate levels
7. MaxRewardNode: Calculate max rewards
8. ArchiveNode: Archive auxiliary files and create done.txt
"""

import os
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from autoenv.coder import ECodeAgent
from autoenv.pipeline.generator.prompt import (
    CRAFT_ENV_CODE_AND_INSTRUCTION_PROMPT,
    CRAFT_ENV_DESIGN_PROMPT,
    CRAFT_ENV_VALIDATOR_PROMPT,
    CRAFT_ENV_YAML_PROMPT,
    ECODE_AGENT_CALCULATE_MAX_REWARD_PROMPT,
    ECODE_AGENT_CODE_FIX_PROMPT,
    ECODE_AGENT_LEVEL_GENERATION_PROMPT,
    VALIDATOR_CHECKLIST,
)
from base.engine.async_llm import AsyncLLM
from base.engine.utils import parse_xml_content, read_file_content, write_file_content
from base.pipeline.base_node import BaseNode, NodeContext


class GeneratorContext(NodeContext):
    """Generator Pipeline context."""

    # Initial input
    requirements: str = ""
    env_theme: str = "random"
    envs_root_path: Path = Field(default_factory=lambda: Path("workspace/envs"))

    # Environment paths
    env_id: str | None = None
    env_folder_path: Path | None = None

    # EnvDescNode output
    env_desc: str | None = None

    # EnvYamlNode output
    env_yaml: str | None = None
    env_implement_help: str | None = None

    # EnvCodeNode output
    env_main_code: str | None = None
    env_obs_code: str | None = None
    env_generate_code: str | None = None
    env_main_use_code: str | None = None
    agent_instruction: str | None = None
    action_space: str | None = None

    # EnvValidatorNode output
    env_validator_code: str | None = None

    # CodeFixNode/LevelGenNode/MaxRewardNode output
    code_fix_result: Any = None
    level_gen_result: Any = None
    max_reward_result: Any = None

    # Status
    success: bool = False
    error: str | None = None


class EnvDescNode(BaseNode):
    """Generate environment description document env_desc.txt."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.requirements:
            ctx.error = "EnvDescNode requires requirements"
            return

        self._init_env_folder(ctx)

        prompt = CRAFT_ENV_DESIGN_PROMPT.format(requirements=ctx.requirements)
        resp = await self.llm(prompt)
        env_desc = parse_xml_content(resp, "env_design")["env_design"]

        write_file_content(str(ctx.env_folder_path / "env_desc.txt"), env_desc)
        ctx.env_desc = env_desc
        print(f"[EnvDescNode] ✓ env_desc.txt saved to {ctx.env_folder_path}")

    def _init_env_folder(self, ctx: GeneratorContext) -> None:
        """Initialize environment folder."""
        if not ctx.env_id:
            t = time.time()
            local_time = time.localtime(t)
            ctx.env_id = time.strftime("%Y%m%d_%H%M%S", local_time) + f"_env_{ctx.env_theme}"
        if not ctx.env_folder_path:
            ctx.env_folder_path = ctx.envs_root_path / ctx.env_id
        ctx.env_folder_path.mkdir(parents=True, exist_ok=True)


class EnvYamlNode(BaseNode):
    """Generate environment config config.yaml."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_desc:
            ctx.error = "EnvYamlNode requires env_desc from EnvDescNode"
            return

        prompt = CRAFT_ENV_YAML_PROMPT.format(
            env_desc=ctx.env_desc,
            config_yaml_example=read_file_content("base/env/base_env_config.yaml"),
            environment_abstraction=read_file_content("base/env/base_env.py"),
            observation_abstraction=read_file_content("base/env/base_observation.py"),
            generator_abstraction=read_file_content("base/env/base_generator.py"),
        )
        resp = await self.llm(prompt)

        ctx.env_yaml = parse_xml_content(resp, "env_config")["env_config"]
        ctx.env_implement_help = parse_xml_content(resp, "env_implement_help")["env_implement_help"]

        write_file_content(str(ctx.env_folder_path / "config.yaml"), ctx.env_yaml)
        write_file_content(str(ctx.env_folder_path / "env_implement.txt"), ctx.env_implement_help)
        print(f"[EnvYamlNode] ✓ config.yaml saved")


class EnvCodeNode(BaseNode):
    """Generate environment code + agent_instruction + action_space."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_desc or not ctx.env_yaml:
            ctx.error = "EnvCodeNode requires env_desc and env_yaml"
            return

        prompt = CRAFT_ENV_CODE_AND_INSTRUCTION_PROMPT.format(
            env_desc=ctx.env_desc,
            config_yaml=ctx.env_yaml,
            env_implement_help=ctx.env_implement_help or "",
            environment_abstraction=read_file_content("base/env/base_env.py"),
            observation_abstraction=read_file_content("base/env/base_observation.py"),
            generator_abstraction=read_file_content("base/env/base_generator.py"),
            env_folder_path=ctx.env_folder_path,
        )
        resp = await self.llm(prompt, max_tokens=32768)

        ctx.env_main_code = parse_xml_content(resp, "env_main_code")["env_main_code"]
        ctx.env_obs_code = parse_xml_content(resp, "env_obs_code")["env_obs_code"]
        ctx.env_generate_code = parse_xml_content(resp, "env_generate_code")["env_generate_code"]
        ctx.env_main_use_code = parse_xml_content(resp, "env_main_code_use")["env_main_code_use"]
        ctx.agent_instruction = parse_xml_content(resp, "agent_instruction")["agent_instruction"]
        ctx.action_space = parse_xml_content(resp, "action_space")["action_space"]

        folder = ctx.env_folder_path
        write_file_content(str(folder / "env_main.py"), ctx.env_main_code)
        write_file_content(str(folder / "env_obs.py"), ctx.env_obs_code)
        write_file_content(str(folder / "env_generate.py"), ctx.env_generate_code)
        write_file_content(str(folder / "env_main_use.py"), ctx.env_main_use_code)
        write_file_content(str(folder / "agent_instruction.txt"), ctx.agent_instruction)
        write_file_content(str(folder / "action_space.txt"), ctx.action_space)
        print(f"[EnvCodeNode] ✓ env code files saved")


class EnvValidatorNode(BaseNode):
    """Generate environment validator env_validator.py."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_desc or not ctx.env_main_code:
            ctx.error = "EnvValidatorNode requires env_desc and env_main_code"
            return

        prompt = CRAFT_ENV_VALIDATOR_PROMPT.format(
            validator_checklist=VALIDATOR_CHECKLIST,
            env_desc=ctx.env_desc,
            config_yaml=ctx.env_yaml or "",
            env_code=ctx.env_main_code,
            observation_code=ctx.env_obs_code or "",
            generator_code=ctx.env_generate_code or "",
        )
        resp = await self.llm(prompt, max_tokens=16384)

        ctx.env_validator_code = parse_xml_content(resp, "env_validator_code")["env_validator_code"]
        write_file_content(str(ctx.env_folder_path / "env_validator.py"), ctx.env_validator_code)
        print(f"[EnvValidatorNode] ✓ env_validator.py saved")


class CodeFixNode(BaseNode):
    """Fix environment code using ECodeAgent."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_folder_path:
            ctx.error = "CodeFixNode requires env_folder_path"
            return

        code_agent = ECodeAgent(llm=AsyncLLM(self.llm.config))

        task = ECODE_AGENT_CODE_FIX_PROMPT.format(
            env_id=ctx.env_id,
            workspace=ctx.env_folder_path,
            validator_checklist=VALIDATOR_CHECKLIST,
        )
        ctx.code_fix_result = await code_agent(requirements=task, cwds=str(ctx.env_folder_path))
        print(f"[CodeFixNode] ✓ code fix completed")


class LevelGenNode(BaseNode):
    """Generate levels using ECodeAgent."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_folder_path:
            ctx.error = "LevelGenNode requires env_folder_path"
            return

        code_agent = ECodeAgent(llm=AsyncLLM(self.llm.config))

        task = ECODE_AGENT_LEVEL_GENERATION_PROMPT.format(
            env_id=ctx.env_id,
            workspace=ctx.env_folder_path,
            validator_checklist=VALIDATOR_CHECKLIST,
        )
        ctx.level_gen_result = await code_agent(requirements=task, cwds=str(ctx.env_folder_path))
        print(f"[LevelGenNode] ✓ level generation completed")


class MaxRewardNode(BaseNode):
    """Calculate max rewards using ECodeAgent."""

    llm: AsyncLLM | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_folder_path:
            ctx.error = "MaxRewardNode requires env_folder_path"
            return

        code_agent = ECodeAgent(llm=AsyncLLM(self.llm.config))

        task = ECODE_AGENT_CALCULATE_MAX_REWARD_PROMPT.format(
            env_id=ctx.env_id,
            workspace=ctx.env_folder_path,
        )
        ctx.max_reward_result = await code_agent(requirements=task, cwds=str(ctx.env_folder_path))
        print(f"[MaxRewardNode] ✓ max reward calculation completed")


class ArchiveNode(BaseNode):
    """Archive auxiliary files and create done.txt marker."""

    async def execute(self, ctx: GeneratorContext) -> None:
        if not ctx.env_folder_path:
            ctx.error = "ArchiveNode requires env_folder_path"
            return

        from scripts.run_archive_files import archive_auxiliary_files

        # Archive auxiliary files (move non-core files to archive/)
        archive_auxiliary_files(str(ctx.env_folder_path), dry_run=False)

        # Create done.txt marker
        done_file = ctx.env_folder_path / "done.txt"
        done_file.write_text("", encoding="utf-8")

        ctx.success = True
        print(f"[ArchiveNode] ✓ Archived and done.txt created → {ctx.env_folder_path}")

