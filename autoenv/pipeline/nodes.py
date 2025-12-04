"""
Pipeline Nodes
基于 BaseNode 的 AgentNode 抽象，可组合任意 Agent 实现

流程：
1. AnalyzerNode: 分析 instruction/benchmark，生成 analysis.json
2. StrategistNode: 根据分析结果制定策略，生成 strategy.json
3. AssetGeneratorNode: 根据策略生成图像素材
4. AssemblyNode: 组装素材生成可运行的游戏
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field

from autoenv.pipeline.prompt import (
    BENCHMARK_ANALYSIS_PROMPT,
    DEFAULT_GAME_CODE,
    GAME_ASSEMBLY_PROMPT,
    INSTRUCTION_ANALYSIS_PROMPT,
    STRATEGY_PROMPT,
    STYLE_CONSISTENT_PROMPT,
)
from base.agent.base_agent import BaseAgent
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_node import BaseNode, NodeContext
from base.utils.image import save_base64_image


class AutoEnvContext(NodeContext):
    """AutoEnv Pipeline 的上下文，定义所有节点的输入输出字段"""

    # 初始输入
    benchmark_path: Path | None = None
    instruction: str | None = None
    output_dir: Path = Field(default_factory=lambda: Path("."))

    # AnalyzerNode 输出
    analysis: dict[str, Any] | None = None
    analysis_file: Path | None = None

    # StrategistNode 输出
    strategy: dict[str, Any] | None = None
    strategy_file: Path | None = None

    # AssetGeneratorNode 输出
    generated_assets: dict[str, str] = Field(default_factory=dict)
    style_anchor_image: str | None = None

    # AssemblyNode 输出
    game_dir: Path | None = None
    game_file: Path | None = None
    success: bool = False
    error: str | None = None


class AgentNode(BaseNode):
    """AgentNode: 组合 BaseNode 和 Agent，子类实现 execute"""

    agent: BaseAgent | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}


class AnalyzerNode(AgentNode):
    """分析 Benchmark 环境或用户指令，生成 analysis.json"""

    async def execute(self, ctx: AutoEnvContext) -> None:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = ctx.output_dir / "analysis.json"

        if ctx.instruction:
            task = INSTRUCTION_ANALYSIS_PROMPT.format(
                instruction=ctx.instruction,
                cwd=Path.cwd(),
                output_file=output_file,
                output_filename=output_file.name,
            )
        elif ctx.benchmark_path:
            task = BENCHMARK_ANALYSIS_PROMPT.format(
                benchmark_path=ctx.benchmark_path,
                cwd=Path.cwd(),
                output_file=output_file,
                output_filename=output_file.name,
            )
        else:
            ctx.error = "AnalyzerNode requires instruction or benchmark_path"
            return

        if not self.agent:
            raise ValueError("AnalyzerNode requires an agent")

        await self.agent.run(request=task)

        if output_file.exists():
            with open(output_file, encoding="utf-8") as f:
                ctx.analysis = json.load(f)
                ctx.analysis_file = output_file
        else:
            ctx.error = "analysis output_file not found"


class StrategistNode(AgentNode):
    """基于分析结果制定可视化策略，生成 strategy.json"""

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.analysis_file:
            ctx.error = "StrategistNode requires analysis_file from AnalyzerNode"
            return

        output_file = ctx.output_dir / "strategy.json"
        task = STRATEGY_PROMPT.format(
            analysis_file=ctx.analysis_file,
            output_file=output_file,
            output_filename=output_file.name,
            cwd=Path.cwd(),
        )

        if not self.agent:
            raise ValueError("StrategistNode requires an agent")

        await self.agent.run(request=task)

        if output_file.exists():
            with open(output_file, encoding="utf-8") as f:
                ctx.strategy = json.load(f)
                ctx.strategy_file = output_file
        else:
            ctx.error = "strategy output_file not found"


class AssetGeneratorNode(AgentNode):
    """根据策略生成游戏素材"""

    image_llm: AsyncLLM | None = Field(default=None)

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.strategy:
            ctx.error = "AssetGeneratorNode requires strategy from StrategistNode"
            return

        if not self.image_llm:
            ctx.error = "AssetGeneratorNode requires image_llm"
            return

        assets = ctx.strategy.get("assets", [])
        if not assets:
            ctx.error = "No assets defined in strategy"
            return

        assets_dir = ctx.output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        print(f"[AssetGenerator] Starting generation for {len(assets)} assets → {assets_dir}")

        # 按 priority 排序，style_anchor 优先
        sorted_assets = sorted(assets, key=lambda x: -x.get("priority", 0))

        # 1. 生成 style_anchor（text-to-image）- 必须先完成，其他素材依赖它
        style_anchor_id = ctx.strategy.get("style_anchor")
        for asset in sorted_assets:
            if asset.get("id") == style_anchor_id:
                print(f"[AssetGenerator] Generating style anchor: {style_anchor_id}")
                prompt = self._get_asset_prompt(asset)
                result = await self.image_llm.generate_text_to_image(prompt)
                if result["success"]:
                    ctx.generated_assets[asset["id"]] = result["image_base64"]
                    ctx.style_anchor_image = result["image_base64"]
                    save_base64_image(result["image_base64"], assets_dir / f"{style_anchor_id}.png")
                    print(f"[AssetGenerator] ✓ Style anchor saved: {style_anchor_id}.png")
                else:
                    print(f"[AssetGenerator] ✗ Style anchor failed: {result.get('error')}")
                break

        # 2. 并行生成其他素材（image-to-image，使用 style_anchor 作为参考）
        other_assets = [a for a in sorted_assets if a.get("id") != style_anchor_id]
        if other_assets:
            print(f"[AssetGenerator] Generating {len(other_assets)} assets in parallel...")
            tasks = [self._generate_asset(asset, ctx, assets_dir) for asset in other_assets]
            await asyncio.gather(*tasks)

        print(f"[AssetGenerator] Done. Total: {len(ctx.generated_assets)} assets")

    async def _generate_asset(
        self, asset: dict[str, Any], ctx: AutoEnvContext, assets_dir: Path
    ) -> None:
        """生成单个素材并立即保存"""
        asset_id = asset.get("id", "unknown")
        print(f"[AssetGenerator] → Generating: {asset_id}")

        prompt = STYLE_CONSISTENT_PROMPT.format(base_prompt=self._get_asset_prompt(asset))
        if ctx.style_anchor_image:
            result = await self.image_llm.generate_image_to_image(
                prompt, [ctx.style_anchor_image]
            )
        else:
            result = await self.image_llm.generate_text_to_image(prompt)

        if result["success"]:
            ctx.generated_assets[asset_id] = result["image_base64"]
            save_base64_image(result["image_base64"], assets_dir / f"{asset_id}.png")
            print(f"[AssetGenerator] ✓ Saved: {asset_id}.png")
        else:
            print(f"[AssetGenerator] ✗ Failed: {asset_id} - {result.get('error')}")

    def _get_asset_prompt(self, asset: dict[str, Any]) -> str:
        """获取素材生成 prompt"""
        prompt = asset.get("prompt_strategy", {}).get("base_prompt", "")
        if not prompt:
            prompt = asset.get("description", asset.get("name", "game asset"))
        return prompt


class AssemblyNode(AgentNode):
    """组装素材生成可运行的 pygame 游戏"""

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.strategy:
            ctx.error = "AssemblyNode requires strategy"
            return

        if not ctx.generated_assets:
            ctx.error = "AssemblyNode requires generated_assets"
            return

        game_dir = ctx.output_dir / "game"
        game_dir.mkdir(parents=True, exist_ok=True)

        # 复制素材到 game/assets
        assets_src = ctx.output_dir / "assets"
        assets_dst = game_dir / "assets"
        if assets_src.exists():
            if assets_dst.exists():
                shutil.rmtree(assets_dst)
            shutil.copytree(assets_src, assets_dst)

        # 生成游戏代码
        if self.agent:
            game_code_prompt = self._build_game_code_prompt(ctx)
            await self.agent.run(request=game_code_prompt)

        game_file = game_dir / "game.py"
        if game_file.exists():
            ctx.game_dir = game_dir
            ctx.game_file = game_file
            ctx.success = True
        else:
            # 生成默认游戏代码
            self._generate_default_game(ctx, game_dir)
            ctx.game_dir = game_dir
            ctx.game_file = game_dir / "game.py"
            ctx.success = True

    def _build_game_code_prompt(self, ctx: AutoEnvContext) -> str:
        """构建游戏代码生成 prompt"""
        strategy_json = json.dumps(ctx.strategy, indent=2, ensure_ascii=False)
        asset_list = list(ctx.generated_assets.keys())
        game_dir = ctx.output_dir / "game"
        return GAME_ASSEMBLY_PROMPT.format(
            strategy_json=strategy_json,
            asset_list=asset_list,
            game_dir=game_dir,
        )

    def _generate_default_game(self, ctx: AutoEnvContext, game_dir: Path) -> None:
        """生成默认的 pygame 游戏代码"""
        asset_list = list(ctx.generated_assets.keys())
        game_code = DEFAULT_GAME_CODE.format(asset_list=asset_list)
        (game_dir / "game.py").write_text(game_code, encoding="utf-8")
