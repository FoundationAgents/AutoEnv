"""
Pipeline Nodes

Flow:
1. AnalyzerNode: Analyze instruction/benchmark, generate analysis.json
2. StrategistNode: Create strategy based on analysis, generate strategy.json
3. AssetGeneratorNode: Generate image assets based on strategy
4. AssemblyNode: Assemble assets into runnable game
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field

from autoenv.pipeline.visual.prompt import (
    BENCHMARK_ANALYSIS_PROMPT,
    DEFAULT_GAME_CODE,
    GAME_ASSEMBLY_BENCHMARK_PROMPT,
    GAME_ASSEMBLY_INSTRUCTION_PROMPT,
    INSTRUCTION_ANALYSIS_PROMPT,
    STRATEGY_PROMPT,
    STYLE_CONSISTENT_PROMPT,
)
from base.agent.base_agent import BaseAgent
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_node import BaseNode, NodeContext
from base.utils.image import save_base64_image


class AutoEnvContext(NodeContext):
    """AutoEnv Pipeline context. Defines input/output fields for all nodes."""

    # Initial input
    benchmark_path: Path | None = None
    instruction: str | None = None
    output_dir: Path = Field(default_factory=lambda: Path("."))

    # AnalyzerNode output
    analysis: dict[str, Any] | None = None
    analysis_file: Path | None = None

    # StrategistNode output
    strategy: dict[str, Any] | None = None
    strategy_file: Path | None = None

    # AssetGeneratorNode output
    generated_assets: dict[str, str] = Field(default_factory=dict)
    style_anchor_image: str | None = None

    # AssemblyNode output
    game_dir: Path | None = None
    game_file: Path | None = None
    success: bool = False
    error: str | None = None


class AgentNode(BaseNode):
    """AgentNode: Compose BaseNode with Agent. Subclasses implement execute."""

    agent: BaseAgent | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}


class AnalyzerNode(AgentNode):
    """Analyze benchmark environment or user instruction, generate analysis.json."""

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
    """Create visualization strategy based on analysis, generate strategy.json."""

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
    """Generate game assets based on strategy."""

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

        # Sort by priority, style_anchor first
        sorted_assets = sorted(assets, key=lambda x: -x.get("priority", 0))

        # 1. Generate style_anchor (text-to-image) - must complete first, other assets depend on it
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

        # 2. Generate other assets in parallel (image-to-image, using style_anchor as reference)
        other_assets = [a for a in sorted_assets if a.get("id") != style_anchor_id]
        if other_assets:
            print(f"[AssetGenerator] Generating {len(other_assets)} assets in parallel...")
            tasks = [self._generate_asset(asset, ctx, assets_dir) for asset in other_assets]
            await asyncio.gather(*tasks)

        print(f"[AssetGenerator] Done. Total: {len(ctx.generated_assets)} assets")

    async def _generate_asset(
        self, asset: dict[str, Any], ctx: AutoEnvContext, assets_dir: Path
    ) -> None:
        """Generate a single asset and save immediately."""
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
        """Get asset generation prompt."""
        prompt = asset.get("prompt_strategy", {}).get("base_prompt", "")
        if not prompt:
            prompt = asset.get("description", asset.get("name", "game asset"))
        return prompt


class BackgroundRemovalNode(BaseNode):
    """Remove image background and crop to subject."""

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.generated_assets:
            ctx.error = "BackgroundRemovalNode requires generated_assets"
            return

        assets_dir = ctx.output_dir / "assets"
        if not assets_dir.exists():
            ctx.error = "Assets directory not found"
            return

        print(f"[BackgroundRemoval] Processing {len(ctx.generated_assets)} assets...")

        tasks = []
        for asset_id in ctx.generated_assets:
            image_path = assets_dir / f"{asset_id}.png"
            if image_path.exists():
                tasks.append(self._process_image(image_path, asset_id))

        await asyncio.gather(*tasks)
        print("[BackgroundRemoval] Done.")

    async def _process_image(self, image_path: Path, asset_id: str) -> None:
        """Remove background and crop to subject."""
        from PIL import Image
        from rembg import remove

        def _process() -> None:
            img = Image.open(image_path)
            output = remove(img)
            bbox = output.getbbox()
            if bbox:
                output = output.crop(bbox)
            output.save(image_path)

        await asyncio.to_thread(_process)
        print(f"[BackgroundRemoval] ✓ Processed: {asset_id}.png")


class AssemblyNode(AgentNode):
    """Assemble assets into a runnable pygame game."""

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

        # Generate game code
        if self.agent:
            game_code_prompt = self._build_game_code_prompt(ctx)
            await self.agent.run(request=game_code_prompt)

        game_file = game_dir / "game.py"
        if game_file.exists():
            ctx.game_dir = game_dir
            ctx.game_file = game_file
            ctx.success = True
        else:
            # Generate default game code
            self._generate_default_game(ctx, game_dir)
            ctx.game_dir = game_dir
            ctx.game_file = game_dir / "game.py"
            ctx.success = True

    def _build_game_code_prompt(self, ctx: AutoEnvContext) -> str:
        """Build game code generation prompt based on source type"""
        from PIL import Image

        strategy_json = json.dumps(ctx.strategy, indent=2, ensure_ascii=False)
        game_dir = ctx.output_dir / "game"
        assets_dir = game_dir / "assets"

        # Get actual dimensions for each asset
        asset_dimensions = []
        for asset_id in ctx.generated_assets:
            img_path = assets_dir / f"{asset_id}.png"
            if img_path.exists():
                with Image.open(img_path) as img:
                    w, h = img.size
                asset_dimensions.append(f"- {asset_id}.png: {w}x{h} pixels")
            else:
                asset_dimensions.append(f"- {asset_id}.png: (file not found)")

        # Choose prompt based on source type
        if ctx.benchmark_path:
            return GAME_ASSEMBLY_BENCHMARK_PROMPT.format(
                strategy_json=strategy_json,
                asset_dimensions="\n".join(asset_dimensions),
                game_dir=game_dir,
                benchmark_path=ctx.benchmark_path,
            )
        else:
            return GAME_ASSEMBLY_INSTRUCTION_PROMPT.format(
                strategy_json=strategy_json,
                asset_dimensions="\n".join(asset_dimensions),
                game_dir=game_dir,
            )

    def _generate_default_game(self, ctx: AutoEnvContext, game_dir: Path) -> None:
        """Generate default pygame game code."""
        asset_list = list(ctx.generated_assets.keys())
        game_code = DEFAULT_GAME_CODE.format(asset_list=asset_list)
        (game_dir / "game.py").write_text(game_code, encoding="utf-8")
