"""
Pipeline Nodes

Flow:
1. AnalyzerNode: Analyze instruction/benchmark, generate analysis.json
2. StrategistNode: Create strategy based on analysis, generate strategy.json
3. AssetGeneratorNode: Generate image assets based on strategy
4. BackgroundRemovalNode: Remove backgrounds from assets
5. Image3DConvertNode: Convert 2D assets to 3D models (optional)
6. AssemblyNode: Assemble assets into runnable game
"""

import asyncio
import json
import shutil
import time
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

    # Image23DNode output (3D models)
    models_3d: dict[str, Any] = Field(default_factory=dict)
    small_assets_dir: Path | None = None

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

        # Copy generated assets into game/assets
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
            # Validate that the generated code references existing assets; if not, fallback
            if not self._validate_generated_game(ctx, game_dir, game_file):
                self._generate_default_game(ctx, game_dir)
            ctx.game_dir = game_dir
            ctx.game_file = game_dir / "game.py"
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

    def _validate_generated_game(self, ctx: AutoEnvContext, game_dir: Path, game_file: Path) -> bool:
        """Check if generated game.py references assets that exist. Return True if valid, False to trigger fallback."""
        try:
            code = game_file.read_text(encoding="utf-8")
        except Exception:
            return False

        # Collect referenced image filenames from the code (simple heuristic)
        import re
        referenced = set(re.findall(r"([A-Za-z0-9_\-]+\.png)", code))

        # Build set of available asset filenames
        assets_dir = game_dir / "assets"
        available = set()
        if assets_dir.exists():
            for p in assets_dir.glob("*.png"):
                available.add(p.name)

        # If code references any filename not available, consider invalid
        missing = [fn for fn in referenced if fn not in available]
        if missing:
            print(f"[Assembly] Detected missing asset files referenced in game.py: {missing}. Falling back to default game code.")
            return False
        return True


class Image3DConvertNode(BaseNode):
    """Convert 2D image assets to 3D models using Meshy API (parallel execution)."""

    meshy_api_key: str = Field(default="")
    meshy_base_url: str = Field(default="https://api.meshy.ai/v1")
    target_polycount: int = Field(default=10000)
    timeout: float = Field(default=600)
    max_assets_to_convert: int = Field(default=4)

    model_config = {"arbitrary_types_allowed": True}

    async def execute(self, ctx: AutoEnvContext) -> None:
        """Execute 3D conversion for selected assets in parallel."""
        from autoenv.pipeline.visual.meshy_client import MeshyClient

        if not self.meshy_api_key:
            ctx.error = "Image3DConvertNode requires meshy_api_key"
            return

        assets_dir = ctx.output_dir / "assets"
        if not assets_dir.exists():
            ctx.error = "Assets directory not found"
            return

        models_dir = ctx.output_dir / "models_3d"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize timing logs
        timing_log = {
            "start_time": time.time(),
            "assets": {},
        }

        print(f"[Image3DConvert] Starting 3D conversion → {models_dir}")
        print(f"[Image3DConvert] API: {self.meshy_base_url}")
        print(f"[Image3DConvert] Target polycount: {self.target_polycount}")

        client = MeshyClient(
            api_key=self.meshy_api_key,
            base_url=self.meshy_base_url,
        )

        # Select assets to convert
        assets_to_convert = self._select_assets_for_3d(ctx)
        print(f"[Image3DConvert] Converting {len(assets_to_convert)} assets in PARALLEL: {assets_to_convert}")

        # Store 3D model info in context
        if not hasattr(ctx, "models_3d"):
            ctx.models_3d = {}

        # Filter valid assets
        valid_assets = []
        for asset_id in assets_to_convert:
            image_path = assets_dir / f"{asset_id}.png"
            if not image_path.exists():
                print(f"[Image3DConvert] ⚠ Skip {asset_id}: image not found")
            else:
                valid_assets.append((asset_id, image_path))

        if not valid_assets:
            print("[Image3DConvert] No valid assets to convert")
            return

        # Create parallel conversion tasks
        async def convert_single_asset(asset_id: str, image_path: Path) -> dict:
            """Convert a single asset to 3D."""
            asset_timing = {"start": time.time(), "asset_id": asset_id}
            print(f"[Image3DConvert] → Starting: {asset_id}")

            def progress_callback(task_id, status, progress):
                elapsed = time.time() - asset_timing["start"]
                print(f"[Image3DConvert]   {asset_id}: {status} {progress}% ({elapsed:.1f}s)")

            try:
                result = await client.image_to_3d(
                    image_path=image_path,
                    timeout=self.timeout,
                    progress_callback=progress_callback,
                    target_polycount=self.target_polycount,
                )

                asset_timing["end"] = time.time()
                asset_timing["duration"] = asset_timing["end"] - asset_timing["start"]

                if result["success"]:
                    model_urls = result.get("model_urls", {})
                    glb_url = model_urls.get("glb")

                    if glb_url:
                        output_path = models_dir / f"{asset_id}.glb"
                        download_start = time.time()

                        download_result = await client.download_model(
                            model_url=glb_url,
                            output_path=output_path,
                        )

                        download_time = time.time() - download_start
                        asset_timing["download_time"] = download_time

                        if download_result["success"]:
                            asset_timing["success"] = True
                            asset_timing["output_path"] = str(output_path)
                            asset_timing["model_urls"] = model_urls
                            print(
                                f"[Image3DConvert] ✓ {asset_id}.glb saved "
                                f"(convert: {asset_timing['duration']:.1f}s, "
                                f"download: {download_time:.1f}s)"
                            )
                        else:
                            asset_timing["success"] = False
                            asset_timing["error"] = f"Download failed: {download_result.get('error')}"
                            print(f"[Image3DConvert] ✗ {asset_id}: Download failed: {download_result.get('error')}")
                    else:
                        asset_timing["success"] = False
                        asset_timing["error"] = "No GLB URL in result"
                        print(f"[Image3DConvert] ✗ {asset_id}: No GLB URL in result")
                else:
                    asset_timing["success"] = False
                    asset_timing["error"] = result.get("error", "Unknown error")
                    print(f"[Image3DConvert] ✗ {asset_id}: Conversion failed: {result.get('error')}")

            except Exception as e:
                asset_timing["success"] = False
                asset_timing["error"] = str(e)
                asset_timing["end"] = time.time()
                asset_timing["duration"] = asset_timing["end"] - asset_timing["start"]
                print(f"[Image3DConvert] ✗ {asset_id}: Error: {e}")

            return asset_timing

        # Execute all conversions in parallel
        print(f"[Image3DConvert] Launching {len(valid_assets)} parallel conversion tasks...")
        tasks = [convert_single_asset(asset_id, image_path) for asset_id, image_path in valid_assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"[Image3DConvert] ✗ Task exception: {result}")
                continue

            asset_id = result.get("asset_id")
            timing_log["assets"][asset_id] = {
                "start": result.get("start"),
                "end": result.get("end"),
                "duration": result.get("duration"),
                "download_time": result.get("download_time"),
            }

            if result.get("success"):
                success_count += 1
                ctx.models_3d[asset_id] = {
                    "path": result.get("output_path"),
                    "model_urls": result.get("model_urls"),
                    "conversion_time": result.get("duration"),
                    "download_time": result.get("download_time"),
                }

        # Final timing summary
        timing_log["end_time"] = time.time()
        timing_log["total_duration"] = timing_log["end_time"] - timing_log["start_time"]
        timing_log["parallel_count"] = len(valid_assets)
        timing_log["success_count"] = success_count

        print(f"[Image3DConvert] Done. Total time: {timing_log['total_duration']:.1f}s")
        print(f"[Image3DConvert] Converted: {success_count}/{len(valid_assets)} models in parallel")

        # Save timing log
        self._save_timing_log(ctx.output_dir, timing_log)

    def _select_assets_for_3d(self, ctx: AutoEnvContext) -> list[str]:
        """Select which assets to convert to 3D."""
        strategy = ctx.strategy or {}
        all_assets = list(ctx.generated_assets.keys()) if ctx.generated_assets else []
        
        if not all_assets:
            return []
        
        # Define priority categories for game elements
        priority_keywords = {
            "critical": ["player", "character", "avatar", "hero"],
            "high": ["block", "box", "crate", "cube", "piece", "tile", "platform", "goal", "target","ground", "floor", "wall"],
            "medium": ["obstacle", "enemy", "item", "collectible"]
        }
        
        # Keywords to filter out (not suitable for 3D conversion)
        exclude_keywords = ["ui", "hud", "button", "text", "overlay", "background", 
                           "menu", "icon", "cursor", "arrow", "indicator", "display",
                           "counter", "timer", "score", "health", "bar"]
        
        # Score each asset
        scored_assets = []
        for asset_id in all_assets:
            asset_name = asset_id.lower()
            
            # Filter out UI/overlay elements
            if any(keyword in asset_name for keyword in exclude_keywords):
                print(f"[Image3DConvert] Skipping UI element: {asset_id}")
                continue
            
            # Calculate priority score
            score = 0
            if any(keyword in asset_name for keyword in priority_keywords["critical"]):
                score = 4
            elif any(keyword in asset_name for keyword in priority_keywords["high"]):
                score = 3
            elif any(keyword in asset_name for keyword in priority_keywords["medium"]):
                score = 2
            else:
                score = 2  # Default medium priority
            
            # Boost score if it's the style anchor
            style_anchor = strategy.get("style_anchor")
            if style_anchor and asset_id == style_anchor:
                score += 1
            
            scored_assets.append((score, asset_id))
        
        # Sort by score (descending) and select top assets
        scored_assets.sort(reverse=True, key=lambda x: x[0])
        selected = [asset_id for _, asset_id in scored_assets[:self.max_assets_to_convert]]
        
        print(f"[Image3DConvert] Selected {len(selected)} critical assets from {len(all_assets)} total")
        for asset_id in selected:
            score = next(s for s, a in scored_assets if a == asset_id)
            print(f"  - {asset_id} (priority: {score})")
        
        return selected

    def _save_timing_log(self, output_dir: Path, timing_log: dict[str, Any]) -> None:
        """Save timing log to file."""
        log_file = output_dir / "3d_timing_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(timing_log, f, indent=2)
        print(f"[Image3DConvert] Timing log saved: {log_file}")


class ThreeJSAssemblyNode(AgentNode):
    """Assemble 3D models into a runnable three.js HTML scene."""

    async def execute(self, ctx: AutoEnvContext) -> None:
        if not ctx.models_3d:
            ctx.error = "ThreeJSAssemblyNode requires 3D models from Image3DConvertNode"
            return

        if not ctx.strategy:
            ctx.error = "ThreeJSAssemblyNode requires strategy"
            return

        game_dir = ctx.output_dir / "game"
        game_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt context for reproducibility
        self._save_checkpoint(ctx, game_dir)

        # Copy 3D models into game/models
        models_src = ctx.output_dir / "models_3d"
        models_dst = game_dir / "models"
        if models_src.exists():
            if models_dst.exists():
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)

        # Generate three.js HTML scene
        html_file = game_dir / "index.html"
        self._generate_threejs_scene(ctx, game_dir, html_file)

        # Optionally ask agent to enhance scene with custom logic
        if self.agent:
            enhancement_prompt = self._build_enhancement_prompt(ctx, html_file)
            await self.agent.run(request=enhancement_prompt)

        if html_file.exists():
            ctx.game_dir = game_dir
            ctx.game_file = html_file
            ctx.success = True
            print(f"[ThreeJSAssembly] ✓ Generated: {html_file}")
            print(f"[ThreeJSAssembly] To view: Open {html_file} in browser or run: python -m http.server --directory {game_dir}")

    def _save_checkpoint(self, ctx: AutoEnvContext, game_dir: Path) -> None:
        """Save pipeline context and templates for reproducibility."""
        prompt_dir = game_dir / "checkpoint"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Save node source
        try:
            import re
            nodes_path = Path(__file__)
            nodes_text = nodes_path.read_text(encoding="utf-8")
            match = re.search(r"class ThreeJSAssemblyNode[\s\S]*", nodes_text)
            node_content = match.group(0) if match else nodes_text
            
            # Optionally prepend instruction
            instruction_header = ""
            try:
                config_path = Path.cwd() / "config" / "env_skin_gen.yaml"
                if config_path.exists():
                    config = yaml.safe_load(config_path.read_text())
                    if config and "instruction" in config:
                        instruction_header = f"{config['instruction']}\n{'=' * 51}\n"
            except Exception:
                pass
            
            full_content = instruction_header + node_content
            (prompt_dir / "ThreeJSAssemblyNode.txt").write_text(full_content, encoding="utf-8")
        except Exception as e:
            print(f"[ThreeJSAssembly] Warning: failed to save node source: {e}")

        # Save three.js template
        try:
            template_path = Path(__file__).parent / "threejs_template.html"
            if template_path.exists():
                content = template_path.read_text(encoding="utf-8")
                (prompt_dir / "threejs_template.html").write_text(content, encoding="utf-8")
        except Exception as e:
            print(f"[ThreeJSAssembly] Warning: failed to save template: {e}")

    def _generate_threejs_scene(self, ctx: AutoEnvContext, game_dir: Path, html_file: Path) -> None:
        """Generate three.js HTML file from template."""
        # Read template
        template_path = Path(__file__).parent / "threejs_template.html"
        with open(template_path, encoding="utf-8") as f:
            template = f.read()

        # Build model paths mapping
        model_paths = {}
        for asset_id, model_info in ctx.models_3d.items():
            # Use relative path: models/asset_id.glb
            model_paths[asset_id] = f"models/{asset_id}.glb"

        # Generate positioning code based on strategy
        assets = ctx.strategy.get("assets", [])
        positioning_code = self._generate_positioning_code(assets)

        # Generate animation code (placeholder for now, can be enhanced)
        animation_code = self._generate_animation_code(assets)

        # Fill template
        html_content = template.replace(
            "{MODEL_COUNT}", str(len(ctx.models_3d))
        ).replace(
            "{MODEL_PATHS_JSON}", json.dumps(model_paths, indent=12)
        ).replace(
            "{MODEL_POSITIONING_CODE}", positioning_code
        ).replace(
            "{ANIMATION_CODE}", animation_code
        )

        html_file.write_text(html_content, encoding="utf-8")

    def _generate_positioning_code(self, assets: list[dict[str, Any]]) -> str:
        """Generate JavaScript code to position models based on strategy with bounding box awareness."""
        import math
        
        positioning = []
        
        for i, asset in enumerate(assets):
            asset_id = asset.get("id", f"asset_{i}")
            asset_name = asset.get("name", "").lower()
            
            # Skip UI elements
            if any(keyword in asset_name for keyword in ["ui", "overlay", "hud", "button"]):
                continue
            
            # Player/character models: center placement with auto height adjustment
            if "player" in asset_name or "character" in asset_name:
                positioning.append(f"""
                    if (assetId === '{asset_id}') {{
                        const bbox = new THREE.Box3().setFromObject(model);
                        const modelHeight = bbox.max.y - bbox.min.y;
                        const minY = bbox.min.y;
                        
                        // Center placement
                        model.position.set(0, 0, 0);
                        
                        // Auto lift: ensure bottom of model is above ground
                        if (Number.isFinite(minY) && minY < 0) {{
                            model.position.y += Math.abs(minY) + 0.1;
                        }}
                    }}""")
            
            # Background/environment models: scale and position
            elif "background" in asset_name or "environment" in asset_name:
                positioning.append(f"""
                    if (assetId === '{asset_id}') {{
                        model.position.y = -0.5;
                        model.scale.setScalar(2.0);
                    }}""")
            
            # Other props: circular layout with auto height adjustment
            else:
                angle = (i / max(len(assets), 1)) * 2 * math.pi
                x = 3 * math.cos(angle)
                z = 3 * math.sin(angle)
                
                positioning.append(f"""
                    if (assetId === '{asset_id}') {{
                        const bbox = new THREE.Box3().setFromObject(model);
                        const minY = bbox.min.y;
                        
                        // Circular layout
                        model.position.set({x:.2f}, 0, {z:.2f});
                        
                        // Auto adjust height to prevent sinking
                        if (Number.isFinite(minY) && minY < 0) {{
                            model.position.y += Math.abs(minY) + 0.05;
                        }}
                    }}""")
        
        return "\n".join(positioning) if positioning else "// Default positioning"

    def _generate_animation_code(self, assets: list[dict[str, Any]]) -> str:
        """Generate JavaScript animation code."""
        animations = []
        
        for asset in assets:
            asset_id = asset.get("id")
            asset_name = asset.get("name", "").lower()
            
            # Add simple rotation animation for non-character objects
            if "box" in asset_name or "crate" in asset_name or "cube" in asset_name:
                animations.append(f"""
            if (models['{asset_id}']) {{
                models['{asset_id}'].rotation.y += 0.01;
            }}""")
        
        return "\n".join(animations) if animations else "// No animations"

    def _build_enhancement_prompt(self, ctx: AutoEnvContext, html_file: Path) -> str:
        """Build prompt for agent to enhance three.js scene."""
        strategy_json = json.dumps(ctx.strategy, indent=2, ensure_ascii=False)
        model_list = "\n".join([f"- {asset_id}.glb" for asset_id in ctx.models_3d.keys()])
        
        return f"""You are enhancing a three.js 3D game scene. The current HTML file is at: {html_file}

    Available 3D models:
    {model_list}

    Game strategy:
    {strategy_json}

    Goal: Deliver a polished, visually refined and interactive 3D scene that plays smoothly, respects the strategy, and avoids rendering/interaction bugs.

    Tasks:
    1) Review the generated scene in {html_file}.
    2) Add robust interactive logic (e.g., player/agent movement, collision detection with boundaries/obstacles, clear objectives with win/lose or success/fail conditions.
    3) Improve model positioning to match the strategy (e.g., spawn zones, obstacles, collectibles, goals). Avoid overlapping or sunken models.
    4) Add sensible animations (e.g., idle, walk/run, interact) and simple feedback (highlights) that do not break performance.
    5) Implement camera controls: keep ORBIT; add FIRST-PERSON or follow camera only if it fits the game; prevent clipping through ground/objects and keep near/far planes reasonable.
    6) Add game state management (score/health/progress), UI hints if helpful, and reset/restart hooks.

     Quality & safety requirements:
     - Keep using the existing three.js CDN imports (modelPaths:models/asset_name.glb).
     - Add three.js CDN assets for environmental enrichment if needed.
     - Ensure stable rendering: enable shadows carefully, cap lights, avoid excessive post-processing; keep frame rate smooth.
     - Ensure reliable interaction: solid ground, no falling through, prevent camera from going underground, clamp movement to play area, and guard against null/undefined models before use.
     - Ensure the scene is playable and matches the strategy
     - Keep the scene bright/clear (light background + ambientLight ≥ 1.0), and ensure player models visible and stay fully above ground/floor (use bounding box to lift if needed; add lights if still dark).

     Output:
     - Write back a single, working {html_file} with enhanced gameplay, controls, brightness/visibility, and safety checks (no separate files required)."""
