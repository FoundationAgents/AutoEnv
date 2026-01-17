"""
Skin Generation Entry Point

Generates visual assets for environments using VisualPipeline.
Supports both 2D and 3D generation modes.

Modes:
  1. Instruction mode: Use `requirements` as input prompt
  2. Existing environment mode: Use `exist_environment_path` to analyze and visualize

Dimension modes:
  - 2D (default): Generate 2D image assets
  - 3D: Generate 2D assets and convert to 3D models via Meshy API

Usage:
    # 2D generation (default)
    python run_skin_generation.py --config config/env_skin_gen.yaml
    python run_skin_generation.py --instruction "A pixel art dungeon game"
    
    # 3D generation
    python run_skin_generation.py --3d --instruction "A 3D Sokoban puzzle game"
    python run_skin_generation.py --dimension 3d --instruction "A 3D puzzle game"
"""

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path

import yaml

from autoenv.pipeline import VisualPipeline
from base.engine.cost_monitor import CostMonitor

DEFAULT_CONFIG = "config/env_skin_gen.yaml"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


async def run_skin_gen(
    model: str,
    image_model: str,
    output_dir: Path,
    exist_env_path: Path | None = None,
    instruction: str | None = None,
    dimension: str = "2d",
    meshy_config: dict | None = None,
):
    """Run skin generation pipeline.
    
    Args:
        model: LLM model name
        image_model: Image generation model name
        output_dir: Output directory
        exist_env_path: Path to existing environment
        instruction: Text instruction for generation
        dimension: "2d" or "3d"
        meshy_config: Meshy API configuration for 3D mode
    """
    if not exist_env_path and not instruction:
        print("❌ Provide either 'exist_environment_path' or 'requirements'")
        return

    # Timing
    start_time = time.time()

    # Determine output location with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_suffix = "_3d" if dimension == "3d" else ""
    if exist_env_path:
        label = exist_env_path.name
        visual_output = exist_env_path / f"visual{dim_suffix}_{ts}"
    else:
        label = instruction[:30] + "..." if len(instruction) > 30 else instruction
        visual_output = output_dir / f"visual{dim_suffix}_{ts}"

    visual_output.mkdir(parents=True, exist_ok=True)

    # Create pipeline with dimension support
    print(f"🎮 [{label}] Generating {dimension.upper()} scene...")
    
    # Validate Meshy config for 3D mode
    if dimension == "3d" and (not meshy_config or not meshy_config.get("api_key")):
        print("❌ 3D mode requires Meshy API key. Set 'meshy.api_key' in config.")
        return
    
    # Create unified pipeline
    pipeline = VisualPipeline.create_default(
        llm_name=model,
        image_model=image_model,
        dimension=dimension,
        meshy_api_key=meshy_config.get("api_key", "") if meshy_config else "",
        meshy_base_url=meshy_config.get("base_url", "https://api.meshy.ai/v1") if meshy_config else "https://api.meshy.ai/v1",
        max_3d_assets=meshy_config.get("max_assets", 4) if meshy_config else 4,
        target_polycount=meshy_config.get("target_polycount", 10000) if meshy_config else 10000,
    )

    ctx = await pipeline.run(
        benchmark_path=exist_env_path,
        instruction=instruction,
        output_dir=visual_output,
    )

    # Calculate elapsed time
    elapsed = time.time() - start_time

    if ctx.success:
        print(f"✅ [{label}] Generation complete → {visual_output}")
        print(f"⏱️  Total time: {elapsed:.1f}s")
        
        # Show 3D model info if available
        if dimension == "3d" and hasattr(ctx, "models_3d") and ctx.models_3d:
            print(f"🧊 3D Models generated: {len(ctx.models_3d)}")
            for model_id, info in ctx.models_3d.items():
                print(f"   - {model_id}: {info.get('path')}")
    else:
        print(f"❌ [{label}] Generation failed: {ctx.error}")


async def main():
    parser = argparse.ArgumentParser(description="Generate visual skins for environments")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--env", help="Override: existing environment path")
    parser.add_argument("--instruction", help="Override: instruction/requirements text")
    parser.add_argument("--model", help="Override: LLM model name")
    parser.add_argument("--image-model", help="Override: image model name")
    parser.add_argument("--output", help="Override: output directory")
    
    # 3D generation options
    parser.add_argument("--3d", dest="enable_3d", action="store_true",
                        help="Enable 3D generation mode")
    parser.add_argument("--dimension", choices=["2d", "3d"], default=None,
                        help="Generation dimension: 2d or 3d")
    parser.add_argument("--meshy-key", help="Override: Meshy API key")
    parser.add_argument("--max-3d-assets", type=int, default=None,
                        help="Maximum number of assets to convert to 3D")
    
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config
    model = args.model or cfg.get("model") or "claude-sonnet-4-5"
    image_model = args.image_model or cfg.get("image_model")
    output = args.output or cfg.get("envs_root_path") or "workspace/envs"
    exist_env_path = args.env or cfg.get("exist_environment_path")
    instruction = args.instruction or cfg.get("requirements")
    
    # Determine dimension (CLI takes priority)
    if args.enable_3d:
        dimension = "3d"
    elif args.dimension:
        dimension = args.dimension
    else:
        dimension = cfg.get("dimension", "2d")

    # Meshy configuration for 3D
    meshy_config = cfg.get("meshy", {})
    if args.meshy_key:
        meshy_config["api_key"] = args.meshy_key
    if args.max_3d_assets:
        meshy_config["max_assets"] = args.max_3d_assets

    if not image_model:
        print("❌ No image_model configured. Set 'image_model' in config or --image-model")
        return

    # Validate exist_env_path if provided
    if exist_env_path:
        exist_env_path = Path(exist_env_path)
        if not exist_env_path.exists():
            print(f"❌ Environment path not found: {exist_env_path}")
            return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Config: {args.config}")
    print(f"🤖 Model: {model}")
    print(f"🎨 Image Model: {image_model}")
    print(f"📁 Output: {output}")
    print(f"📐 Dimension: {dimension.upper()}")
    if dimension == "3d":
        print(f"🧊 Meshy API: {meshy_config.get('base_url', 'https://api.meshy.ai/v1')}")
    if exist_env_path:
        print(f"📂 Environment: {exist_env_path}")
    if instruction:
        print(f"📝 Instruction: {instruction[:50]}...")

    with CostMonitor() as monitor:
        await run_skin_gen(
            model=model,
            image_model=image_model,
            output_dir=output_dir,
            exist_env_path=exist_env_path,
            instruction=instruction,
            dimension=dimension,
            meshy_config=meshy_config,
        )

        # Print and save cost summary
        summary = monitor.summary()
        print("\n" + "=" * 50)
        print("💰 Cost Summary")
        print("=" * 50)
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"Total Calls: {summary['call_count']}")
        print(f"Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Output Tokens: {summary['total_output_tokens']:,}")

        if summary["by_model"]:
            print("\nBy Model:")
            for model_name, stats in summary["by_model"].items():
                print(f"  {model_name}: ${stats['cost']:.4f} ({stats['calls']} calls)")

        cost_file = monitor.save()
        print(f"\n📊 Cost saved: {cost_file}")


if __name__ == "__main__":
    asyncio.run(main())

