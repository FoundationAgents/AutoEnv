"""
Skin Generation Entry Point

Generates visual assets for environments using VisualPipeline.

Two modes:
  1. Instruction mode: Use `requirements` as input prompt
  2. Existing environment mode: Use `exist_environment_path` to analyze and visualize

Usage:
    python run_skin_generation.py --config config/env_skin_gen.yaml
    python run_skin_generation.py --env benchmarks/01_Maze
    python run_skin_generation.py --instruction "A pixel art dungeon game"
"""

import argparse
import asyncio
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
):
    """Run skin generation pipeline."""
    if not exist_env_path and not instruction:
        print("❌ Provide either 'exist_environment_path' or 'requirements'")
        return

    # Determine output location with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exist_env_path:
        label = exist_env_path.name
        visual_output = exist_env_path / f"visual_{ts}"
    else:
        label = instruction[:30] + "..." if len(instruction) > 30 else instruction
        visual_output = output_dir / f"visual_{ts}"

    visual_output.mkdir(parents=True, exist_ok=True)

    print(f"🎨 [{label}] Generating visuals...")
    pipeline = VisualPipeline.create_default(
        llm_name=model,
        image_model=image_model,
    )

    ctx = await pipeline.run(
        benchmark_path=exist_env_path,
        instruction=instruction,
        output_dir=visual_output,
    )

    if ctx.success:
        print(f"✅ [{label}] Visuals generated → {visual_output}")
    else:
        print(f"❌ [{label}] Visual generation failed: {ctx.error}")


async def main():
    parser = argparse.ArgumentParser(description="Generate visual skins for environments")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--env", help="Override: existing environment path")
    parser.add_argument("--instruction", help="Override: instruction/requirements text")
    parser.add_argument("--model", help="Override: LLM model name")
    parser.add_argument("--image-model", help="Override: image model name")
    parser.add_argument("--output", help="Override: output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config
    model = args.model or cfg.get("model") or "claude-sonnet-4-5"
    image_model = args.image_model or cfg.get("image_model")
    output = args.output or cfg.get("envs_root_path") or "workspace/envs"
    exist_env_path = args.env or cfg.get("exist_environment_path")
    instruction = args.instruction or cfg.get("requirements")

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

