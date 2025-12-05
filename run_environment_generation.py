"""
Environment Generation Entry Point

Usage:
    python run_environment_generation.py --config config/env_gen.yaml
    python run_environment_generation.py --theme "A puzzle game"
"""

import argparse
import asyncio
from pathlib import Path

import yaml

from autoenv.pipeline import VisualPipeline, GeneratorPipeline
from base.engine.cost_monitor import CostMonitor

DEFAULT_CONFIG = "config/env_gen.yaml"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def get_themes(themes_folder: str) -> list[str]:
    folder = Path(themes_folder)
    if not folder.is_dir():
        return []
    return sorted(str(f) for f in folder.glob("*.txt"))


async def run_generation(
    theme: str,
    model: str,
    output: str,
    mode: str = "textual",
    image_model: str | None = None,
):
    """Run a single generation task."""
    label = theme
    if theme.endswith(".txt") and Path(theme).exists():
        label = Path(theme).stem
        theme = Path(theme).read_text(encoding="utf-8")

    # Step 1: Run generator pipeline
    print(f"🚀 [{label}] Generating environment...")
    gen_pipeline = GeneratorPipeline.create_default(llm_name=model)
    gen_ctx = await gen_pipeline.run(requirements=theme, output_dir=output)

    if not gen_ctx.success:
        print(f"❌ [{label}] Generation failed: {gen_ctx.error}")
        return

    env_path = gen_ctx.env_folder_path
    print(f"✅ [{label}] Environment generated → {env_path}")

    # Step 2: Run visual pipeline if multimodal mode
    if mode == "multimodal":
        if not image_model:
            print(f"⚠️ [{label}] Skipping visual: no image_model configured")
            return

        print(f"🎨 [{label}] Generating visuals...")
        visual_output = env_path / "visual"
        visual_pipeline = VisualPipeline.create_default(
            llm_name=model,
            image_model=image_model,
        )
        visual_ctx = await visual_pipeline.run(
            benchmark_path=env_path,
            output_dir=visual_output,
        )

        if visual_ctx.success:
            print(f"✅ [{label}] Visuals generated → {visual_output}")
        else:
            print(f"❌ [{label}] Visual generation failed: {visual_ctx.error}")


async def main():
    parser = argparse.ArgumentParser(description="Generate environments")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--theme", help="Override: single theme text or .txt file")
    parser.add_argument("--model", help="Override: LLM model name")
    parser.add_argument("--output", help="Override: output directory")
    parser.add_argument("--mode", choices=["textual", "multimodal"], help="Override mode")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config
    model = args.model or cfg.get("model") or "claude-sonnet-4-5"
    output = args.output or cfg.get("envs_root_path") or "workspace/envs"
    mode = args.mode or cfg.get("mode") or "textual"
    image_model = cfg.get("image_model")
    concurrency = cfg.get("concurrency", 1)

    Path(output).mkdir(parents=True, exist_ok=True)
    print(f"🔧 Config: {args.config}")
    print(f"🤖 Model: {model}")
    print(f"🎨 Image Model: {image_model}")
    print(f"📁 Output: {output}")
    print(f"📦 Mode: {mode}")

    # Determine themes (priority: CLI --theme > themes_folder > theme)
    themes: list[str] = []
    if args.theme:
        themes = [args.theme]
    elif cfg.get("themes_folder"):
        themes = get_themes(cfg["themes_folder"])
    elif cfg.get("theme"):
        themes = [cfg["theme"]]

    if not themes:
        print("❌ No theme provided. Set 'theme' or 'themes_folder' in config.")
        return

    # Concurrent execution with cost tracking
    sem = asyncio.Semaphore(concurrency)

    async def task(t: str):
        async with sem:
            await run_generation(t, model, output, mode, image_model)

    with CostMonitor() as monitor:
        await asyncio.gather(*[task(t) for t in themes])

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
