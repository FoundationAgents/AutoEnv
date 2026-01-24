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
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from datetime import datetime
from pathlib import Path

import yaml

from autoenv.pipeline import VisualPipeline
from autoenv.pipeline.visual.nodes import ThreeJSAssemblyNode, AutoEnvContext
from base.engine.cost_monitor import CostMonitor

DEFAULT_CONFIG = "config/env_skin_gen.yaml"

logger = logging.getLogger("skin_gen")


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


async def test_prompt_mode(
    input_path: Path,
    output_dir: Path,
    model: str,
    image_model: str,
    requirements: str = "",
    envs_root_path: str = "",
):
    """Test ThreeJSAssemblyNode enhancement prompt with existing .glb files.
    
    Args:
        input_path: Directory containing models_3d/ folder with .glb files
        output_dir: Output directory for the test
        model: LLM model name
        image_model: Image model name (unused in test mode)
        requirements: User requirements/instruction text
        envs_root_path: Root path for environments
    """
    logger.info("🧪 TEST PROMPT MODE: Testing ThreeJSAssemblyNode enhancement prompt")
    
    # Create timestamped output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_output_dir = output_dir / f"test_prompt_{ts}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reconfigure logger to write to the new timestamped directory
    log_file = test_output_dir / "log.txt"
    
    # Remove old handlers and add new ones pointing to the timestamped log
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
    ]
    logger.handlers = handlers
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    # Log configuration parameters
    logger.info("=" * 60)
    logger.info("TEST PROMPT MODE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Requirements: {requirements if requirements else '(Not provided)'}")
    logger.info(f"Envs Root Path: {envs_root_path if envs_root_path else '(Not provided)'}")
    logger.info(f"Input Path: {input_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Output Directory: {test_output_dir}")
    logger.info("=" * 60)
    logger.info("")
    
    # Verify input_path exists
    if not input_path.exists():
        logger.error(f"❌ input_path not found: {input_path}")
        return
    
    # Check for models_3d directory with .glb files
    models_dir = input_path / "models_3d"
    if not models_dir.exists():
        logger.error(f"❌ models_3d directory not found: {models_dir}")
        return
    
    glb_files = list(models_dir.glob("*.glb"))
    if not glb_files:
        logger.error(f"❌ No .glb files found in {models_dir}")
        return
    
    logger.info(f"✓ Found {len(glb_files)} .glb files in {models_dir}")
    
    # Create output game directory
    game_dir = test_output_dir / "game"
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy models_3d to game/models
    import shutil
    models_dst = game_dir / "models"
    if models_dst.exists():
        shutil.rmtree(models_dst)
    shutil.copytree(models_dir, models_dst)
    logger.info(f"✓ Copied models to {models_dst}")
    
    # Try to load strategy.json from input_path
    strategy = {}
    strategy_file = input_path / "strategy.json"
    if strategy_file.exists():
        with open(strategy_file, encoding="utf-8") as f:
            strategy = json.load(f)
        logger.info(f"✓ Loaded strategy from {strategy_file}")
    else:
        # Create minimal strategy if not found
        strategy = {
            "rendering_approach": {"type": "grid_3d"},
            "style_anchor": glb_files[0].stem,
            "assets": [
                {
                    "id": glb_file.stem,
                    "name": glb_file.stem,
                    "type": "game_object",
                    "priority": 10
                }
                for glb_file in glb_files
            ]
        }
        logger.info(f"⚠️  No strategy.json found, created minimal strategy for {len(glb_files)} models")
    
    # Create context with models_3d
    ctx = AutoEnvContext(
        output_dir=test_output_dir,
        strategy=strategy,
        models_3d={
            glb_file.stem: {
                "path": f"models/{glb_file.name}",
                "asset_id": glb_file.stem
            }
            for glb_file in glb_files
        }
    )
    
    logger.info(f"🔨 Creating ThreeJSAssemblyNode with model: {model}")
    
    # Initialize ThreeJSAssemblyNode with agent
    from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
    
    # Create MiniSWE agent for enhancement
    agent = MiniSWEAutoEnvAgent(
        llm_name=model,
        mode="yolo",
        step_limit=60,
        cost_limit=12.0,
        environment_type="local",
        cwd=str(Path.cwd()),
    )
    
    node = ThreeJSAssemblyNode(agent=agent)
    
    # Execute node
    logger.info("⚡ Running ThreeJSAssemblyNode...")
    try:
        await node.execute(ctx)
        if ctx.success:
            logger.info(f"✅ Test complete! Generated: {ctx.game_file}")
            logger.info(f"📁 Game directory: {game_dir}")
            logger.info("📝 Enhancement prompt was sent to LLM for testing")
        else:
            logger.error(f"❌ Node execution failed: {ctx.error}")
    except Exception as e:
        logger.error(f"❌ Error during node execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


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
        logger.error("❌ Provide either 'exist_environment_path' or 'requirements'")
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
    logger.info(f"🎮 [{label}] Generating {dimension.upper()} scene...")
    
    # Validate Meshy config for 3D mode
    if dimension == "3d" and (not meshy_config or not meshy_config.get("api_key")):
        logger.error("❌ 3D mode requires Meshy API key. Set 'meshy.api_key' in config.")
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
        logger.info(f"✅ [{label}] Generation complete → {visual_output}")
        logger.info(f"⏱️  Total time: {elapsed:.1f}s")
        
        # Show 3D model info if available
        if dimension == "3d" and hasattr(ctx, "models_3d") and ctx.models_3d:
            logger.info(f"🧊 3D Models generated: {len(ctx.models_3d)}")
            for model_id, info in ctx.models_3d.items():
                logger.info(f"   - {model_id}: {info.get('path')}")
    else:
        logger.error(f"❌ [{label}] Generation failed: {ctx.error}")


async def main():
    parser = argparse.ArgumentParser(description="Generate visual skins for environments")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument("--env", help="Override: existing environment path")
    parser.add_argument("--instruction", help="Override: instruction/requirements text")
    parser.add_argument("--model", help="Override: LLM model name")
    parser.add_argument("--image-model", help="Override: image model name")
    parser.add_argument("--output", help="Override: output directory")
    parser.add_argument("--input-path", help="Test mode: path with .glb files to test ThreeJSAssemblyNode")
    
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
    input_path = args.input_path or cfg.get("input_path")
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging: console + rotating file in envs_root_path/log.txt
    # NOTE: test_prompt_mode will reconfigure logging for timestamped directory
    log_file = output_dir / "log.txt"
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    # ===== TEST PROMPT MODE =====
    if input_path:
        input_path_obj = Path(input_path).expanduser().resolve()
        # Check if it has .glb files
        models_dir = input_path_obj / "models_3d"
        has_glb = models_dir.exists() and any(models_dir.glob("*.glb"))
        
        if has_glb:
            logger.info("🧪 TEST PROMPT MODE DETECTED")
            logger.info(f"📁 Input path: {input_path_obj}")
            logger.info(f"🤖 Model: {model}")
            
            with CostMonitor() as monitor:
                await test_prompt_mode(
                    input_path=input_path_obj,
                    output_dir=output_dir,
                    model=model,
                    image_model=image_model,
                    requirements=instruction or "",
                    envs_root_path=output,
                )
                
                # Print cost summary
                summary = monitor.summary()
                if summary["call_count"] > 0:
                    logger.info("\n" + "=" * 50)
                    logger.info("💰 Test Prompt Cost Summary")
                    logger.info("=" * 50)
                    logger.info(f"Total Cost: ${summary['total_cost']:.4f}")
                    logger.info(f"Total Calls: {summary['call_count']}")
                    logger.info(f"Input Tokens: {summary['total_input_tokens']:,}")
                    logger.info(f"Output Tokens: {summary['total_output_tokens']:,}")
                    
                    if summary["by_model"]:
                        logger.info("\nBy Model:")
                        for model_name, stats in summary["by_model"].items():
                            logger.info(f"  {model_name}: ${stats['cost']:.4f} ({stats['calls']} calls)")
                    
                    cost_file = monitor.save()
                    logger.info(f"\n📊 Cost saved: {cost_file}")
            return
    
    # ===== NORMAL PIPELINE MODE =====
    
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
        logger.error("❌ No image_model configured. Set 'image_model' in config or --image-model")
        return

    # Validate exist_env_path if provided
    if exist_env_path:
        exist_env_path = Path(exist_env_path)
        if not exist_env_path.exists():
            logger.error(f"❌ Environment path not found: {exist_env_path}")
            return

    logger.info(f"🔧 Config: {args.config}")
    logger.info(f"🤖 Model: {model}")
    logger.info(f"🎨 Image Model: {image_model}")
    logger.info(f"📁 Output: {output}")
    logger.info(f"📐 Dimension: {dimension.upper()}")
    if dimension == "3d":
        logger.info(f"🧊 Meshy API: {meshy_config.get('base_url', 'https://api.meshy.ai/v1')}")
    if exist_env_path:
        logger.info(f"📂 Environment: {exist_env_path}")
    if instruction:
        logger.info(f"📝 Instruction: {instruction[:50]}...")

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
        logger.info("\n" + "=" * 50)
        logger.info("💰 Cost Summary")
        logger.info("=" * 50)
        logger.info(f"Total Cost: ${summary['total_cost']:.4f}")
        logger.info(f"Total Calls: {summary['call_count']}")
        logger.info(f"Input Tokens: {summary['total_input_tokens']:,}")
        logger.info(f"Output Tokens: {summary['total_output_tokens']:,}")

        if summary["by_model"]:
            logger.info("\nBy Model:")
            for model_name, stats in summary["by_model"].items():
                logger.info(f"  {model_name}: ${stats['cost']:.4f} ({stats['calls']} calls)")

        cost_file = monitor.save()
        logger.info(f"\n📊 Cost saved: {cost_file}")


if __name__ == "__main__":
    asyncio.run(main())

