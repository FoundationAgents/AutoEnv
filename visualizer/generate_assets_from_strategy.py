#!/usr/bin/env python3
"""
Standalone asset generation script.
Generate assets directly from strategy.json without running the full pipeline.

Usage:
    python visualizer/generate_assets_from_strategy.py strategy.json output_dir
    python visualizer/generate_assets_from_strategy.py strategy.json output_dir --model gemini-2.0-flash-exp
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load modules directly to avoid __init__ side effects
import importlib.util

# Manually load ImageGenAgent
spec = importlib.util.spec_from_file_location(
    "image_gen_agent",
    project_root / "visualizer" / "agents" / "image_gen_agent.py"
)
image_gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_gen_module)
ImageGenAgent = image_gen_module.ImageGenAgent

# Manually load ImageProcessor
spec = importlib.util.spec_from_file_location(
    "image_processor",
    project_root / "visualizer" / "agents" / "image_processor.py"
)
image_proc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_proc_module)
ImageProcessor = image_proc_module.ImageProcessor

from visualizer.core.core import DependencyScheduler


class StandaloneAssetGenerator:
    """Standalone asset generator."""

    # Pixel style variants
    PIXEL_STYLE_VARIANTS = [
        "8-bit retro pixel art style, NES era aesthetic, limited color palette",
        "16-bit pixel art style, SNES era aesthetic, rich colors and details",
        "32-bit pixel art style, GBA era aesthetic, smooth gradients",
        "minimalist pixel art, geometric shapes, flat colors, modern indie game style",
        "detailed pixel art, high pixel density, intricate patterns",
        "chunky pixel art, large pixels, bold outlines, Game Boy style",
        "isometric pixel art style, 3D perspective, detailed shading",
        "monochrome pixel art with single accent color, high contrast",
        "pastel pixel art, soft colors, cute kawaii aesthetic",
        "dark pixel art, moody atmosphere, limited light palette",
        "vibrant pixel art, saturated colors, energetic feel",
        "pixel art with dithering effects, gradient transitions, retro computer style"
    ]

    def __init__(
        self,
        strategy_file: Path,
        output_dir: Path,
        image_model: str = "gemini-2.5-flash-image-preview",
        target_size: int = 256,
        enable_refinement: bool = False,
        random_style: bool = True
    ):
        self.strategy_file = strategy_file
        self.output_dir = output_dir
        self.image_model = image_model
        self.target_size = target_size
        self.enable_refinement = enable_refinement
        self.random_style = random_style

        # Randomly choose a pixel style
        self.chosen_style = random.choice(self.PIXEL_STYLE_VARIANTS) if random_style else None

        # Load strategy
        with open(strategy_file, 'r', encoding='utf-8') as f:
            self.strategy = json.load(f)

        # Initialize components
        self.image_gen_agent = ImageGenAgent(model_name=image_model)
        self.image_processor = ImageProcessor(target_size=target_size)
        self.scheduler = DependencyScheduler()

    async def generate(self) -> Dict[str, Any]:
        """Generate all assets."""

        print(f"\n🎨 Standalone Asset Generator")
        print(f"   Strategy: {self.strategy_file.name}")
        print(f"   Output: {self.output_dir}")
        print(f"   Model: {self.image_model}")
        print(f"   Target Size: {self.target_size}x{self.target_size}")
        print(f"   Refinement: {'Enabled' if self.enable_refinement else 'Disabled'}")
        if self.chosen_style:
            print(f"   🎲 Random Style: {self.chosen_style}")
        print()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Extract asset list and dependencies
        # Supports 'assets_to_generate' or 'assets'
        assets_to_generate = self.strategy.get('assets_to_generate') or self.strategy.get('assets', [])

        if not assets_to_generate:
            print("❌ No assets found in strategy.json")
            return {'success': False, 'error': 'No assets to generate'}

        print(f"📋 Found {len(assets_to_generate)} assets to generate\n")

        # Build dependency graph
        print("🔗 Building dependency graph...")
        for asset in assets_to_generate:
            # Support either 'asset_id' or 'id'
            asset_id = asset.get('asset_id') or asset.get('id')
            dependencies = asset.get('dependencies', [])
            priority = asset.get('priority', 10)
            self.scheduler.add_asset(asset_id, dependencies, priority)

        # Validate DAG
        is_valid, error_msg = self.scheduler.validate_dag()
        if not is_valid:
            print(f"❌ Dependency graph validation failed: {error_msg}")
            return {'success': False, 'error': error_msg}

        print(f"   ✓ Dependency graph validated\n")

        # Build execution order
        execution_order = self.scheduler.get_execution_plan()
        print(f"📊 Execution order: {len(execution_order)} batches")
        for i, batch in enumerate(execution_order, 1):
            print(f"   Batch {i}: {batch}")
        print()

        # Generate assets
        generated_assets = {}
        # Support either 'asset_id' or 'id'
        asset_map = {(asset.get('asset_id') or asset.get('id')): asset for asset in assets_to_generate}

        total_assets = len(assets_to_generate)
        completed = 0

        for batch_idx, batch in enumerate(execution_order, 1):
            print(f"\n🎨 Batch {batch_idx}/{len(execution_order)}: Generating {len(batch)} assets in parallel")
            
            # Generate current batch in parallel
            tasks = []
            for asset_id in batch:
                asset_info = asset_map[asset_id]
                task = self._generate_single_asset(
                    asset_info,
                    generated_assets,
                    completed + 1,
                    total_assets
                )
                tasks.append(task)

            # Wait for batch to finish
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results
            for asset_id, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"   ❌ {asset_id}: {str(result)}")
                    generated_assets[asset_id] = None
                elif result and result.get('success'):
                    generated_assets[asset_id] = result['image_base64']
                    completed += 1
                    print(f"   ✓ {asset_id} ({completed}/{total_assets})")
                else:
                    error = result.get('error', 'Unknown error') if result else 'No result'
                    print(f"   ❌ {asset_id}: {error}")
                    generated_assets[asset_id] = None

        # Save assets to disk
        print(f"\n💾 Saving assets to {assets_dir}")
        saved_count = 0
        for asset_id, image_base64 in generated_assets.items():
            if image_base64:
                self._save_asset(asset_id, image_base64, assets_dir)
                saved_count += 1

        # Save metadata
        metadata = {
            'strategy_file': str(self.strategy_file),
            'total_assets': total_assets,
            'generated_assets': saved_count,
            'failed_assets': total_assets - saved_count,
            'image_model': self.image_model,
            'target_size': self.target_size,
            'assets': list(generated_assets.keys())
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Generation complete!")
        print(f"   Total: {total_assets}")
        print(f"   Success: {saved_count}")
        print(f"   Failed: {total_assets - saved_count}")
        print(f"   Output: {self.output_dir}")

        return {
            'success': True,
            'total': total_assets,
            'generated': saved_count,
            'failed': total_assets - saved_count,
            'output_dir': str(self.output_dir)
        }

    async def _generate_single_asset(
        self,
        asset_info: Dict[str, Any],
        generated_assets: Dict[str, str],
        current: int,
        total: int
    ) -> Dict[str, Any]:
        """Generate a single asset."""

        # Support either 'asset_id' or 'id'
        asset_id = asset_info.get('asset_id') or asset_info.get('id')

        # Collect dependency assets (for image-to-image)
        dependency_images = []
        for dep_id in asset_info.get('dependencies', []):
            if dep_id in generated_assets and generated_assets[dep_id]:
                dependency_images.append(generated_assets[dep_id])

        # Build generation prompt
        prompt_strategy = asset_info.get('prompt_strategy', {})
        base_prompt = prompt_strategy.get('base_prompt', '')

        # If anchor (no dependencies) and random style enabled, inject style description
        is_anchor = len(asset_info.get('dependencies', [])) == 0
        if is_anchor and self.chosen_style:
            # Inject chosen style into base_prompt
            base_prompt = base_prompt.replace("pixel art style", self.chosen_style)
            base_prompt = base_prompt.replace("pixel art", self.chosen_style)

        # Add technical requirements (white background for processing)
        tech_requirements = (
            "\n\nTECHNICAL REQUIREMENTS:\n"
            "- Use SOLID WHITE background (#FFFFFF)\n"
            "- Center the subject, fill 70-85% of canvas\n"
            "- Clean edges for easy background removal\n"
            "- No shadows or glow effects on background\n"
        )
        final_prompt = base_prompt.strip() + tech_requirements

        # Choose generation method based on dependencies
        if dependency_images:
            result = await self.image_gen_agent.generate_image_to_image(
                prompt=final_prompt,
                reference_images=dependency_images
            )
        else:
            result = await self.image_gen_agent.generate_text_to_image(
                prompt=final_prompt
            )

        if not result.get('success'):
            return result

        # Post-process image (background removal, crop, resize)
        raw_image = result['image_base64']
        processed = self.image_processor.process_asset(raw_image, asset_id)

        if processed.get('success'):
            result['image_base64'] = processed['image_base64']
            # Optional: print processing steps
            # for step in processed.get('steps', []):
            #     print(f"      {step}")

        return result

    def _save_asset(self, asset_id: str, image_base64: str, assets_dir: Path):
        """Save an asset to disk."""
        import base64
        
        # Remove potential data URL prefix
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        
        # Decode and save
        image_data = base64.b64decode(image_base64)
        output_file = assets_dir / f"{asset_id}.png"
        
        with open(output_file, 'wb') as f:
            f.write(image_data)


def main():
    """CLI entrypoint."""
    
    parser = argparse.ArgumentParser(
        description='Generate assets from strategy.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualizer/generate_assets_from_strategy.py strategy.json output_dir

  # Specify model
  python visualizer/generate_assets_from_strategy.py strategy.json output_dir --model gemini-2.0-flash-exp

  # Custom size
  python visualizer/generate_assets_from_strategy.py strategy.json output_dir --size 512

  # Enable refinement (optional)
  python visualizer/generate_assets_from_strategy.py strategy.json output_dir --refinement
        """
    )

    parser.add_argument(
        'strategy',
        type=str,
        help='Path to strategy.json file'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Output directory for generated assets'
    )

    parser.add_argument(
        '--model',
        '-m',
        default='gemini-2.5-flash-image-preview',
        help='Image generation model (default: gemini-2.5-flash-image-preview)'
    )

    parser.add_argument(
        '--size',
        '-s',
        type=int,
        default=256,
        help='Target asset size (default: 256)'
    )

    parser.add_argument(
        '--refinement',
        '-r',
        action='store_true',
        help='Enable AI refinement (not recommended, deterministic processing is more stable)'
    )

    parser.add_argument(
        '--no-random-style',
        action='store_true',
        help='Disable random style variation (use original prompts as-is)'
    )

    args = parser.parse_args()

    # Validate input file
    strategy_file = Path(args.strategy)
    if not strategy_file.exists():
        print(f"❌ Strategy file not found: {args.strategy}")
        sys.exit(1)

    if not strategy_file.suffix == '.json':
        print(f"❌ Strategy file must be a JSON file: {args.strategy}")
        sys.exit(1)

    # Create generator
    output_dir = Path(args.output)
    generator = StandaloneAssetGenerator(
        strategy_file=strategy_file,
        output_dir=output_dir,
        image_model=args.model,
        target_size=args.size,
        enable_refinement=args.refinement,
        random_style=not args.no_random_style
    )

    # Run generation
    try:
        result = asyncio.run(generator.generate())
        
        if result.get('success'):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
