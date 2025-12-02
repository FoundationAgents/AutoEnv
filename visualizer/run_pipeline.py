#!/usr/bin/env python3
"""
Asset Generation Pipeline - CLI entry point.
"""

import asyncio
import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualizer.core.pipeline import Pipeline
from visualizer.core.config import PipelineConfig, PresetConfigs


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description='Universal Asset Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark mode (generate from existing benchmark)
  python visualizer/run_pipeline.py benchmarks/20_GridNavigation
  python visualizer/run_pipeline.py benchmarks/20_GridNavigation --output my_output --preset fast

  # Instruction mode (generate from user description)
  python visualizer/run_pipeline.py --mode instruction --instruction "Side-scrolling pixel maze game" --output games/test5
  python visualizer/run_pipeline.py --mode instruction --instruction "Classic Tetris" --output tetris_game

  # Custom config file
  python visualizer/run_pipeline.py benchmarks/20_GridNavigation --config config.json

  # Resume from checkpoint (must use the same output dir)
  python visualizer/run_pipeline.py benchmarks/20_GridNavigation --output my_output --resume

Modes:
  benchmark     - generate from an existing benchmark (default)
  instruction   - generate from a natural-language description

Presets:
  fast          - fast mode, no refinement
  high_quality  - enable all optimizations
  debug         - verbose logging
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        default=None,
        help='Input source: benchmark mode expects a path (e.g., benchmarks/20_GridNavigation); optional in instruction mode'
    )

    parser.add_argument(
        '--mode',
        '-m',
        choices=['benchmark', 'instruction'],
        default='benchmark',
        help='Run mode: benchmark (existing benchmark) or instruction (user description)'
    )

    parser.add_argument(
        '--instruction',
        '-i',
        default=None,
        help='Game description (instruction mode only), e.g., "A side-view box-pushing game with 3x3, 5x5, 13x13 difficulties"'
    )

    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output directory (default: visualizer/output/<benchmark_name>_<timestamp>)'
    )

    parser.add_argument(
        '--preset',
        '-p',
        choices=['fast', 'high_quality', 'debug'],
        default=None,
        help='Use a preset configuration'
    )

    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help='Custom config file path (JSON)'
    )

    parser.add_argument(
        '--no-refinement',
        action='store_true',
        help='Disable image refinement'
    )

    parser.add_argument(
        '--analyzer-model',
        default=None,
        help='Analyzer model (default: claude-sonnet-4-5)'
    )

    parser.add_argument(
        '--image-model',
        default=None,
        help='Image generation model (default: gemini-2.5-flash-image-preview)'
    )

    parser.add_argument(
        '--target-size',
        type=int,
        default=None,
        help='Target asset size (default: 256)'
    )

    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Do not append timestamp to output directory'
    )

    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Quiet mode, reduce console output'
    )

    parser.add_argument(
        '--resume',
        '-r',
        action='store_true',
        help='Resume from checkpoint'
    )

    return parser.parse_args()


def load_config(args) -> PipelineConfig:
    """Load PipelineConfig based on CLI arguments."""

    # 1) Load from config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {args.config}")
            sys.exit(1)

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = PipelineConfig.from_dict(config_dict)
        print(f"✅ Loaded config from: {args.config}")
        return config

    # 2) Load preset if provided
    if args.preset:
        preset_map = {
            'fast': PresetConfigs.fast,
            'high_quality': PresetConfigs.high_quality,
            'debug': PresetConfigs.debug
        }
        config = preset_map[args.preset]()
        print(f"✅ Using preset: {args.preset}")
    else:
        # 3) Use default config
        config = PipelineConfig()

    # 4) Apply CLI overrides
    if args.no_refinement:
        config.generation.enable_refinement = False

    if args.analyzer_model:
        config.model.analyzer = args.analyzer_model
        config.model.strategist = args.analyzer_model  # strategist uses same model

    if args.image_model:
        config.model.image_generator = args.image_model

    if args.target_size:
        config.generation.target_asset_size = args.target_size

    if args.no_timestamp:
        config.output.use_timestamp = False

    if args.quiet:
        config.verbose = False

    # Set run mode
    config.mode = args.mode

    return config


async def main():
    """Program entrypoint."""

    args = parse_args()

    # Validate inputs
    if args.mode == 'benchmark':
        if not args.input:
            print("❌ Benchmark mode requires a benchmark path")
            print("   Example: python visualizer/run_pipeline.py benchmarks/20_GridNavigation")
            sys.exit(1)

        benchmark_path = Path(args.input)
        if not benchmark_path.exists():
            print(f"❌ Benchmark not found: {args.input}")
            sys.exit(1)

        input_source = str(benchmark_path)
        instruction = None

    elif args.mode == 'instruction':
        if not args.instruction:
            print("❌ Instruction mode requires a game description")
            print("   Example: python visualizer/run_pipeline.py --mode instruction --instruction 'A side-view box-pushing game'")
            sys.exit(1)

        input_source = args.instruction.replace(' ', '_')[:50]  # used for naming the output directory
        instruction = args.instruction

    else:
        print(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)

    # Load configuration
    config = load_config(args)

    # Create pipeline
    pipeline = Pipeline(config=config)

    try:
        # Run pipeline
        result = await pipeline.run(
            input_source=input_source,
            output_dir=args.output,
            resume=args.resume,
            instruction=instruction
        )

        # Check result
        if 'error' in result:
            print(f"\n❌ Pipeline failed: {result['error']}")
            sys.exit(1)

        # Success
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Output: {result['output_dir']}")
        print(f"🎨 Generated: {len(result['generated_assets'])} assets")

    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
