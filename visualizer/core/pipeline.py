"""
Core Pipeline Class for asset generation.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from visualizer.core.config import PipelineConfig
from visualizer.core.core import OutputManager, CheckpointManager, DependencyScheduler, StrategyParser
from visualizer.core.logger import setup_logger, PipelineLogger

from visualizer.agents.analyzer_agent import AnalyzerAgent
from visualizer.agents.instruction_analyzer_agent import InstructionAnalyzerAgent
from visualizer.agents.strategist_agent import StrategistAgent
from visualizer.agents.image_gen_agent import ImageGenAgent
from visualizer.agents.refinement_agent import RefinementAgent
from visualizer.agents.asset_generator import AdaptiveAssetGenerator
from visualizer.agents.game_assembly_agent import GameAssemblyAgent


class Pipeline:
    """
    General-purpose asset generation pipeline, fully AI-driven.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: Pipeline configuration (defaults if None)
        """
        self.config = config or PipelineConfig()

        # Delay creation of logger/output_manager until run()
        self.logger: Optional[PipelineLogger] = None
        self.output: Optional[OutputManager] = None

        # Initialize agents
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents."""

        if self.config.model.use_agent:
            # Agent-based mode
            self.analyzer = AnalyzerAgent(
                llm_name=self.config.model.analyzer,
                step_limit=self.config.generation.agent_step_limit,
                cost_limit=self.config.generation.agent_cost_limit,
                timeout=self.config.generation.agent_timeout
            )
            self.instruction_analyzer = InstructionAnalyzerAgent(
                llm_name=self.config.model.analyzer,
                step_limit=self.config.generation.agent_step_limit * 2,
                cost_limit=self.config.generation.agent_cost_limit * 2,
                timeout=self.config.generation.agent_timeout * 2
            )
            self.strategist = StrategistAgent(
                llm_name=self.config.model.strategist,
                step_limit=self.config.generation.agent_step_limit * 2,
                cost_limit=self.config.generation.agent_cost_limit * 2,
                timeout=self.config.generation.agent_timeout * 2
            )
        else:
            # Simple LLM mode (placeholder for future)
            self.analyzer = None
            self.instruction_analyzer = None
            self.strategist = None

        # Image generation agent
        self.image_agent = ImageGenAgent(self.config.model.image_generator)
        self.asset_generator = AdaptiveAssetGenerator(self.image_agent)

        # Optional refinement agent
        if self.config.generation.enable_refinement:
            self.refinement_agent = RefinementAgent(
                analyzer_llm=self.config.model.analyzer,
                editor_model=self.config.model.image_generator
            )
        else:
            self.refinement_agent = None

        # Game assembly agent (build runnable pygame game)
        self.game_assembly_agent = GameAssemblyAgent(
            llm_name=self.config.model.analyzer,
            use_miniswe=self.config.model.use_agent,
            agent_step_limit=max(self.config.generation.agent_step_limit, 80),
            agent_cost_limit=max(self.config.generation.agent_cost_limit, 8.0),
            agent_timeout=max(self.config.generation.agent_timeout, 120)
        )

    async def run(
        self,
        input_source: str,
        output_dir: Optional[str] = None,
        resume: bool = False,
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            input_source: Benchmark path or instruction-name
            output_dir: Custom output directory (optional)
            resume: Resume from checkpoint
            instruction: User instruction (instruction mode only)

        Returns:
            Result dictionary
        """

        # Handle input by mode
        if self.config.mode == "benchmark":
            benchmark_path = Path(input_source)
            game_name = benchmark_path.name
        else:  # instruction mode
            benchmark_path = None
            game_name = input_source.replace(" ", "_")[:50]  # Use first 50 chars of instruction

        # Create output manager
        self.output = OutputManager(
            benchmark_name=game_name,
            config=self.config.output,
            custom_output_dir=output_dir
        )

        # Create checkpoint manager
        self.checkpoint = CheckpointManager(self.output.root_dir)

        # Create logger
        self.logger = setup_logger(
            name="pipeline",
            log_file=self.output.pipeline_log,
            verbose=self.config.verbose
        )

        # Start pipeline
        self.logger.stage("🚀 Universal Asset Generation Pipeline")
        self.logger.info(f"Mode: {self.config.mode}")
        if self.config.mode == "benchmark":
            self.logger.info(f"Benchmark: {benchmark_path}")
        else:
            self.logger.info(f"Instruction: {instruction}")
        self.logger.info(f"Output: {self.output.root_dir}")

        # Check for checkpoint
        if resume and self.checkpoint.exists():
            checkpoint_info = self.checkpoint.summary()
            self.logger.info(f"📂 Resuming from checkpoint: {checkpoint_info['stage']}")
            self.logger.info(f"   Timestamp: {checkpoint_info['timestamp']}")
            self.logger.info(f"   Completed assets: {checkpoint_info['completed_assets_count']}")
        elif resume:
            self.logger.warning("No checkpoint found, starting from beginning")
            resume = False

        # === Stage 1: Analysis ===
        if resume and self.checkpoint.is_stage_completed('analysis'):
            self.logger.info("✓ Stage 1 already completed, loading from checkpoint")
            analysis = self.output.load_json(self.config.output.analysis_file)
        else:
            if self.config.mode == "benchmark":
                self.logger.stage("Stage 1: Analyzing Benchmark", 1)
                analysis = await self._run_analysis_stage(benchmark_path)
            else:
                self.logger.stage("Stage 1: Analyzing Instruction", 1)
                analysis = await self._run_instruction_analysis_stage(instruction)

            if 'error' in analysis:
                self.logger.error(f"Analysis failed: {analysis.get('error')}")
                self._save_error_result(analysis, "01_analysis.json")
                return {'error': 'analysis_failed', 'analysis': analysis}

            # Save checkpoint
            self.checkpoint.save('analysis', analysis)

        self.logger.success("Analysis complete")
        self.logger.info(f"Theme: {analysis.get('visual_theme', 'Unknown')}")
        self.logger.info(f"Art Style: {analysis.get('art_style', 'Unknown')}")

        # === Stage 2: Strategy ===
        if resume and self.checkpoint.is_stage_completed('strategy'):
            self.logger.info("✓ Stage 2 already completed, loading from checkpoint")
            strategy = self.output.load_json(self.config.output.strategy_file)
        else:
            self.logger.stage("Stage 2: Devising Strategy", 2)
            strategy = await self._run_strategy_stage(analysis)

            if 'error' in strategy:
                self.logger.error(f"Strategy failed: {strategy.get('error')}")
                self._save_error_result(strategy, "02_strategy.json")
                return {'error': 'strategy_failed', 'strategy': strategy}

            # Save checkpoint
            self.checkpoint.save('strategy', strategy)

        self.logger.success("Strategy devised")
        self.logger.info(f"Rendering: {strategy.get('rendering_approach', {}).get('type', 'Unknown')}")
        self.logger.info(f"Assets: {len(strategy.get('assets', []))}")

        # === Stage 3: Generate assets ===
        self.logger.stage("Stage 3: Generating Assets", 3)

        generated_assets = await self._run_generation_stage(strategy)

        self.logger.success(f"Generated {len(generated_assets)} assets")

        # === Stage 4: Game assembly ===
        self.logger.stage("Stage 4: Game Assembly (Pygame)", 4)

        assembly_result = await self._run_game_assembly_stage(
            benchmark_path,
            analysis,
            generated_assets
        )

        if assembly_result.get('success'):
            self.logger.success("Game assembly complete!")
            self.logger.info(f"Game directory: {assembly_result['game_dir'].name}")
            self.logger.info(f"Entry point: {assembly_result['entry_point'].name}")
            self.logger.info(f"Assets: {assembly_result['assets_count']}")

            if assembly_result.get('code_valid'):
                self.logger.success("✅ Code validation passed")
            else:
                self.logger.warning("⚠️  Code validation warning")
        else:
            self.logger.warning(f"Game assembly failed: {assembly_result.get('error')}")
            # Continue; do not block pipeline

        # === Stage 5: Save results ===
        self.logger.stage("Stage 5: Saving Results", 5)

        manifest = self._create_manifest(game_name, strategy, generated_assets)
        self.output.save_json(manifest, self.config.output.manifest_file)

        self.logger.success("Pipeline Complete!")
        self.logger.info(f"Output: {self.output.root_dir}")
        self.logger.info(f"Assets: {len(generated_assets)}")

        return {
            'analysis': analysis,
            'strategy': strategy,
            'generated_assets': generated_assets,
            'assembly_result': assembly_result if assembly_result.get('success') else None,
            'manifest': manifest,
            'output_dir': str(self.output.root_dir)
        }

    async def _run_analysis_stage(self, benchmark_path: Path) -> Dict[str, Any]:
        """Run the analysis stage."""

        output_file = self.output.analysis_file
        log_file = self.output.get_agent_log_path("01_analysis")

        self.logger.info("Running Analyzer Agent...")

        analysis = await self.analyzer.analyze(
            benchmark_path=benchmark_path,
            output_file=output_file,
            log_file=log_file
        )

        # Add benchmark name
        if 'error' not in analysis:
            analysis['benchmark_name'] = benchmark_path.name

        # Save analysis
        if 'error' not in analysis:
            self.output.save_json(analysis, self.config.output.analysis_file)

        # Show log file path (prefer relative)
        try:
            rel_path = log_file.resolve().relative_to(Path.cwd().resolve())
            self.logger.info(f"Agent log: {rel_path}")
        except ValueError:
            # If relative calc fails, show absolute
            self.logger.info(f"Agent log: {log_file}")

        return analysis

    async def _run_instruction_analysis_stage(self, instruction: str) -> Dict[str, Any]:
        """Run instruction-based analysis stage."""

        output_file = self.output.analysis_file
        log_file = self.output.get_agent_log_path("01_instruction_analysis")

        self.logger.info("Running Instruction Analyzer Agent...")

        analysis = await self.instruction_analyzer.analyze_from_instruction(
            instruction=instruction,
            output_file=output_file,
            log_file=log_file
        )

        # Add metadata
        if 'error' not in analysis:
            analysis['source_type'] = 'instruction'
            analysis['original_instruction'] = instruction

        # Save analysis
        if 'error' not in analysis:
            self.output.save_json(analysis, self.config.output.analysis_file)

        # Show log path
        try:
            rel_path = log_file.resolve().relative_to(Path.cwd().resolve())
            self.logger.info(f"Agent log: {rel_path}")
        except ValueError:
            self.logger.info(f"Agent log: {log_file}")

        return analysis

    async def _run_strategy_stage(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run strategy stage."""

        # Temp file for agent input
        temp_analysis_file = self.output.root_dir / "temp_input_analysis.json"
        output_file = self.output.strategy_file
        log_file = self.output.get_agent_log_path("02_strategy")

        self.logger.info("Running Strategist Agent...")

        strategy = await self.strategist.devise_strategy(
            analysis=analysis,
            analysis_file=temp_analysis_file,
            output_file=output_file,
            log_file=log_file
        )

        # Save strategy
        if 'error' not in strategy:
            self.output.save_json(strategy, self.config.output.strategy_file)

        # Show log path
        try:
            rel_path = log_file.resolve().relative_to(Path.cwd().resolve())
            self.logger.info(f"Agent log: {rel_path}")
        except ValueError:
            self.logger.info(f"Agent log: {log_file}")

        return strategy

    async def _run_generation_stage(self, strategy: Dict[str, Any]) -> Dict[str, str]:
        """
        Run asset generation stage (parallel supported).

        Returns:
            {asset_id: image_base64} dict
        """
        # Choose parallel vs sequential
        if self.config.generation.enable_parallel and self.config.generation.use_dependency_graph:
            return await self._run_generation_parallel(strategy)
        else:
            return await self._run_generation_sequential(strategy)

    async def _run_generation_parallel(self, strategy: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate assets in parallel (DAG-based).

        Returns:
            {asset_id: image_base64} dict
        """
        import base64

        # Parse dependencies
        try:
            scheduler = StrategyParser.parse_dependencies(strategy)
        except Exception as e:
            self.logger.warning(f"Failed to parse dependencies: {e}, falling back to sequential")
            return await self._run_generation_sequential(strategy)

        # Show execution plan (simplified)
        plan = scheduler.get_execution_plan()
        self.logger.info(f"Execution batches: {plan}")

        # Check for previously generated assets (resume)
        completed_assets_list = self.checkpoint.get_completed_assets()
        generated_assets = {}  # {asset_id: image_b64}

        # Load completed assets
        if completed_assets_list:
            self.logger.info(f"Loading {len(completed_assets_list)} completed assets from checkpoint")
            for asset_id in completed_assets_list:
                asset_path = self.output.get_asset_path(
                    asset_id,
                    refined=self.config.generation.enable_refinement
                )
                if asset_path.exists():
                    with open(asset_path, 'rb') as f:
                        generated_assets[asset_id] = base64.b64encode(f.read()).decode()
                    self.logger.info(f"✓ Loaded: {asset_id}")
                    scheduler.mark_completed(asset_id)

        # Define generation task
        async def generate_task(asset_id: str, dependencies: List[str]):
            """Generate a single asset."""
            # Skip already generated
            if asset_id in generated_assets:
                return generated_assets[asset_id]

            # Find asset spec
            asset_spec = self._find_asset_spec(strategy, asset_id)
            if not asset_spec:
                raise ValueError(f"Asset {asset_id} not found in strategy")

            # Prepare reference images (all dependencies)
            reference_images = []
            ref_assets = asset_spec.get('reference_assets', dependencies)
            for ref_id in ref_assets:
                if ref_id in generated_assets:
                    reference_images.append(generated_assets[ref_id])

            self.logger.info(f"🎨 Generating: {asset_spec.get('name', asset_id)}")
            if reference_images:
                self.logger.info(f"   References: {ref_assets}")

            # Generate
            result = await self.asset_generator.generate_by_strategy(
                asset_spec=asset_spec,
                strategy=strategy,
                reference_images=reference_images
            )

            if not result.get('success'):
                raise Exception(f"Generation failed: {result.get('error', 'Unknown')}")

            raw_image_b64 = result['image_base64']

            # Save raw image
            raw_save_path = self.output.get_asset_path(asset_id, refined=False)
            self.image_agent.save_image(raw_image_b64, str(raw_save_path))

            # Optional refinement
            final_image_b64 = raw_image_b64
            if self.refinement_agent:
                refined_result = await self.refinement_agent.refine_asset(
                    raw_image_b64=raw_image_b64,
                    asset_spec=asset_spec,
                    target_size=self.config.generation.target_asset_size,
                    reference_assets=list(generated_assets.values()),
                    processing_guidance=asset_spec.get('processing_guidance')
                )

                if refined_result['success']:
                    final_image_b64 = refined_result['image_base64']
                    dims = refined_result['dimensions']
                    self.logger.info(f"   ✨ Refined: {dims[0]}x{dims[1]}px")

            # Save final image
            generated_assets[asset_id] = final_image_b64
            save_path = self.output.get_asset_path(
                asset_id,
                refined=self.config.generation.enable_refinement
            )
            self.image_agent.save_image(final_image_b64, str(save_path))

            # Save checkpoint
            completed_assets_list.append(asset_id)
            self.checkpoint.save(
                'generation',
                {'generated_count': len(completed_assets_list)},
                completed_assets=completed_assets_list
            )

            return final_image_b64

        # Progress callback
        total_assets = len(scheduler.nodes)
        completed_count = len(completed_assets_list)

        def on_progress(asset_id: str, status: str):
            nonlocal completed_count
            if status == 'completed':
                completed_count += 1
                self.logger.success(f"✅ [{completed_count}/{total_assets}] {asset_id}")
            elif status == 'started':
                self.logger.info(f"▶️  [{completed_count}/{total_assets}] Starting: {asset_id}")
            elif 'failed' in status:
                self.logger.error(f"❌ {asset_id}: {status}")

        # Execute with scheduler
        self.logger.info(f"\n🚀 Starting parallel generation (max_concurrent={self.config.generation.max_concurrent})")

        results = await scheduler.schedule(
            task_func=generate_task,
            on_progress=on_progress
        )

        # Check failed tasks
        failed = [aid for aid, result in results.items() if isinstance(result, dict) and 'error' in result]
        if failed:
            self.logger.warning(f"⚠️  {len(failed)} assets failed: {failed}")

        # Save final checkpoint
        self.checkpoint.save('completed', {'total_assets': len(generated_assets)}, completed_assets=completed_assets_list)

        return generated_assets

    async def _run_generation_sequential(self, strategy: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate assets sequentially (fallback).

        Returns:
            {asset_id: image_base64} dict
        """
        import base64

        workflow = strategy.get('generation_workflow', {})
        phases = workflow.get('phases', [])

        if not phases:
            self.logger.warning("No generation phases defined, creating fallback")
            phases = self._create_fallback_phases(strategy)

        # Check for previously generated assets (resume)
        completed_assets = self.checkpoint.get_completed_assets()
        generated_assets = {}  # {asset_id: image_b64}

        # Load completed assets
        if completed_assets:
            self.logger.info(f"Loading {len(completed_assets)} completed assets from checkpoint")
            for asset_id in completed_assets:
                asset_path = self.output.get_asset_path(
                    asset_id,
                    refined=self.config.generation.enable_refinement
                )
                if asset_path.exists():
                    with open(asset_path, 'rb') as f:
                        generated_assets[asset_id] = base64.b64encode(f.read()).decode()
                    self.logger.info(f"✓ Loaded: {asset_id}")

        for phase_idx, phase in enumerate(phases, 1):
            self.logger.phase(phase.get('phase_name', 'Unknown'), phase_idx)

            phase_strategy = phase.get('strategy', {})
            asset_ids = phase.get('assets', [])

            for idx, asset_id in enumerate(asset_ids, 1):
                # Skip already completed assets
                if asset_id in completed_assets:
                    self.logger.info(f"[{idx}/{len(asset_ids)}] ✓ Already completed: {asset_id}")
                    continue

                # Find asset spec
                asset_spec = self._find_asset_spec(strategy, asset_id)
                if not asset_spec:
                    self.logger.warning(f"Asset {asset_id} not found in strategy")
                    continue

                self.logger.progress(idx, len(asset_ids), asset_spec.get('name', asset_id))

                # Prepare reference images
                reference_images = []
                if phase_strategy.get('generation_method') == 'image-to-image':
                    style_anchor = phase_strategy.get('style_anchor')
                    if style_anchor and style_anchor in generated_assets:
                        reference_images = [generated_assets[style_anchor]]

                # Generate
                result = await self.asset_generator.generate_by_strategy(
                    asset_spec=asset_spec,
                    strategy=strategy,
                    reference_images=reference_images
                )

                if result.get('success'):
                    raw_image_b64 = result['image_base64']

                    # Save raw image (for reference)
                    raw_save_path = self.output.get_asset_path(asset_id, refined=False)
                    self.image_agent.save_image(raw_image_b64, str(raw_save_path))
                    self.logger.info(f"Raw: {raw_save_path.name}")

                    # Refinement (if enabled)
                    if self.refinement_agent:
                        self.logger.info("Refining with Vision Agent...")

                        refined_result = await self.refinement_agent.refine_asset(
                            raw_image_b64=raw_image_b64,
                            asset_spec=asset_spec,
                            target_size=self.config.generation.target_asset_size,
                            reference_assets=list(generated_assets.values()),
                            processing_guidance=asset_spec.get('processing_guidance')
                        )

                        if refined_result['success']:
                            final_image_b64 = refined_result['image_base64']
                            dims = refined_result['dimensions']
                            self.logger.success(f"Refined: {dims[0]}x{dims[1]}px")
                        else:
                            self.logger.warning("Refinement failed, using raw image")
                            final_image_b64 = raw_image_b64
                    else:
                        # Refinement disabled; use raw image
                        self.logger.info("Refinement disabled, using raw image")
                        final_image_b64 = raw_image_b64

                    # Save final image
                    generated_assets[asset_id] = final_image_b64
                    save_path = self.output.get_asset_path(
                        asset_id,
                        refined=self.config.generation.enable_refinement
                    )
                    self.image_agent.save_image(final_image_b64, str(save_path))
                    self.logger.info(f"Saved: {save_path.name}")

                    # Save checkpoint per asset
                    completed_assets.append(asset_id)
                    self.checkpoint.save(
                        'generation',
                        {'generated_count': len(completed_assets)},
                        completed_assets=completed_assets
                    )

                else:
                    self.logger.error(f"Generation failed: {result.get('error', 'Unknown')}")

        # Final checkpoint after generation
        self.checkpoint.save('completed', {'total_assets': len(generated_assets)}, completed_assets=completed_assets)

        return generated_assets

    def _create_fallback_phases(self, strategy: Dict[str, Any]) -> List[Dict]:
        """Create fallback phases if none are provided in strategy."""

        assets = strategy.get('assets', [])
        if not assets:
            return []

        return [{
            'phase_name': 'Sequential Generation',
            'description': 'Generate all assets sequentially',
            'assets': [asset.get('id', asset.get('name')) for asset in assets],
            'strategy': {
                'generation_method': 'text-to-image'
            }
        }]

    def _find_asset_spec(self, strategy: Dict[str, Any], asset_id: str) -> Optional[Dict[str, Any]]:
        """Find the spec for a given asset_id."""

        for asset in strategy.get('assets', []):
            if asset.get('id') == asset_id or asset.get('name') == asset_id:
                return asset
        return None

    def _create_manifest(
        self,
        benchmark_name: str,
        strategy: Dict[str, Any],
        generated_assets: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create asset manifest."""

        return {
            'benchmark': benchmark_name,
            'strategy': strategy.get('rendering_approach', {}),
            'generated_count': len(generated_assets),
            'assets': [
                {
                    'id': asset_id,
                    'name': self._find_asset_spec(strategy, asset_id).get('name', asset_id),
                    'path': str(self.output.get_asset_path(
                        asset_id,
                        refined=self.config.generation.enable_refinement
                    ).relative_to(self.output.root_dir))
                }
                for asset_id in generated_assets
            ]
        }

    async def _run_game_assembly_stage(
        self,
        benchmark_path: Optional[Path],
        analysis: Dict[str, Any],
        generated_assets: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Run game assembly stage to build a runnable pygame game.

        Args:
            benchmark_path: Benchmark path (None in instruction mode)
            analysis: Analysis results (game design info)
            generated_assets: Generated assets

        Returns:
            Assembly result with game directory and entry point
        """

        self.logger.info("Running Game Assembly Agent...")

        try:
            result = await self.game_assembly_agent.assemble_game(
                benchmark_path=benchmark_path,
                analysis=analysis,
                generated_assets=generated_assets,
                output_dir=self.output.root_dir,
                enable_code_generation=True  # Enable LLM code generation
            )

            return result

        except Exception as e:
            self.logger.error(f"Game assembly stage failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _save_error_result(self, error_data: Dict[str, Any], filename: str):
        """Persist error result."""
        self.output.save_json(error_data, filename)
