"""
AutoEnv Pipeline
DAG-based environment generation pipeline
"""

from pathlib import Path

from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
from autoenv.pipeline.visual.nodes import (
    AnalyzerNode,
    AssetGeneratorNode,
    AssemblyNode,
    AutoEnvContext,
    BackgroundRemovalNode,
    StrategistNode,
)
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_pipeline import BasePipeline


class VisualPipeline(BasePipeline):
    """
    Visualization pipeline.

    DAG structure:
        Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Assembly
    """

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        image_model: str,
        llm_name: str = "claude-sonnet-4-5",
    ) -> "VisualPipeline":
        """
        Factory method: Create default visualization pipeline.

        Args:
            image_model: Image generation model name (required)
            llm_name: LLM name for agents

        Usage:
            pipeline = VisualPipeline.create_default(
                image_model="gemini-2.5-flash-image",
                llm_name="gemini-2.5-flash"
            )
            ctx = await pipeline.run()
        """
        analyzer_agent = MiniSWEAutoEnvAgent(
            llm_name=llm_name,
            mode="yolo",
            step_limit=40,
            cost_limit=8.0,
            environment_type="local",
            cwd=str(Path.cwd()),
        )

        strategist_agent = MiniSWEAutoEnvAgent(
            llm_name=llm_name,
            mode="yolo",
            step_limit=40,
            cost_limit=8.0,
            environment_type="local",
            cwd=str(Path.cwd()),
        )

        assembly_agent = MiniSWEAutoEnvAgent(
            llm_name=llm_name,
            mode="yolo",
            step_limit=40,
            cost_limit=8.0,
            environment_type="local",
            cwd=str(Path.cwd()),
        )

        image_llm = AsyncLLM(image_model)

        analyzer = AnalyzerNode(agent=analyzer_agent)
        strategist = StrategistNode(agent=strategist_agent)
        asset_generator = AssetGeneratorNode(image_llm=image_llm)
        bg_removal = BackgroundRemovalNode()
        assembly = AssemblyNode(agent=assembly_agent)

        analyzer >> strategist >> asset_generator >> bg_removal >> assembly

        return cls(root=analyzer)

    async def run(
        self,
        benchmark_path: Path | None = None,
        instruction: str | None = None,
        output_dir: Path = Path("."),
    ) -> AutoEnvContext:
        """Execute pipeline."""
        ctx = AutoEnvContext(
            benchmark_path=benchmark_path,
            instruction=instruction,
            output_dir=output_dir,
        )
        return await super().run(ctx)