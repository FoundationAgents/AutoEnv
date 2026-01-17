"""
AutoEnv Pipeline
DAG-based environment generation pipeline
"""

from pathlib import Path
from typing import Optional

from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
from autoenv.pipeline.visual.nodes import (
    AnalyzerNode,
    AssetGeneratorNode,
    AssemblyNode,
    AutoEnvContext,
    BackgroundRemovalNode,
    Image3DConvertNode,
    StrategistNode,
    ThreeJSAssemblyNode,
)
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_pipeline import BasePipeline


class VisualPipeline(BasePipeline):
    """
    Visualization pipeline with optional 3D conversion.

    DAG structure (2D mode):
        Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Assembly (pygame)
    
    DAG structure (3D mode):
        Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Image3DConvert → ThreeJSAssembly (three.js)
    """

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        image_model: str,
        llm_name: str = "claude-sonnet-4-5",
        dimension: str = "2d",
        meshy_api_key: str = "",
        meshy_base_url: str = "https://api.meshy.ai/v1",
        max_3d_assets: int = 4,
        target_polycount: int = 10000,
    ) -> "VisualPipeline":
        """
        Factory method: Create visualization pipeline with optional 3D support.

        Args:
            image_model: Image generation model name (required)
            llm_name: LLM name for agents
            dimension: "2d" or "3d" - determines if 3D conversion is enabled
            meshy_api_key: Meshy API key for 3D conversion (required if dimension="3d")
            meshy_base_url: Meshy API base URL
            max_3d_assets: Maximum number of assets to convert to 3D
            target_polycount: Target polygon count for 3D models

        Usage:
            # 2D pipeline
            pipeline = VisualPipeline.create_default(
                image_model="gemini-2.5-flash-image",
                dimension="2d"
            )
            
            # 3D pipeline
            pipeline = VisualPipeline.create_default(
                image_model="gemini-2.5-flash-image",
                dimension="3d",
                meshy_api_key="your-key"
            )
        """
        # Create agents
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

        threejs_agent = MiniSWEAutoEnvAgent(
            llm_name=llm_name,
            mode="yolo",
            step_limit=60,
            cost_limit=12.0,
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

        # Create core nodes
        analyzer = AnalyzerNode(agent=analyzer_agent)
        strategist = StrategistNode(agent=strategist_agent)
        asset_generator = AssetGeneratorNode(image_llm=image_llm)
        bg_removal = BackgroundRemovalNode()

        # Build pipeline DAG based on dimension
        if dimension == "3d":
            # 3D pipeline: add 3D conversion + three.js assembly
            image_to_3d = Image3DConvertNode(
                meshy_api_key=meshy_api_key,
                meshy_base_url=meshy_base_url,
                max_assets_to_convert=max_3d_assets,
                target_polycount=target_polycount,
            )
            threejs_assembly = ThreeJSAssemblyNode(agent=threejs_agent)
            analyzer >> strategist >> asset_generator >> bg_removal >> image_to_3d >> threejs_assembly
        else:
            # 2D pygame pipeline
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