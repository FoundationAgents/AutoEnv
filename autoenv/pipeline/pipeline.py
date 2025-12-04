"""
AutoEnv Pipeline
基于 DAG 的环境生成处理流水线
"""

from pathlib import Path

from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
from autoenv.pipeline.nodes import (
    AnalyzerNode,
    AssetGeneratorNode,
    AssemblyNode,
    AutoEnvContext,
    StrategistNode,
)
from base.engine.async_llm import AsyncLLM
from base.pipeline.base_pipeline import BasePipeline


class AutoEnvPipeline(BasePipeline):
    """
    可视化处理流水线

    DAG 结构:
        Analyzer → Strategist → AssetGenerator → Assembly
    """

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create_default(
        cls,
        image_model: str,
        llm_name: str = "claude-sonnet-4-5",
    ) -> "AutoEnvPipeline":
        """
        工厂方法：创建默认的可视化流水线

        Args:
            image_model: 图像生成模型名称（必填）
            llm_name: Agent 使用的 LLM 名称

        Usage:
            pipeline = AutoEnvPipeline.create_default(
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

        image_llm = AsyncLLM(config=image_model)

        analyzer = AnalyzerNode(agent=analyzer_agent)
        strategist = StrategistNode(agent=strategist_agent)
        asset_generator = AssetGeneratorNode(image_llm=image_llm)
        assembly = AssemblyNode(agent=assembly_agent)

        analyzer >> strategist >> asset_generator >> assembly

        return cls(root=analyzer)

    async def run(
        self,
        benchmark_path: Path | None = None,
        instruction: str | None = None,
        output_dir: Path = Path("."),
    ) -> AutoEnvContext:
        """执行流水线"""
        ctx = AutoEnvContext(
            benchmark_path=benchmark_path,
            instruction=instruction,
            output_dir=output_dir,
        )
        return await super().run(ctx)
