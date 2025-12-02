"""
Pipeline Configuration Management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration."""
    analyzer: str = "claude-sonnet-4-5"
    strategist: str = "claude-sonnet-4-5"
    image_generator: str = "gemini-2.5-flash-image-preview"
    use_agent: bool = True  # Use MiniSWEAgent


@dataclass
class GenerationConfig:
    """Generation configuration."""
    target_asset_size: int = 256  # Target asset size (px); keep at least 256
    enable_refinement: bool = True  # Enable refinement
    max_refinement_iterations: int = 5  # Max refinement iterations

    # Parallel generation settings
    enable_parallel: bool = True  # Enable parallel generation (DAG-based)
    max_concurrent: int = 5  # Max concurrency (consider API limits)
    use_dependency_graph: bool = True  # Use dependency graph scheduling

    # Agent settings
    agent_step_limit: int = 30  # Max agent steps
    agent_cost_limit: float = 5.0  # Max spend (USD)
    agent_timeout: int = 60  # Timeout (seconds)


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: Path = field(default_factory=lambda: Path("visualizer/output"))
    use_timestamp: bool = True  # Append timestamp to output dir

    # Subdirectory names
    raw_assets_dir: str = "raw_generated"
    refined_assets_dir: str = "refined_assets"
    agent_logs_dir: str = "agent_logs"

    # Output file names
    analysis_file: str = "01_analysis.json"
    strategy_file: str = "02_strategy.json"
    manifest_file: str = "03_manifest.json"
    pipeline_log: str = "pipeline.log"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Runtime flags
    verbose: bool = True  # Verbose output
    save_intermediate: bool = True  # Save intermediate artifacts

    # Input mode
    mode: str = "benchmark"  # "benchmark" or "instruction"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from a dict."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            generation=GenerationConfig(**config_dict.get('generation', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            verbose=config_dict.get('verbose', True),
            save_intermediate=config_dict.get('save_intermediate', True),
            mode=config_dict.get('mode', 'benchmark')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            'model': {
                'analyzer': self.model.analyzer,
                'strategist': self.model.strategist,
                'image_generator': self.model.image_generator,
                'use_agent': self.model.use_agent
            },
            'generation': {
                'target_asset_size': self.generation.target_asset_size,
                'enable_refinement': self.generation.enable_refinement,
                'max_refinement_iterations': self.generation.max_refinement_iterations,
                'enable_parallel': self.generation.enable_parallel,
                'max_concurrent': self.generation.max_concurrent,
                'use_dependency_graph': self.generation.use_dependency_graph,
                'agent_step_limit': self.generation.agent_step_limit,
                'agent_cost_limit': self.generation.agent_cost_limit,
                'agent_timeout': self.generation.agent_timeout
            },
            'output': {
                'base_dir': str(self.output.base_dir),
                'use_timestamp': self.output.use_timestamp,
                'raw_assets_dir': self.output.raw_assets_dir,
                'refined_assets_dir': self.output.refined_assets_dir,
                'agent_logs_dir': self.output.agent_logs_dir
            },
            'verbose': self.verbose,
            'save_intermediate': self.save_intermediate,
            'mode': self.mode
        }


# Preset configurations
class PresetConfigs:
    """Preset configurations."""

    @staticmethod
    def fast() -> PipelineConfig:
        """Fast mode - no refinement, lighter models."""
        config = PipelineConfig()
        config.model.use_agent = False
        config.generation.enable_refinement = False
        return config

    @staticmethod
    def high_quality() -> PipelineConfig:
        """High-quality mode - enable all optimizations."""
        config = PipelineConfig()
        config.model.use_agent = True
        config.generation.enable_refinement = True
        config.generation.max_refinement_iterations = 5
        return config

    @staticmethod
    def debug() -> PipelineConfig:
        """Debug mode - verbose logging, keep all intermediates."""
        config = PipelineConfig()
        config.verbose = True
        config.save_intermediate = True
        config.generation.agent_step_limit = 20  # Fewer steps to speed debugging
        return config
