"""
Strategist Agent
Use MiniSWEAgent to devise a visualization strategy.
"""

import json
from pathlib import Path
from typing import Dict, Any

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
from visualizer.agents.prompt_mm import strategy_prompt


class StrategistAgent:
    """Visualization strategy agent."""

    def __init__(
        self,
        llm_name: str = "claude-sonnet-4-5",
        step_limit: int = 40,
        cost_limit: float = 10.0,
        timeout: int = 90
    ):
        """
        Args:
            llm_name: LLM model name
            step_limit: Max agent steps
            cost_limit: Max spend (USD)
            timeout: Timeout in seconds
        """
        self.llm_name = llm_name
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.timeout = timeout

    async def devise_strategy(
        self,
        analysis: Dict[str, Any],
        analysis_file: Path,  # temp input file
        output_file: Path,
        log_file: Path
    ) -> Dict[str, Any]:
        """
        Devise a visualization strategy.

        Args:
            analysis: Analysis result
            analysis_file: Temp analysis file path (for the agent to read)
            output_file: Strategy JSON path
            log_file: Agent log path

        Returns:
            Strategy dict, or a dict containing 'error' on failure.
        """

        # Persist analysis to a temp file
        analysis_file.parent.mkdir(parents=True, exist_ok=True)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Build task
        task = self._build_task(analysis_file, output_file)

        # Create agent
        agent = MiniSWEAutoEnvAgent(
            llm_name=self.llm_name,
            mode="yolo",
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
            environment_type="local",
            cwd=str(Path.cwd()),
            timeout=self.timeout
        )

        # Run agent
        result_str = await agent.run(task=task)

        # Save log
        self._save_log(task, result_str, log_file)

        # Clean up temp file
        if analysis_file.exists():
            analysis_file.unlink()

        # Parse result
        try:
            result = eval(result_str)

            # Consider success if exit_status completed or file exists
            if result.get('exit_status') == 'completed' or output_file.exists():
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        strategy = json.load(f)
                    return strategy
                else:
                    return {'error': 'output_file_not_found', 'details': result}
            else:
                return {'error': 'agent_failed', 'details': result}

        except Exception as e:
            return {'error': 'parse_failed', 'exception': str(e), 'raw_result': result_str}

    def _build_task(self, analysis_file: Path, output_file: Path) -> str:
        """Build the prompt for the agent."""

        return strategy_prompt(analysis_file, output_file)

    def _save_log(self, task: str, result: str, log_file: Path):
        """Persist agent run log."""
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Strategist Agent Log\n")
            f.write("=" * 70 + "\n\n")
            f.write("TASK:\n")
            f.write(task + "\n\n")
            f.write("=" * 70 + "\n")
            f.write("RESULT:\n")
            f.write("=" * 70 + "\n")
            f.write(result + "\n")
