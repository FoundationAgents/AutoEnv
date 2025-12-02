"""
Core abstractions for the visualizer pipeline.

Contains minimal implementations for output layout management, checkpoints,
dependency scheduling, and shared strategy parsing.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from visualizer.core.utils import ensure_dir, save_json, load_json, save_text


# ==============================
# Output management
# ==============================

class OutputManager:
    """Manage output directory structure and common save/load helpers."""

    def __init__(self, benchmark_name: str, config, custom_output_dir: Optional[str] = None):
        self.benchmark_name = benchmark_name
        self.config = config

        if custom_output_dir:
            self.root_dir = Path(custom_output_dir)
        else:
            dir_name = benchmark_name
            if config.use_timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir_name = f"{benchmark_name}_{timestamp}"
            self.root_dir = Path(config.base_dir) / dir_name

        self._create_structure()

    def _create_structure(self):
        ensure_dir(self.root_dir)
        ensure_dir(self.raw_assets_dir)
        ensure_dir(self.refined_assets_dir)
        ensure_dir(self.agent_logs_dir)

    @property
    def raw_assets_dir(self) -> Path:
        return self.root_dir / self.config.raw_assets_dir

    @property
    def refined_assets_dir(self) -> Path:
        return self.root_dir / self.config.refined_assets_dir

    @property
    def agent_logs_dir(self) -> Path:
        return self.root_dir / self.config.agent_logs_dir

    @property
    def analysis_file(self) -> Path:
        return self.root_dir / self.config.analysis_file

    @property
    def strategy_file(self) -> Path:
        return self.root_dir / self.config.strategy_file

    @property
    def manifest_file(self) -> Path:
        return self.root_dir / self.config.manifest_file

    @property
    def pipeline_log(self) -> Path:
        return self.root_dir / self.config.pipeline_log

    def get_asset_path(self, asset_id: str, refined: bool = False) -> Path:
        base = self.refined_assets_dir if refined else self.raw_assets_dir
        return base / f"{asset_id}.png"

    def get_agent_log_path(self, agent_name: str) -> Path:
        return self.agent_logs_dir / f"{agent_name}_agent.log"

    def save_json(self, data: Dict[str, Any], filename: str) -> Path:
        return save_json(self.root_dir / filename, data)

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        return load_json(self.root_dir / filename)

    def save_text(self, content: str, filename: str) -> Path:
        return save_text(self.root_dir / filename, content)

    def __str__(self) -> str:
        return f"OutputManager(root={self.root_dir})"

    def __repr__(self) -> str:
        return self.__str__()


# ==============================
# Checkpoint management
# ==============================

class CheckpointManager:
    """Persist minimal checkpoint data for resume."""

    def __init__(self, output_dir: Path):
        self.checkpoint_file = Path(output_dir) / "checkpoint.json"

    def save(self, stage: str, data: Dict[str, Any], completed_assets: Optional[List[str]] = None):
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "data": data,
            "completed_assets": completed_assets or [],
        }
        save_json(self.checkpoint_file, checkpoint)

    def load(self) -> Optional[Dict[str, Any]]:
        return load_json(self.checkpoint_file)

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def get_completed_assets(self) -> List[str]:
        data = self.load() or {}
        return data.get("completed_assets", [])

    def is_stage_completed(self, stage: str) -> bool:
        """Return True if checkpoint has advanced beyond the given stage."""
        if not self.exists():
            return False
        stage_order = ["analysis", "strategy", "generation", "completed"]
        data = self.load() or {}
        current = data.get("stage")
        if current not in stage_order:
            return False
        return stage_order.index(current) > stage_order.index(stage)

    def summary(self) -> Dict[str, Any]:
        data = self.load()
        if not data:
            return {"exists": False}
        return {
            "exists": True,
            "timestamp": data.get("timestamp"),
            "stage": data.get("stage"),
            "completed_assets_count": len(data.get("completed_assets", [])),
        }


# ==============================
# Dependency scheduling
# ==============================

@dataclass
class AssetNode:
    asset_id: str
    dependencies: List[str]
    priority: int = 0


class DependencyScheduler:
    """DAG-based dependency scheduler with basic concurrency control."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.nodes: Dict[str, AssetNode] = {}
        self.completed: Set[str] = set()
        self.in_progress: Set[str] = set()

    def add_asset(self, asset_id: str, dependencies: Optional[List[str]] = None, priority: int = 10):
        self.nodes[asset_id] = AssetNode(asset_id=asset_id, dependencies=dependencies or [], priority=priority)

    def _validate_dag(self) -> Tuple[bool, Optional[str]]:
        visited: Set[str] = set()
        stack: Set[str] = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            stack.add(node_id)
            for dep in self.nodes.get(node_id, AssetNode(node_id, [])).dependencies:
                if dep not in visited and has_cycle(dep):
                    return True
                if dep in stack:
                    return True
            stack.discard(node_id)
            return False

        for nid in self.nodes:
            if nid not in visited and has_cycle(nid):
                return False, f"Cycle detected involving {nid}"
        return True, None

    def _ready_tasks(self) -> List[str]:
        ready: List[str] = []
        for asset_id, node in self.nodes.items():
            if asset_id in self.completed or asset_id in self.in_progress:
                continue
            if all(dep in self.completed for dep in node.dependencies):
                ready.append(asset_id)
        ready.sort(key=lambda aid: self.nodes[aid].priority, reverse=True)
        return ready

    def validate_dag(self) -> Tuple[bool, Optional[str]]:
        """Public DAG validation helper."""
        return self._validate_dag()

    def get_execution_plan(self) -> List[List[str]]:
        """Return batches of tasks respecting dependencies and priority."""
        plan: List[List[str]] = []
        temp_completed: Set[str] = set()

        while len(temp_completed) < len(self.nodes):
            batch: List[str] = []
            for asset_id, node in self.nodes.items():
                if asset_id in temp_completed:
                    continue
                if all(dep in temp_completed for dep in node.dependencies):
                    batch.append(asset_id)
            if not batch:
                break
            batch.sort(key=lambda aid: self.nodes[aid].priority, reverse=True)
            plan.append(batch)
            temp_completed.update(batch)
        return plan

    async def schedule(
        self,
        task_func: Callable[[str, List[str]], Any],
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute tasks respecting dependencies and concurrency."""
        is_valid, error = self._validate_dag()
        if not is_valid:
            raise ValueError(f"Invalid DAG: {error}")

        results: Dict[str, Any] = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def execute(asset_id: str):
            async with semaphore:
                self.in_progress.add(asset_id)
                if on_progress:
                    on_progress(asset_id, "started")
                try:
                    node = self.nodes[asset_id]
                    result = await task_func(asset_id, node.dependencies)
                    results[asset_id] = result
                    if on_progress:
                        on_progress(asset_id, "completed")
                except Exception as e:  # noqa: BLE001
                    results[asset_id] = {"error": str(e)}
                    if on_progress:
                        on_progress(asset_id, f"failed: {e}")
                finally:
                    self.in_progress.discard(asset_id)
                    self.completed.add(asset_id)

        pending_tasks: Set[asyncio.Task] = set()

        while len(self.completed) < len(self.nodes):
            ready = self._ready_tasks()
            for asset_id in ready:
                if len(self.in_progress) >= self.max_concurrent:
                    break
                pending_tasks.add(asyncio.create_task(execute(asset_id)))

            if pending_tasks:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            else:
                remaining = set(self.nodes.keys()) - self.completed - self.in_progress
                if remaining:
                    raise RuntimeError(f"Scheduling deadlock. Remaining: {remaining}")
                break

        if pending_tasks:
            await asyncio.wait(pending_tasks)

        return results


class StrategyParser:
    """Parse dependency info from strategy JSON."""

    @staticmethod
    def parse_dependencies(strategy: Dict[str, Any]) -> DependencyScheduler:
        scheduler = DependencyScheduler()

        assets = strategy.get("assets", [])
        if any("dependencies" in asset for asset in assets):
            for asset in assets:
                scheduler.add_asset(
                    asset_id=asset["id"],
                    dependencies=asset.get("dependencies", []),
                    priority=asset.get("priority", 10),
                )
            return scheduler

        workflow = strategy.get("generation_workflow", {})
        phases = workflow.get("phases", [])

        if not phases:
            for asset in assets:
                scheduler.add_asset(asset_id=asset["id"], dependencies=[], priority=10)
            return scheduler

        phase_priority = 100
        for phase in phases:
            phase_assets = phase.get("assets", [])
            phase_strategy = phase.get("strategy", {})
            style_anchor = phase_strategy.get("style_anchor")

            for asset_id in phase_assets:
                dependencies: List[str] = []
                if style_anchor:
                    dependencies.append(style_anchor)

                asset_spec = StrategyParser._find_asset_spec(strategy, asset_id)
                if asset_spec and "reference_assets" in asset_spec:
                    dependencies.extend(asset_spec["reference_assets"])

                scheduler.add_asset(asset_id=asset_id, dependencies=dependencies, priority=phase_priority)

            phase_priority -= 10

        return scheduler

    @staticmethod
    def _find_asset_spec(strategy: Dict[str, Any], asset_id: str) -> Optional[Dict[str, Any]]:
        for asset in strategy.get("assets", []):
            if asset.get("id") == asset_id:
                return asset
        return None


__all__ = [
    "OutputManager",
    "CheckpointManager",
    "DependencyScheduler",
    "StrategyParser",
]
