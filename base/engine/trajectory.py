import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    step_index: int = Field(...)
    obs: Any = Field(..., description="Skinned observation provided to the agent")
    action: Dict[str, Any] = Field(...)
    thinking: Optional[str] = Field(None, description="Agent's reasoning/thought for this step")
    result: Any = Field(None, description="last_action_result from env info")
    reward: float = Field(0)
    events: List[Any] = Field(default_factory=list)
    parse_error: Optional[str] = Field(None, description="If action parsing failed, the error/warning message")


class Trajectory(BaseModel):
    world_id: str = Field(...)
    agent_name: str = Field(...)
    run_id: Optional[str] = Field(None)  # Deprecated, keeping for compatibility
    metadata: Dict[str, Any] = Field(default_factory=dict)
    steps: List[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = Field(0)
    finished: bool = Field(False)


class TrajectoryCollector:
    """Collects per-run trajectories of (obs, action, result, reward, events).

    Usage:
        collector = TrajectoryCollector()
        collector.start_run({"agent_name": "solver", "world_id": 1})
        collector.record_step(...)
        traj = collector.end_run(summary={"total_reward": 10})
    """

    def __init__(self, save_dir: Optional[str] = None, save_jsonl: bool = False) -> None:
        self.save_dir = save_dir or os.path.join("workspace/logs", "trajectories")
        self.save_jsonl = save_jsonl
        self._current: Optional[Trajectory] = None

    def start_run(self, metadata: Dict[str, Any]) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        world_id = metadata.get("world_id") or "unknown"
        agent_name = metadata.get("agent_name") or "agent"
        self._current = Trajectory(
            world_id=world_id,
            agent_name=agent_name,
            run_id=world_id,  # For backward compatibility
            metadata={k: v for k, v in metadata.items() if k not in {"world_id", "agent_name"}},
        )

    def record_step(
        self,
        *,
        step_index: int,
        obs: Any,
        action: Dict[str, Any],
        thinking: Optional[str] = None,
        result: Any,
        reward: float,
        events: List[Any],
        parse_error: Optional[str] = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("TrajectoryCollector.start_run must be called before record_step")
        self._current.steps.append(
            TrajectoryStep(
                step_index=step_index,
                obs=obs,
                action=action,
                thinking=thinking,
                result=result,
                reward=reward,
                events=events or [],
                parse_error=parse_error,
            )
        )

    def end_run(self, summary: Optional[Dict[str, Any]] = None) -> Trajectory:
        if self._current is None:
            raise RuntimeError("TrajectoryCollector.start_run must be called before end_run")
        if summary:
            self._current.total_reward = float(summary.get("total_reward", self._current.total_reward))
            self._current.finished = True
            # Attach any other summary fields to metadata to avoid schema churn
            for key, value in summary.items():
                if key not in {"total_reward"}:
                    self._current.metadata[key] = value

        # Persist
        try:
            if self.save_jsonl:
                path = os.path.join(self.save_dir, f"{self._current.world_id}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    header = {
                        "world_id": self._current.world_id,
                        "agent_name": self._current.agent_name,
                        "metadata": self._current.metadata,
                    }
                    f.write(json.dumps({"type": "header", **header}, ensure_ascii=False) + "\n")
                    for step in self._current.steps:
                        f.write(json.dumps({"type": "step", **step.model_dump()}, ensure_ascii=False) + "\n")
                    f.write(
                        json.dumps(
                            {
                                "type": "footer",
                                "total_reward": self._current.total_reward,
                                "finished": self._current.finished,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                path = os.path.join(self.save_dir, f"{self._current.world_id}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._current.model_dump(), f, ensure_ascii=False, indent=2)
            # Record path into metadata for reference
            self._current.metadata["file_path"] = path
        except Exception:
            # Best-effort persistence; avoid breaking the run
            pass

        result = self._current
        self._current = None
        return result

    def get_current(self) -> Optional[Trajectory]:
        return self._current
