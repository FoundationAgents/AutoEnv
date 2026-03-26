# ============================================================
# SKIN RENDER OUTPUT CONTRACT
# Purpose: Unified render output for text / 2D / 3D environment skins
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

SkinModality = Literal["text", "2d", "3d", "multi"]


@dataclass
class SkinRenderOutput:
    """Normalized output contract for all skin renderers.

    agent_view:
        The content passed into the solver/agent.
    human_view:
        Optional richer view for UI or debugging. Defaults to agent_view.
    modality:
        Main modality used by this render output.
    artifacts:
        File paths / runtime handles (e.g. html entrypoint, model folder).
    metadata:
        Extra renderer metadata (camera settings, legend, etc.).
    """

    agent_view: Any
    human_view: Any | None = None
    modality: SkinModality = "text"
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolved_human_view(self) -> Any:
        if self.human_view is None:
            return self.agent_view
        return self.human_view

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_view": self.agent_view,
            "human_view": self.resolved_human_view(),
            "modality": self.modality,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }
