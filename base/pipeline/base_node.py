from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


def _generate_node_id() -> str:
    return uuid.uuid4().hex[:8]


class NodeContext(BaseModel):
    """Base context for node execution. Subclasses define specific fields."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}


class BaseNode(BaseModel, ABC):
    """Abstract base class for DAG nodes."""

    node_id: str = Field(default_factory=_generate_node_id)
    successors: list["BaseNode"] = Field(default_factory=list)
    predecessors: list["BaseNode"] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def add(self, nodes: BaseNode | list[BaseNode]) -> BaseNode | list[BaseNode]:
        """Add successor node(s)."""
        node_list = [nodes] if isinstance(nodes, BaseNode) else nodes
        for node in node_list:
            if node not in self.successors:
                self.successors.append(node)
            if self not in node.predecessors:
                node.predecessors.append(self)
        return nodes

    def __rshift__(self, other: BaseNode | list[BaseNode]) -> BaseNode | list[BaseNode]:
        """Syntactic sugar for a >> b."""
        return self.add(other)

    @abstractmethod
    async def execute(self, ctx: NodeContext) -> None:
        """Execute node logic. Read inputs from ctx and write outputs to ctx."""
