from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


def _generate_node_id() -> str:
    return uuid.uuid4().hex[:8]


class NodeContext(BaseModel):
    """节点执行上下文基类，子类定义具体字段"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}


class BaseNode(BaseModel, ABC):
    """DAG 节点抽象基类"""

    node_id: str = Field(default_factory=_generate_node_id)
    successors: list["BaseNode"] = Field(default_factory=list)
    predecessors: list["BaseNode"] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def add(self, nodes: BaseNode | list[BaseNode]) -> BaseNode | list[BaseNode]:
        """添加后继节点"""
        node_list = [nodes] if isinstance(nodes, BaseNode) else nodes
        for node in node_list:
            if node not in self.successors:
                self.successors.append(node)
            if self not in node.predecessors:
                node.predecessors.append(self)
        return nodes

    def __rshift__(self, other: BaseNode | list[BaseNode]) -> BaseNode | list[BaseNode]:
        """a >> b 语法糖"""
        return self.add(other)

    @abstractmethod
    async def execute(self, ctx: NodeContext) -> None:
        """执行节点逻辑，从 ctx 读取输入，写入输出"""
