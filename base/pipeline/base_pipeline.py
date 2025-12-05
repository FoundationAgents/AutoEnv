from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from base.pipeline.base_node import BaseNode, NodeContext


class BasePipeline(BaseModel):
    """DAG Pipeline. Executes nodes in parallel by level, sharing NodeContext."""

    root: BaseNode = Field(...)

    model_config = {"arbitrary_types_allowed": True}

    def _collect_nodes(self) -> list[BaseNode]:
        """Collect all nodes from root via DFS."""
        visited: set[str] = set()
        nodes: list[BaseNode] = []

        def dfs(node: BaseNode) -> None:
            if node.node_id in visited:
                return
            visited.add(node.node_id)
            nodes.append(node)
            for s in node.successors:
                dfs(s)

        dfs(self.root)
        return nodes

    async def run(self, ctx: NodeContext | None = None) -> NodeContext:
        """Execute nodes in parallel by level. All nodes share ctx."""
        if ctx is None:
            ctx = NodeContext()

        nodes = self._collect_nodes()
        node_map = {n.node_id: n for n in nodes}
        in_degree = {n.node_id: len(n.predecessors) for n in nodes}
        executed: set[str] = set()

        while len(executed) < len(nodes):
            ready = [nid for nid, deg in in_degree.items() if deg == 0 and nid not in executed]
            if not ready:
                raise ValueError("Cycle detected in DAG")

            await asyncio.gather(*[node_map[nid].execute(ctx) for nid in ready])

            for node_id in ready:
                executed.add(node_id)
                for s in node_map[node_id].successors:
                    in_degree[s.node_id] -= 1

        return ctx

    def visualize(self) -> str:
        """返回 Mermaid 格式 DAG"""
        lines = ["graph TD"]
        for node in self._collect_nodes():
            if not node.predecessors:
                lines.append(f"    {node.node_id}")
            for p in node.predecessors:
                lines.append(f"    {p.node_id} --> {node.node_id}")
        return "\n".join(lines)
