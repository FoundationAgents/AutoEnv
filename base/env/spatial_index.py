# ============================================================
# SPATIAL INDEX DATA STRUCTURES
# Purpose: Efficient spatial queries using Octree and QuadTree
# ============================================================

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from dataclasses import dataclass, field

from base.env.bounding_box import BoundingBox, BoundingBox2D, BoundingBox3D


class SpatialIndex(ABC):
    """
    空间索引抽象基类。
    
    提供统一的空间查询接口，用于快速查找指定区域内的物体。
    子类实现具体的空间分割算法（QuadTree, Octree 等）。
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回维度 (2 或 3)"""
        pass
    
    @abstractmethod
    def insert(
        self,
        obj_id: str,
        bounds: BoundingBox
    ) -> None:
        """
        插入物体到空间索引。
        
        Args:
            obj_id: 物体唯一标识符
            bounds: 物体的包围盒
        """
        pass
    
    @abstractmethod
    def remove(self, obj_id: str) -> bool:
        """
        从空间索引移除物体。
        
        Args:
            obj_id: 物体唯一标识符
            
        Returns:
            True 如果成功移除
        """
        pass
    
    @abstractmethod
    def update(
        self,
        obj_id: str,
        bounds: BoundingBox
    ) -> None:
        """
        更新物体的包围盒。
        
        Args:
            obj_id: 物体唯一标识符
            bounds: 新的包围盒
        """
        pass
    
    @abstractmethod
    def query_region(
        self,
        bounds: BoundingBox
    ) -> List[str]:
        """
        查询与指定区域相交的所有物体。
        
        Args:
            bounds: 查询区域的包围盒
            
        Returns:
            物体 ID 列表
        """
        pass
    
    @abstractmethod
    def query_point(
        self,
        point: Tuple[float, ...]
    ) -> List[str]:
        """
        查询包含指定点的所有物体。
        
        Args:
            point: 查询点坐标
            
        Returns:
            物体 ID 列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空索引中的所有物体"""
        pass
    
    @property
    @abstractmethod
    def count(self) -> int:
        """返回索引中的物体数量"""
        pass


@dataclass
class QuadTreeNode:
    """四叉树节点"""
    bounds: BoundingBox2D
    objects: Dict[str, BoundingBox2D] = field(default_factory=dict)
    children: Optional[List["QuadTreeNode"]] = None
    depth: int = 0


class QuadTree(SpatialIndex):
    """
    四叉树空间索引，用于 2D 场景。
    
    将 2D 空间递归分割为四个象限，加速区域查询。
    
    Attributes:
        bounds: 整个空间的边界
        max_objects: 节点分裂前的最大物体数
        max_depth: 最大深度
    
    Example:
        >>> tree = QuadTree(
        ...     bounds=BoundingBox2D((0, 0), (1000, 1000)),
        ...     max_objects=10,
        ...     max_depth=8
        ... )
        >>> tree.insert("player", BoundingBox2D((100, 100), (150, 150)))
        >>> tree.insert("enemy", BoundingBox2D((200, 200), (250, 250)))
        >>> results = tree.query_region(BoundingBox2D((0, 0), (300, 300)))
        >>> print(results)  # ['player', 'enemy']
    """
    
    def __init__(
        self,
        bounds: BoundingBox2D,
        max_objects: int = 10,
        max_depth: int = 8
    ):
        self._bounds = bounds
        self._max_objects = max_objects
        self._max_depth = max_depth
        self._root = QuadTreeNode(bounds=bounds, depth=0)
        self._object_nodes: Dict[str, QuadTreeNode] = {}
        self._object_bounds: Dict[str, BoundingBox2D] = {}
    
    @property
    def dimension(self) -> int:
        return 2
    
    @property
    def count(self) -> int:
        return len(self._object_bounds)
    
    def insert(self, obj_id: str, bounds: BoundingBox2D) -> None:
        if not isinstance(bounds, BoundingBox2D):
            raise TypeError("QuadTree requires BoundingBox2D")
        
        self._object_bounds[obj_id] = bounds
        self._insert_to_node(self._root, obj_id, bounds)
    
    def _insert_to_node(
        self,
        node: QuadTreeNode,
        obj_id: str,
        bounds: BoundingBox2D
    ) -> None:
        # 如果有子节点，尝试插入到子节点
        if node.children is not None:
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    self._insert_to_node(child, obj_id, bounds)
            return
        
        # 插入到当前节点
        node.objects[obj_id] = bounds
        self._object_nodes[obj_id] = node
        
        # 检查是否需要分裂
        if (len(node.objects) > self._max_objects and
            node.depth < self._max_depth):
            self._split(node)
    
    def _split(self, node: QuadTreeNode) -> None:
        """将节点分裂为四个子节点"""
        mid_x = (node.bounds.min_point[0] + node.bounds.max_point[0]) / 2
        mid_y = (node.bounds.min_point[1] + node.bounds.max_point[1]) / 2
        min_p = node.bounds.min_point
        max_p = node.bounds.max_point
        
        # 创建四个子节点 (NW, NE, SW, SE)
        node.children = [
            QuadTreeNode(
                bounds=BoundingBox2D((min_p[0], mid_y), (mid_x, max_p[1])),
                depth=node.depth + 1
            ),
            QuadTreeNode(
                bounds=BoundingBox2D((mid_x, mid_y), (max_p[0], max_p[1])),
                depth=node.depth + 1
            ),
            QuadTreeNode(
                bounds=BoundingBox2D((min_p[0], min_p[1]), (mid_x, mid_y)),
                depth=node.depth + 1
            ),
            QuadTreeNode(
                bounds=BoundingBox2D((mid_x, min_p[1]), (max_p[0], mid_y)),
                depth=node.depth + 1
            )
        ]
        
        # 重新分配物体到子节点
        old_objects = node.objects
        node.objects = {}
        
        for obj_id, bounds in old_objects.items():
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    child.objects[obj_id] = bounds
                    self._object_nodes[obj_id] = child
    
    def remove(self, obj_id: str) -> bool:
        if obj_id not in self._object_bounds:
            return False
        
        bounds = self._object_bounds.pop(obj_id)
        self._remove_from_tree(self._root, obj_id, bounds)
        self._object_nodes.pop(obj_id, None)
        return True
    
    def _remove_from_tree(
        self,
        node: QuadTreeNode,
        obj_id: str,
        bounds: BoundingBox2D
    ) -> None:
        if node.children is not None:
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    self._remove_from_tree(child, obj_id, bounds)
        else:
            node.objects.pop(obj_id, None)
    
    def update(self, obj_id: str, bounds: BoundingBox2D) -> None:
        self.remove(obj_id)
        self.insert(obj_id, bounds)
    
    def query_region(self, bounds: BoundingBox) -> List[str]:
        if not isinstance(bounds, BoundingBox2D):
            # 尝试转换
            bounds = BoundingBox2D(
                min_point=(bounds.min_point[0], bounds.min_point[1]),
                max_point=(bounds.max_point[0], bounds.max_point[1])
            )
        
        results: Set[str] = set()
        self._query_node(self._root, bounds, results)
        return list(results)
    
    def _query_node(
        self,
        node: QuadTreeNode,
        query_bounds: BoundingBox2D,
        results: Set[str]
    ) -> None:
        if not node.bounds.intersects_aabb(
            query_bounds.min_point,
            query_bounds.max_point
        ):
            return
        
        if node.children is not None:
            for child in node.children:
                self._query_node(child, query_bounds, results)
        else:
            for obj_id, obj_bounds in node.objects.items():
                if obj_bounds.intersects_aabb(
                    query_bounds.min_point,
                    query_bounds.max_point
                ):
                    results.add(obj_id)
    
    def query_point(self, point: Tuple[float, float]) -> List[str]:
        results: Set[str] = set()
        self._query_point_node(self._root, point, results)
        return list(results)
    
    def _query_point_node(
        self,
        node: QuadTreeNode,
        point: Tuple[float, float],
        results: Set[str]
    ) -> None:
        if not node.bounds.contains_point(point):
            return
        
        if node.children is not None:
            for child in node.children:
                self._query_point_node(child, point, results)
        else:
            for obj_id, obj_bounds in node.objects.items():
                if obj_bounds.contains_point(point):
                    results.add(obj_id)
    
    def clear(self) -> None:
        self._root = QuadTreeNode(bounds=self._bounds, depth=0)
        self._object_nodes.clear()
        self._object_bounds.clear()


@dataclass
class OctreeNode:
    """八叉树节点"""
    bounds: BoundingBox3D
    objects: Dict[str, BoundingBox3D] = field(default_factory=dict)
    children: Optional[List["OctreeNode"]] = None
    depth: int = 0


class Octree(SpatialIndex):
    """
    八叉树空间索引，用于 3D 场景。
    
    将 3D 空间递归分割为八个卦限，加速区域查询。
    
    Attributes:
        bounds: 整个空间的边界
        max_objects: 节点分裂前的最大物体数
        max_depth: 最大深度
    
    Example:
        >>> tree = Octree(
        ...     bounds=BoundingBox3D((0, 0, 0), (100, 100, 100)),
        ...     max_objects=10,
        ...     max_depth=6
        ... )
        >>> tree.insert("sphere", BoundingBox3D((10, 10, 10), (20, 20, 20)))
        >>> results = tree.query_region(BoundingBox3D((0, 0, 0), (50, 50, 50)))
        >>> print(results)  # ['sphere']
    """
    
    def __init__(
        self,
        bounds: BoundingBox3D,
        max_objects: int = 10,
        max_depth: int = 6
    ):
        self._bounds = bounds
        self._max_objects = max_objects
        self._max_depth = max_depth
        self._root = OctreeNode(bounds=bounds, depth=0)
        self._object_nodes: Dict[str, OctreeNode] = {}
        self._object_bounds: Dict[str, BoundingBox3D] = {}
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def count(self) -> int:
        return len(self._object_bounds)
    
    def insert(self, obj_id: str, bounds: BoundingBox3D) -> None:
        if not isinstance(bounds, BoundingBox3D):
            raise TypeError("Octree requires BoundingBox3D")
        
        self._object_bounds[obj_id] = bounds
        self._insert_to_node(self._root, obj_id, bounds)
    
    def _insert_to_node(
        self,
        node: OctreeNode,
        obj_id: str,
        bounds: BoundingBox3D
    ) -> None:
        # 如果有子节点，尝试插入到子节点
        if node.children is not None:
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    self._insert_to_node(child, obj_id, bounds)
            return
        
        # 插入到当前节点
        node.objects[obj_id] = bounds
        self._object_nodes[obj_id] = node
        
        # 检查是否需要分裂
        if (len(node.objects) > self._max_objects and
            node.depth < self._max_depth):
            self._split(node)
    
    def _split(self, node: OctreeNode) -> None:
        """将节点分裂为八个子节点"""
        min_p = node.bounds.min_point
        max_p = node.bounds.max_point
        mid_x = (min_p[0] + max_p[0]) / 2
        mid_y = (min_p[1] + max_p[1]) / 2
        mid_z = (min_p[2] + max_p[2]) / 2
        
        # 创建八个子节点
        node.children = [
            # 下层四个 (z < mid_z)
            OctreeNode(
                bounds=BoundingBox3D((min_p[0], min_p[1], min_p[2]), (mid_x, mid_y, mid_z)),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((mid_x, min_p[1], min_p[2]), (max_p[0], mid_y, mid_z)),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((min_p[0], mid_y, min_p[2]), (mid_x, max_p[1], mid_z)),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((mid_x, mid_y, min_p[2]), (max_p[0], max_p[1], mid_z)),
                depth=node.depth + 1
            ),
            # 上层四个 (z >= mid_z)
            OctreeNode(
                bounds=BoundingBox3D((min_p[0], min_p[1], mid_z), (mid_x, mid_y, max_p[2])),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((mid_x, min_p[1], mid_z), (max_p[0], mid_y, max_p[2])),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((min_p[0], mid_y, mid_z), (mid_x, max_p[1], max_p[2])),
                depth=node.depth + 1
            ),
            OctreeNode(
                bounds=BoundingBox3D((mid_x, mid_y, mid_z), (max_p[0], max_p[1], max_p[2])),
                depth=node.depth + 1
            )
        ]
        
        # 重新分配物体到子节点
        old_objects = node.objects
        node.objects = {}
        
        for obj_id, bounds in old_objects.items():
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    child.objects[obj_id] = bounds
                    self._object_nodes[obj_id] = child
    
    def remove(self, obj_id: str) -> bool:
        if obj_id not in self._object_bounds:
            return False
        
        bounds = self._object_bounds.pop(obj_id)
        self._remove_from_tree(self._root, obj_id, bounds)
        self._object_nodes.pop(obj_id, None)
        return True
    
    def _remove_from_tree(
        self,
        node: OctreeNode,
        obj_id: str,
        bounds: BoundingBox3D
    ) -> None:
        if node.children is not None:
            for child in node.children:
                if child.bounds.intersects_aabb(bounds.min_point, bounds.max_point):
                    self._remove_from_tree(child, obj_id, bounds)
        else:
            node.objects.pop(obj_id, None)
    
    def update(self, obj_id: str, bounds: BoundingBox3D) -> None:
        self.remove(obj_id)
        self.insert(obj_id, bounds)
    
    def query_region(self, bounds: BoundingBox) -> List[str]:
        if not isinstance(bounds, BoundingBox3D):
            # 尝试转换
            min_z = bounds.min_point[2] if len(bounds.min_point) > 2 else 0
            max_z = bounds.max_point[2] if len(bounds.max_point) > 2 else 0
            bounds = BoundingBox3D(
                min_point=(bounds.min_point[0], bounds.min_point[1], min_z),
                max_point=(bounds.max_point[0], bounds.max_point[1], max_z)
            )
        
        results: Set[str] = set()
        self._query_node(self._root, bounds, results)
        return list(results)
    
    def _query_node(
        self,
        node: OctreeNode,
        query_bounds: BoundingBox3D,
        results: Set[str]
    ) -> None:
        if not node.bounds.intersects_aabb(
            query_bounds.min_point,
            query_bounds.max_point
        ):
            return
        
        if node.children is not None:
            for child in node.children:
                self._query_node(child, query_bounds, results)
        else:
            for obj_id, obj_bounds in node.objects.items():
                if obj_bounds.intersects_aabb(
                    query_bounds.min_point,
                    query_bounds.max_point
                ):
                    results.add(obj_id)
    
    def query_point(self, point: Tuple[float, float, float]) -> List[str]:
        results: Set[str] = set()
        self._query_point_node(self._root, point, results)
        return list(results)
    
    def _query_point_node(
        self,
        node: OctreeNode,
        point: Tuple[float, float, float],
        results: Set[str]
    ) -> None:
        if not node.bounds.contains_point(point):
            return
        
        if node.children is not None:
            for child in node.children:
                self._query_point_node(child, point, results)
        else:
            for obj_id, obj_bounds in node.objects.items():
                if obj_bounds.contains_point(point):
                    results.add(obj_id)
    
    def clear(self) -> None:
        self._root = OctreeNode(bounds=self._bounds, depth=0)
        self._object_nodes.clear()
        self._object_bounds.clear()
    
    def query_frustum(
        self,
        frustum_planes: List[Tuple[float, float, float, float]]
    ) -> List[str]:
        """
        查询与视锥体相交的所有物体。
        
        使用视锥体剔除算法，对每个节点进行平面测试。
        
        Args:
            frustum_planes: 视锥体的 6 个平面方程 [(a, b, c, d), ...]
                           法向量指向视锥体内部
        
        Returns:
            物体 ID 列表
        """
        results: Set[str] = set()
        self._query_frustum_node(self._root, frustum_planes, results)
        return list(results)
    
    def _query_frustum_node(
        self,
        node: OctreeNode,
        planes: List[Tuple[float, float, float, float]],
        results: Set[str]
    ) -> None:
        # 检测节点 AABB 是否与视锥体相交
        if not self._aabb_intersects_frustum(node.bounds, planes):
            return
        
        if node.children is not None:
            for child in node.children:
                self._query_frustum_node(child, planes, results)
        else:
            for obj_id, obj_bounds in node.objects.items():
                if self._aabb_intersects_frustum(obj_bounds, planes):
                    results.add(obj_id)
    
    def _aabb_intersects_frustum(
        self,
        bounds: BoundingBox3D,
        planes: List[Tuple[float, float, float, float]]
    ) -> bool:
        """检测 AABB 是否与视锥体相交"""
        min_p = bounds.min_point
        max_p = bounds.max_point
        
        for plane in planes:
            a, b, c, d = plane
            
            # 找到 AABB 上距离平面最远的正方向点 (p-vertex)
            px = max_p[0] if a >= 0 else min_p[0]
            py = max_p[1] if b >= 0 else min_p[1]
            pz = max_p[2] if c >= 0 else min_p[2]
            
            # 如果 p-vertex 在平面外侧，则 AABB 完全在视锥体外
            if a * px + b * py + c * pz + d < 0:
                return False
        
        return True


def create_spatial_index(
    dimension: int,
    bounds: BoundingBox,
    max_objects: int = 10,
    max_depth: int = 8
) -> SpatialIndex:
    """
    创建空间索引的工厂函数。
    
    Args:
        dimension: 维度 (2 或 3)
        bounds: 空间边界
        max_objects: 节点分裂前的最大物体数
        max_depth: 最大深度
        
    Returns:
        QuadTree (2D) 或 Octree (3D)
    """
    if dimension == 2:
        if not isinstance(bounds, BoundingBox2D):
            bounds = BoundingBox2D(
                min_point=(bounds.min_point[0], bounds.min_point[1]),
                max_point=(bounds.max_point[0], bounds.max_point[1])
            )
        return QuadTree(bounds, max_objects, max_depth)
    else:
        if not isinstance(bounds, BoundingBox3D):
            min_z = bounds.min_point[2] if len(bounds.min_point) > 2 else 0
            max_z = bounds.max_point[2] if len(bounds.max_point) > 2 else 0
            bounds = BoundingBox3D(
                min_point=(bounds.min_point[0], bounds.min_point[1], min_z),
                max_point=(bounds.max_point[0], bounds.max_point[1], max_z)
            )
        return Octree(bounds, max_objects, max_depth)

