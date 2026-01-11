# ============================================================
# BOUNDING BOX UTILITIES
# Purpose: AABB and OBB collision detection utilities
# ============================================================

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class BoundingBox(ABC):
    """
    包围盒抽象基类。
    
    提供 2D 和 3D 包围盒的统一接口。
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回维度 (2 或 3)"""
        pass
    
    @property
    @abstractmethod
    def min_point(self) -> Tuple[float, ...]:
        """返回最小点坐标"""
        pass
    
    @property
    @abstractmethod
    def max_point(self) -> Tuple[float, ...]:
        """返回最大点坐标"""
        pass
    
    @property
    @abstractmethod
    def center(self) -> Tuple[float, ...]:
        """返回中心点坐标"""
        pass
    
    @property
    @abstractmethod
    def size(self) -> Tuple[float, ...]:
        """返回尺寸"""
        pass
    
    @property
    @abstractmethod
    def half_extents(self) -> Tuple[float, ...]:
        """返回半尺寸"""
        pass
    
    @abstractmethod
    def contains_point(self, point: Tuple[float, ...]) -> bool:
        """检测点是否在包围盒内"""
        pass
    
    @abstractmethod
    def intersects_aabb(
        self,
        min_point: Tuple[float, ...],
        max_point: Tuple[float, ...]
    ) -> bool:
        """检测是否与另一个 AABB 相交"""
        pass
    
    @abstractmethod
    def expand(self, amount: float) -> "BoundingBox":
        """扩展包围盒"""
        pass


@dataclass
class BoundingBox2D(BoundingBox):
    """
    2D 轴对齐包围盒 (AABB)。
    
    Attributes:
        min_point: 最小点 (x_min, y_min)
        max_point: 最大点 (x_max, y_max)
    
    Example:
        >>> bbox = BoundingBox2D((0, 0), (100, 100))
        >>> bbox.contains_point((50, 50))
        True
        >>> bbox.center
        (50.0, 50.0)
    """
    
    _min_point: Tuple[float, float]
    _max_point: Tuple[float, float]
    
    def __init__(
        self,
        min_point: Tuple[float, float],
        max_point: Tuple[float, float]
    ):
        self._min_point = (float(min_point[0]), float(min_point[1]))
        self._max_point = (float(max_point[0]), float(max_point[1]))
    
    @property
    def dimension(self) -> int:
        return 2
    
    @property
    def min_point(self) -> Tuple[float, float]:
        return self._min_point
    
    @property
    def max_point(self) -> Tuple[float, float]:
        return self._max_point
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self._min_point[0] + self._max_point[0]) / 2,
            (self._min_point[1] + self._max_point[1]) / 2
        )
    
    @property
    def size(self) -> Tuple[float, float]:
        return (
            self._max_point[0] - self._min_point[0],
            self._max_point[1] - self._min_point[1]
        )
    
    @property
    def half_extents(self) -> Tuple[float, float]:
        s = self.size
        return (s[0] / 2, s[1] / 2)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        x, y = point[0], point[1]
        return (self._min_point[0] <= x <= self._max_point[0] and
                self._min_point[1] <= y <= self._max_point[1])
    
    def intersects_aabb(
        self,
        min_point: Tuple[float, float],
        max_point: Tuple[float, float]
    ) -> bool:
        return not (
            self._max_point[0] < min_point[0] or
            self._min_point[0] > max_point[0] or
            self._max_point[1] < min_point[1] or
            self._min_point[1] > max_point[1]
        )
    
    def expand(self, amount: float) -> "BoundingBox2D":
        return BoundingBox2D(
            min_point=(self._min_point[0] - amount, self._min_point[1] - amount),
            max_point=(self._max_point[0] + amount, self._max_point[1] + amount)
        )
    
    def union(self, other: "BoundingBox2D") -> "BoundingBox2D":
        """返回两个包围盒的并集"""
        return BoundingBox2D(
            min_point=(
                min(self._min_point[0], other._min_point[0]),
                min(self._min_point[1], other._min_point[1])
            ),
            max_point=(
                max(self._max_point[0], other._max_point[0]),
                max(self._max_point[1], other._max_point[1])
            )
        )
    
    @staticmethod
    def from_center_size(
        center: Tuple[float, float],
        size: Tuple[float, float]
    ) -> "BoundingBox2D":
        """从中心点和尺寸创建包围盒"""
        half_w, half_h = size[0] / 2, size[1] / 2
        return BoundingBox2D(
            min_point=(center[0] - half_w, center[1] - half_h),
            max_point=(center[0] + half_w, center[1] + half_h)
        )


@dataclass
class BoundingBox3D(BoundingBox):
    """
    3D 轴对齐包围盒 (AABB)。
    
    Attributes:
        min_point: 最小点 (x_min, y_min, z_min)
        max_point: 最大点 (x_max, y_max, z_max)
    
    Example:
        >>> bbox = BoundingBox3D((0, 0, 0), (10, 10, 10))
        >>> bbox.contains_point((5, 5, 5))
        True
        >>> bbox.center
        (5.0, 5.0, 5.0)
    """
    
    _min_point: Tuple[float, float, float]
    _max_point: Tuple[float, float, float]
    
    def __init__(
        self,
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float]
    ):
        self._min_point = (
            float(min_point[0]),
            float(min_point[1]),
            float(min_point[2])
        )
        self._max_point = (
            float(max_point[0]),
            float(max_point[1]),
            float(max_point[2])
        )
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def min_point(self) -> Tuple[float, float, float]:
        return self._min_point
    
    @property
    def max_point(self) -> Tuple[float, float, float]:
        return self._max_point
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self._min_point[0] + self._max_point[0]) / 2,
            (self._min_point[1] + self._max_point[1]) / 2,
            (self._min_point[2] + self._max_point[2]) / 2
        )
    
    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self._max_point[0] - self._min_point[0],
            self._max_point[1] - self._min_point[1],
            self._max_point[2] - self._min_point[2]
        )
    
    @property
    def half_extents(self) -> Tuple[float, float, float]:
        s = self.size
        return (s[0] / 2, s[1] / 2, s[2] / 2)
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        x, y, z = point[0], point[1], point[2]
        return (self._min_point[0] <= x <= self._max_point[0] and
                self._min_point[1] <= y <= self._max_point[1] and
                self._min_point[2] <= z <= self._max_point[2])
    
    def intersects_aabb(
        self,
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float]
    ) -> bool:
        return not (
            self._max_point[0] < min_point[0] or
            self._min_point[0] > max_point[0] or
            self._max_point[1] < min_point[1] or
            self._min_point[1] > max_point[1] or
            self._max_point[2] < min_point[2] or
            self._min_point[2] > max_point[2]
        )
    
    def expand(self, amount: float) -> "BoundingBox3D":
        return BoundingBox3D(
            min_point=(
                self._min_point[0] - amount,
                self._min_point[1] - amount,
                self._min_point[2] - amount
            ),
            max_point=(
                self._max_point[0] + amount,
                self._max_point[1] + amount,
                self._max_point[2] + amount
            )
        )
    
    def union(self, other: "BoundingBox3D") -> "BoundingBox3D":
        """返回两个包围盒的并集"""
        return BoundingBox3D(
            min_point=(
                min(self._min_point[0], other._min_point[0]),
                min(self._min_point[1], other._min_point[1]),
                min(self._min_point[2], other._min_point[2])
            ),
            max_point=(
                max(self._max_point[0], other._max_point[0]),
                max(self._max_point[1], other._max_point[1]),
                max(self._max_point[2], other._max_point[2])
            )
        )
    
    @staticmethod
    def from_center_size(
        center: Tuple[float, float, float],
        size: Tuple[float, float, float]
    ) -> "BoundingBox3D":
        """从中心点和尺寸创建包围盒"""
        half = (size[0] / 2, size[1] / 2, size[2] / 2)
        return BoundingBox3D(
            min_point=(center[0] - half[0], center[1] - half[1], center[2] - half[2]),
            max_point=(center[0] + half[0], center[1] + half[1], center[2] + half[2])
        )
    
    def get_vertices(self) -> List[Tuple[float, float, float]]:
        """返回包围盒的 8 个顶点"""
        min_p = self._min_point
        max_p = self._max_point
        return [
            (min_p[0], min_p[1], min_p[2]),
            (max_p[0], min_p[1], min_p[2]),
            (max_p[0], max_p[1], min_p[2]),
            (min_p[0], max_p[1], min_p[2]),
            (min_p[0], min_p[1], max_p[2]),
            (max_p[0], min_p[1], max_p[2]),
            (max_p[0], max_p[1], max_p[2]),
            (min_p[0], max_p[1], max_p[2])
        ]


# ============================================================
# OBB (Oriented Bounding Box) 碰撞检测工具函数
# ============================================================

def obb_2d_intersects_obb(
    center1: Tuple[float, float],
    half_extents1: Tuple[float, float],
    angle1: float,
    center2: Tuple[float, float],
    half_extents2: Tuple[float, float],
    angle2: float
) -> bool:
    """
    检测两个 2D OBB 是否相交。
    
    使用分离轴定理 (SAT)。
    
    Args:
        center1: 第一个 OBB 的中心
        half_extents1: 第一个 OBB 的半尺寸 (hx, hy)
        angle1: 第一个 OBB 的旋转角度（弧度）
        center2: 第二个 OBB 的中心
        half_extents2: 第二个 OBB 的半尺寸 (hx, hy)
        angle2: 第二个 OBB 的旋转角度（弧度）
        
    Returns:
        True 如果两个 OBB 相交
    """
    # 计算两个 OBB 的轴向量
    cos1, sin1 = math.cos(angle1), math.sin(angle1)
    cos2, sin2 = math.cos(angle2), math.sin(angle2)
    
    axes1 = [(cos1, sin1), (-sin1, cos1)]
    axes2 = [(cos2, sin2), (-sin2, cos2)]
    
    # 中心向量
    d = (center2[0] - center1[0], center2[1] - center1[1])
    
    # 检测每个轴
    for axis in axes1 + axes2:
        # 投影半径
        r1 = abs(half_extents1[0] * (axis[0] * axes1[0][0] + axis[1] * axes1[0][1])) + \
             abs(half_extents1[1] * (axis[0] * axes1[1][0] + axis[1] * axes1[1][1]))
        r2 = abs(half_extents2[0] * (axis[0] * axes2[0][0] + axis[1] * axes2[0][1])) + \
             abs(half_extents2[1] * (axis[0] * axes2[1][0] + axis[1] * axes2[1][1]))
        
        # 中心距离在轴上的投影
        dist = abs(d[0] * axis[0] + d[1] * axis[1])
        
        if dist > r1 + r2:
            return False
    
    return True


def obb_3d_intersects_obb(
    center1: Tuple[float, float, float],
    half_extents1: Tuple[float, float, float],
    rotation1: Tuple[float, float, float],
    center2: Tuple[float, float, float],
    half_extents2: Tuple[float, float, float],
    rotation2: Tuple[float, float, float]
) -> bool:
    """
    检测两个 3D OBB 是否相交。
    
    使用分离轴定理 (SAT)，需要检测 15 个潜在分离轴。
    
    Args:
        center1: 第一个 OBB 的中心
        half_extents1: 第一个 OBB 的半尺寸 (hx, hy, hz)
        rotation1: 第一个 OBB 的欧拉角（度）
        center2: 第二个 OBB 的中心
        half_extents2: 第二个 OBB 的半尺寸 (hx, hy, hz)
        rotation2: 第二个 OBB 的欧拉角（度）
        
    Returns:
        True 如果两个 OBB 相交
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("OBB 3D collision detection requires numpy")
    
    c1 = np.array(center1, dtype=np.float64)
    c2 = np.array(center2, dtype=np.float64)
    h1 = np.array(half_extents1, dtype=np.float64)
    h2 = np.array(half_extents2, dtype=np.float64)
    
    R1 = _euler_to_matrix(rotation1)
    R2 = _euler_to_matrix(rotation2)
    
    # 中心向量
    d = c2 - c1
    
    # OBB 1 的轴
    a1 = R1[:, 0]
    a2 = R1[:, 1]
    a3 = R1[:, 2]
    
    # OBB 2 的轴
    b1 = R2[:, 0]
    b2 = R2[:, 1]
    b3 = R2[:, 2]
    
    # 15 个潜在分离轴
    axes = [
        a1, a2, a3,  # OBB1 的 3 个轴
        b1, b2, b3,  # OBB2 的 3 个轴
        np.cross(a1, b1), np.cross(a1, b2), np.cross(a1, b3),  # 叉积
        np.cross(a2, b1), np.cross(a2, b2), np.cross(a2, b3),
        np.cross(a3, b1), np.cross(a3, b2), np.cross(a3, b3)
    ]
    
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        axis = axis / length
        
        # 投影半径
        r1 = abs(h1[0] * np.dot(a1, axis)) + \
             abs(h1[1] * np.dot(a2, axis)) + \
             abs(h1[2] * np.dot(a3, axis))
        r2 = abs(h2[0] * np.dot(b1, axis)) + \
             abs(h2[1] * np.dot(b2, axis)) + \
             abs(h2[2] * np.dot(b3, axis))
        
        # 中心距离在轴上的投影
        dist = abs(np.dot(d, axis))
        
        if dist > r1 + r2:
            return False
    
    return True


def _euler_to_matrix(rotation: Tuple[float, float, float]) -> "np.ndarray":
    """欧拉角转旋转矩阵 (XYZ 顺序，度数)"""
    rx, ry, rz = [math.radians(a) for a in rotation]
    
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    return np.array([
        [cy * cz, -cy * sz, sy],
        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
    ])


def point_in_obb_2d(
    point: Tuple[float, float],
    center: Tuple[float, float],
    half_extents: Tuple[float, float],
    angle: float
) -> bool:
    """检测点是否在 2D OBB 内"""
    # 将点转换到 OBB 局部坐标系
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a
    
    return (abs(local_x) <= half_extents[0] and
            abs(local_y) <= half_extents[1])


def point_in_obb_3d(
    point: Tuple[float, float, float],
    center: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    rotation: Tuple[float, float, float]
) -> bool:
    """检测点是否在 3D OBB 内"""
    if not NUMPY_AVAILABLE:
        raise ImportError("point_in_obb_3d requires numpy")
    
    p = np.array(point, dtype=np.float64)
    c = np.array(center, dtype=np.float64)
    h = np.array(half_extents, dtype=np.float64)
    
    R = _euler_to_matrix(rotation)
    
    # 将点转换到 OBB 局部坐标系
    local_p = R.T @ (p - c)
    
    return (abs(local_p[0]) <= h[0] and
            abs(local_p[1]) <= h[1] and
            abs(local_p[2]) <= h[2])


def aabb_from_points(
    points: List[Tuple[float, ...]],
    dimension: int = 3
) -> Union[BoundingBox2D, BoundingBox3D]:
    """
    从点集创建 AABB。
    
    Args:
        points: 点坐标列表
        dimension: 维度 (2 或 3)
        
    Returns:
        BoundingBox2D 或 BoundingBox3D
    """
    if not points:
        if dimension == 2:
            return BoundingBox2D((0, 0), (0, 0))
        else:
            return BoundingBox3D((0, 0, 0), (0, 0, 0))
    
    if dimension == 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return BoundingBox2D(
            min_point=(min(xs), min(ys)),
            max_point=(max(xs), max(ys))
        )
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] if len(p) > 2 else 0 for p in points]
        return BoundingBox3D(
            min_point=(min(xs), min(ys), min(zs)),
            max_point=(max(xs), max(ys), max(zs))
        )

