# ============================================================
# VIEW VOLUME ABSTRACTION
# Purpose: Unified 2D/3D view volume interface for visibility detection
# ============================================================

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, TYPE_CHECKING, Dict, Any
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    ShapelyPolygon = None
    ShapelyPoint = None

if TYPE_CHECKING:
    from base.env.bounding_box import BoundingBox, BoundingBox2D, BoundingBox3D


class ViewVolume(ABC):
    """
    统一的视野体积抽象基类。
    
    提供 2D 和 3D 视野的统一接口，用于可见性检测。
    子类需要实现具体的几何计算逻辑。
    
    Attributes:
        dimension: 视野维度 (2 或 3)
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回视野维度 (2 或 3)"""
        pass
    
    @property
    @abstractmethod
    def vertices(self) -> List[Tuple[float, ...]]:
        """
        返回视野边界顶点列表。
        
        - 2D: 多边形顶点列表，按顺序连接形成闭合区域
        - 3D: 8 个视锥体顶点 (近平面 4 个 + 远平面 4 个)
        
        Returns:
            顶点坐标列表
        """
        pass
    
    @property
    @abstractmethod
    def planes(self) -> List[Tuple[float, ...]]:
        """
        返回视野边界平面方程列表。
        
        每个平面表示为 (a, b, c, d)，满足 ax + by + cz + d = 0
        法向量指向视野内部。
        
        - 2D: 每条边的法线方程 (a, b, d)，满足 ax + by + d = 0
        - 3D: 6 个视锥体平面 (左、右、上、下、近、远)
        
        Returns:
            平面方程列表
        """
        pass
    
    @property
    @abstractmethod
    def bounds(self) -> "BoundingBox":
        """
        返回轴对齐包围盒 (AABB)。
        
        用于快速剔除明显不在视野内的物体。
        
        Returns:
            BoundingBox 实例
        """
        pass
    
    @abstractmethod
    def contains_point(self, point: Tuple[float, ...]) -> bool:
        """
        检测点是否在视野内。
        
        Args:
            point: 点坐标 (x, y) 或 (x, y, z)
            
        Returns:
            True 如果点在视野内
        """
        pass
    
    @abstractmethod
    def intersects_aabb(
        self,
        min_point: Tuple[float, ...],
        max_point: Tuple[float, ...]
    ) -> bool:
        """
        检测轴对齐包围盒 (AABB) 是否与视野相交。
        
        Args:
            min_point: AABB 最小点坐标
            max_point: AABB 最大点坐标
            
        Returns:
            True 如果 AABB 与视野相交
        """
        pass
    
    @abstractmethod
    def intersects_obb(
        self,
        center: Tuple[float, ...],
        half_extents: Tuple[float, ...],
        rotation: Optional[Tuple[float, ...]] = None
    ) -> bool:
        """
        检测有向包围盒 (OBB) 是否与视野相交。
        
        Args:
            center: OBB 中心点坐标
            half_extents: OBB 半尺寸 (半宽, 半高) 或 (半宽, 半高, 半深)
            rotation: 旋转参数
                - 2D: (angle,) 弧度制旋转角
                - 3D: (rx, ry, rz) 欧拉角或四元数
                
        Returns:
            True 如果 OBB 与视野相交
        """
        pass


class ViewVolume2D(ViewVolume):
    """
    2D 多边形视野区域。
    
    通过顶点列表定义一个闭合多边形作为可视区域。
    顶点按顺时针或逆时针顺序排列。
    
    Example:
        >>> # 矩形视野
        >>> view = ViewVolume2D([
        ...     (0, 0), (100, 0), (100, 100), (0, 100)
        ... ])
        >>> view.contains_point((50, 50))
        True
        >>> 
        >>> # 三角形视野
        >>> view = ViewVolume2D([
        ...     (50, 0), (100, 100), (0, 100)
        ... ])
    """
    
    def __init__(
        self,
        polygon_vertices: List[Tuple[float, float]],
        shape_info: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 2D 视野。
        
        Args:
            polygon_vertices: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
                             按顺序连接形成闭合多边形
            shape_info: 可选的形状元数据，用于渲染器精确遮罩
                        例如: {"shape_type": "circle", "radius": 100, "center": (x, y)}
        """
        if len(polygon_vertices) < 3:
            raise ValueError("多边形至少需要 3 个顶点")
        
        self._vertices = [tuple(v) for v in polygon_vertices]
        self._polygon: Optional[ShapelyPolygon] = None
        self._planes: Optional[List[Tuple[float, float, float]]] = None
        self._bounds: Optional["BoundingBox2D"] = None
        self._shape_info: Optional[Dict[str, Any]] = shape_info
        
        # 预计算
        self._compute_planes()
        self._compute_bounds()
        
        # 如果 shapely 可用，创建多边形对象
        if SHAPELY_AVAILABLE:
            self._polygon = ShapelyPolygon(self._vertices)
    
    @property
    def dimension(self) -> int:
        return 2
    
    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return self._vertices.copy()
    
    @property
    def planes(self) -> List[Tuple[float, float, float]]:
        return self._planes.copy()
    
    @property
    def bounds(self) -> "BoundingBox2D":
        return self._bounds
    
    @property
    def shape_info(self) -> Optional[Dict[str, Any]]:
        """
        返回形状元数据。
        
        用于渲染器进行精确的形状遮罩裁剪。
        如果没有设置，则返回 None（使用多边形顶点）。
        
        Returns:
            形状信息字典或 None
        """
        return self._shape_info
    
    @shape_info.setter
    def shape_info(self, value: Optional[Dict[str, Any]]) -> None:
        """设置形状元数据"""
        self._shape_info = value
    
    def _compute_planes(self) -> None:
        """计算每条边的法线方程"""
        self._planes = []
        n = len(self._vertices)
        
        for i in range(n):
            p1 = self._vertices[i]
            p2 = self._vertices[(i + 1) % n]
            
            # 边向量
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 内向法线 (指向多边形内部)
            # 对于逆时针顶点顺序，左手法则得到内向法线
            # 这里假设逆时针，如果是顺时针则取反
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-10:
                nx = -dy / length
                ny = dx / length
            else:
                nx, ny = 0, 1
            
            # 平面方程: nx * x + ny * y + d = 0
            # d = -(nx * p1.x + ny * p1.y)
            d = -(nx * p1[0] + ny * p1[1])
            
            self._planes.append((nx, ny, d))
    
    def _compute_bounds(self) -> None:
        """计算轴对齐包围盒"""
        from base.env.bounding_box import BoundingBox2D
        
        xs = [v[0] for v in self._vertices]
        ys = [v[1] for v in self._vertices]
        
        self._bounds = BoundingBox2D(
            min_point=(min(xs), min(ys)),
            max_point=(max(xs), max(ys))
        )
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """检测点是否在多边形内"""
        if len(point) < 2:
            return False
        
        x, y = point[0], point[1]
        
        # 快速 AABB 剔除
        if not self._bounds.contains_point(point):
            return False
        
        # 使用 shapely 进行精确检测
        if SHAPELY_AVAILABLE and self._polygon is not None:
            return self._polygon.contains(ShapelyPoint(x, y))
        
        # 降级：射线法
        return self._ray_casting_contains(x, y)
    
    def _ray_casting_contains(self, x: float, y: float) -> bool:
        """射线法检测点是否在多边形内"""
        n = len(self._vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self._vertices[i]
            xj, yj = self._vertices[j]
            
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def intersects_aabb(
        self,
        min_point: Tuple[float, float],
        max_point: Tuple[float, float]
    ) -> bool:
        """检测 AABB 是否与多边形相交"""
        # 快速 AABB vs AABB 检测
        if not self._bounds.intersects_aabb(min_point, max_point):
            return False
        
        # 检测 AABB 的任意角点是否在多边形内
        corners = [
            (min_point[0], min_point[1]),
            (max_point[0], min_point[1]),
            (max_point[0], max_point[1]),
            (min_point[0], max_point[1])
        ]
        
        for corner in corners:
            if self.contains_point(corner):
                return True
        
        # 检测多边形的任意顶点是否在 AABB 内
        for v in self._vertices:
            if (min_point[0] <= v[0] <= max_point[0] and
                min_point[1] <= v[1] <= max_point[1]):
                return True
        
        # 检测边是否相交
        aabb_edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0])
        ]
        
        n = len(self._vertices)
        for i in range(n):
            p1 = self._vertices[i]
            p2 = self._vertices[(i + 1) % n]
            
            for e1, e2 in aabb_edges:
                if self._segments_intersect(p1, p2, e1, e2):
                    return True
        
        return False
    
    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> bool:
        """检测两条线段是否相交"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and
                ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    def intersects_obb(
        self,
        center: Tuple[float, float],
        half_extents: Tuple[float, float],
        rotation: Optional[Tuple[float]] = None
    ) -> bool:
        """检测 OBB 是否与多边形相交"""
        angle = rotation[0] if rotation else 0.0
        
        # 计算 OBB 的 4 个顶点
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        hx, hy = half_extents[0], half_extents[1]
        cx, cy = center[0], center[1]
        
        # 局部坐标系中的角点
        local_corners = [
            (-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)
        ]
        
        # 旋转到世界坐标系
        obb_corners = []
        for lx, ly in local_corners:
            wx = cx + lx * cos_a - ly * sin_a
            wy = cy + lx * sin_a + ly * cos_a
            obb_corners.append((wx, wy))
        
        # 使用 shapely 进行精确检测
        if SHAPELY_AVAILABLE and self._polygon is not None:
            obb_poly = ShapelyPolygon(obb_corners)
            return self._polygon.intersects(obb_poly)
        
        # 降级：检测角点和边相交
        # 检测 OBB 角点是否在多边形内
        for corner in obb_corners:
            if self.contains_point(corner):
                return True
        
        # 检测多边形顶点是否在 OBB 内
        for v in self._vertices:
            if self._point_in_obb(v, center, half_extents, angle):
                return True
        
        return False
    
    def _point_in_obb(
        self,
        point: Tuple[float, float],
        center: Tuple[float, float],
        half_extents: Tuple[float, float],
        angle: float
    ) -> bool:
        """检测点是否在 OBB 内"""
        # 将点转换到 OBB 局部坐标系
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        return (abs(local_x) <= half_extents[0] and
                abs(local_y) <= half_extents[1])


class ViewVolume3D(ViewVolume):
    """
    3D 视锥体视野区域。
    
    通过相机参数定义视锥体，支持透视投影。
    内部自动计算 8 个顶点和 6 个平面。
    
    Attributes:
        position: 相机位置 (x, y, z)
        direction: 观察方向 (dx, dy, dz)，会被归一化
        up: 上方向向量 (ux, uy, uz)
        fov: 垂直视野角度 (度)
        aspect: 宽高比 (width / height)
        near: 近裁剪面距离
        far: 远裁剪面距离
    
    Example:
        >>> view = ViewVolume3D(
        ...     position=(0, 5, 10),
        ...     direction=(0, 0, -1),
        ...     fov=60,
        ...     aspect=16/9,
        ...     near=0.1,
        ...     far=100
        ... )
        >>> view.contains_point((0, 5, 5))
        True
    """
    
    def __init__(
        self,
        position: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        fov: float = 60.0,
        aspect: float = 16/9,
        near: float = 0.1,
        far: float = 100.0,
        up: Tuple[float, float, float] = (0, 1, 0)
    ):
        """
        初始化 3D 视锥体。
        
        Args:
            position: 相机位置
            direction: 观察方向（会被归一化）
            fov: 垂直视野角度（度）
            aspect: 宽高比
            near: 近裁剪面距离
            far: 远裁剪面距离
            up: 上方向向量
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("ViewVolume3D requires numpy. Install with: pip install numpy")
        
        self.position = np.array(position, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        
        # 归一化方向向量
        direction = np.array(direction, dtype=np.float64)
        self.direction = direction / np.linalg.norm(direction)
        
        # 计算相机坐标系
        self._compute_camera_basis()
        
        # 预计算顶点和平面
        self._vertices: List[Tuple[float, float, float]] = []
        self._planes: List[Tuple[float, float, float, float]] = []
        self._bounds: Optional["BoundingBox3D"] = None
        
        self._compute_frustum()
    
    def _compute_camera_basis(self) -> None:
        """计算相机坐标系的三个轴"""
        # 前向 (z)
        self._forward = self.direction
        
        # 右向 (x) = forward × up
        self._right = np.cross(self._forward, self.up)
        right_len = np.linalg.norm(self._right)
        if right_len > 1e-10:
            self._right /= right_len
        else:
            # 处理 forward 与 up 平行的情况
            self._right = np.array([1, 0, 0], dtype=np.float64)
        
        # 上向 (y) = right × forward
        self._up = np.cross(self._right, self._forward)
        self._up /= np.linalg.norm(self._up)
    
    def _compute_frustum(self) -> None:
        """计算视锥体的 8 个顶点和 6 个平面"""
        # 计算近/远平面的半尺寸
        fov_rad = math.radians(self.fov)
        near_h = self.near * math.tan(fov_rad / 2)
        near_w = near_h * self.aspect
        far_h = self.far * math.tan(fov_rad / 2)
        far_w = far_h * self.aspect
        
        # 近平面中心和远平面中心
        near_center = self.position + self._forward * self.near
        far_center = self.position + self._forward * self.far
        
        # 计算 8 个顶点
        # 近平面: ntl, ntr, nbr, nbl (top-left, top-right, bottom-right, bottom-left)
        ntl = near_center + self._up * near_h - self._right * near_w
        ntr = near_center + self._up * near_h + self._right * near_w
        nbr = near_center - self._up * near_h + self._right * near_w
        nbl = near_center - self._up * near_h - self._right * near_w
        
        # 远平面: ftl, ftr, fbr, fbl
        ftl = far_center + self._up * far_h - self._right * far_w
        ftr = far_center + self._up * far_h + self._right * far_w
        fbr = far_center - self._up * far_h + self._right * far_w
        fbl = far_center - self._up * far_h - self._right * far_w
        
        self._vertices = [
            tuple(ntl), tuple(ntr), tuple(nbr), tuple(nbl),
            tuple(ftl), tuple(ftr), tuple(fbr), tuple(fbl)
        ]
        
        # 存储为 numpy 数组用于计算
        self._vertices_np = np.array([
            ntl, ntr, nbr, nbl,
            ftl, ftr, fbr, fbl
        ])
        
        # 计算 6 个平面 (法向量指向内部)
        self._planes = []
        
        # 近平面: 法向量 = forward
        self._planes.append(self._make_plane(self._forward, near_center))
        
        # 远平面: 法向量 = -forward
        self._planes.append(self._make_plane(-self._forward, far_center))
        
        # 左平面: 通过 position, nbl, ntl
        left_normal = self._compute_plane_normal(self.position, fbl, ftl)
        self._planes.append(self._make_plane(left_normal, self.position))
        
        # 右平面: 通过 position, ntr, nbr
        right_normal = self._compute_plane_normal(self.position, ftr, fbr)
        self._planes.append(self._make_plane(right_normal, self.position))
        
        # 上平面: 通过 position, ntl, ntr
        top_normal = self._compute_plane_normal(self.position, ftl, ftr)
        self._planes.append(self._make_plane(top_normal, self.position))
        
        # 下平面: 通过 position, nbr, nbl
        bottom_normal = self._compute_plane_normal(self.position, fbr, fbl)
        self._planes.append(self._make_plane(bottom_normal, self.position))
        
        # 计算包围盒
        self._compute_bounds()
    
    def _make_plane(
        self,
        normal: np.ndarray,
        point: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """创建平面方程 (a, b, c, d)，满足 ax + by + cz + d = 0"""
        n = normal / np.linalg.norm(normal)
        d = -np.dot(n, point)
        return (float(n[0]), float(n[1]), float(n[2]), float(d))
    
    def _compute_plane_normal(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> np.ndarray:
        """计算三点确定的平面的法向量（指向内部）"""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)
    
    def _compute_bounds(self) -> None:
        """计算轴对齐包围盒"""
        from base.env.bounding_box import BoundingBox3D
        
        min_pt = np.min(self._vertices_np, axis=0)
        max_pt = np.max(self._vertices_np, axis=0)
        
        self._bounds = BoundingBox3D(
            min_point=tuple(min_pt),
            max_point=tuple(max_pt)
        )
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def vertices(self) -> List[Tuple[float, float, float]]:
        return self._vertices.copy()
    
    @property
    def planes(self) -> List[Tuple[float, float, float, float]]:
        return self._planes.copy()
    
    @property
    def bounds(self) -> "BoundingBox3D":
        return self._bounds
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """检测点是否在视锥体内"""
        if len(point) < 3:
            return False
        
        p = np.array(point, dtype=np.float64)
        
        # 快速 AABB 剔除
        if not self._bounds.contains_point(point):
            return False
        
        # 检测点在所有平面的内侧
        for plane in self._planes:
            a, b, c, d = plane
            # 如果点在平面外侧（符号距离 < 0），则不在视锥体内
            distance = a * p[0] + b * p[1] + c * p[2] + d
            if distance < 0:
                return False
        
        return True
    
    def intersects_aabb(
        self,
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float]
    ) -> bool:
        """检测 AABB 是否与视锥体相交"""
        # 快速 AABB vs AABB 检测
        if not self._bounds.intersects_aabb(min_point, max_point):
            return False
        
        min_pt = np.array(min_point, dtype=np.float64)
        max_pt = np.array(max_point, dtype=np.float64)
        
        # 对每个平面进行检测
        for plane in self._planes:
            n = np.array(plane[:3])
            d = plane[3]
            
            # 找到 AABB 上距离平面最近的正方向点 (p-vertex)
            p_vertex = np.where(n >= 0, max_pt, min_pt)
            
            # 如果 p-vertex 在平面外侧，则 AABB 完全在视锥体外
            if np.dot(n, p_vertex) + d < 0:
                return False
        
        return True
    
    def intersects_obb(
        self,
        center: Tuple[float, float, float],
        half_extents: Tuple[float, float, float],
        rotation: Optional[Tuple[float, float, float]] = None
    ) -> bool:
        """
        检测 OBB 是否与视锥体相交。
        
        使用分离轴定理 (SAT) 进行精确检测。
        
        Args:
            center: OBB 中心点
            half_extents: OBB 半尺寸 (hx, hy, hz)
            rotation: 欧拉角 (rx, ry, rz) 度数，如果为 None 则视为 AABB
        """
        c = np.array(center, dtype=np.float64)
        h = np.array(half_extents, dtype=np.float64)
        
        # 计算 OBB 的旋转矩阵
        if rotation is not None:
            R = self._euler_to_matrix(rotation)
        else:
            R = np.eye(3)
        
        # OBB 的三个轴向量
        obb_axes = R.T  # 每行是一个轴
        
        # 对每个视锥体平面进行检测
        for plane in self._planes:
            n = np.array(plane[:3])
            d = plane[3]
            
            # 计算 OBB 在平面法向量方向上的投影半径
            r = 0.0
            for i in range(3):
                r += h[i] * abs(np.dot(n, obb_axes[i]))
            
            # 计算 OBB 中心到平面的符号距离
            s = np.dot(n, c) + d
            
            # 如果 OBB 完全在平面外侧
            if s < -r:
                return False
        
        return True
    
    def _euler_to_matrix(self, rotation: Tuple[float, float, float]) -> np.ndarray:
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
    
    def get_camera_params(self) -> dict:
        """
        获取相机参数字典，用于传递给渲染器。
        
        Returns:
            包含 camera_position, camera_target, fov 等的字典
        """
        target = self.position + self.direction * 10  # 目标点
        return {
            "camera_position": list(self.position),
            "camera_target": list(target),
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
            "aspect": self.aspect
        }

