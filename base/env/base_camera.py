# ============================================================
# CAMERA BASE CLASS AND IMPLEMENTATIONS
# Purpose: Abstract base class and concrete implementations for 
#          defining agent observation regions (2D/3D unified interface)
# ============================================================

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Union, Dict
import math

from base.env.view_volume import ViewVolume, ViewVolume2D, ViewVolume3D


# 视野形状类型常量
SHAPE_RECTANGLE = "rectangle"
SHAPE_CIRCLE = "circle"
SHAPE_SECTOR = "sector"
SHAPE_RING = "ring"
SHAPE_POLYGON = "polygon"


class BaseCamera(ABC):
    """
    描述 agent 观察区域的抽象基类。
    
    Camera 负责定义 agent "看哪里"，返回统一的 ViewVolume 对象。
    支持 2D 和 3D 两种模式，提供一致的接口。
    
    子类实现：
    - Camera2D: 2D 多边形视野
    - Camera3D: 3D 视锥体视野
    """
    
    @property
    @abstractmethod
    def view(self) -> ViewVolume:
        """
        返回当前视野区域。
        
        返回 ViewVolume 对象，提供统一的可见性检测接口。
        
        Returns:
            ViewVolume 实例 (ViewVolume2D 或 ViewVolume3D)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回相机维度 (2 或 3)"""
        pass
    
    @abstractmethod
    def update(self, agent_state: Any) -> None:
        """
        根据 agent 状态更新视野。
        
        在每个 step 之后调用，用于跟随 agent 移动或响应其他状态变化。
        
        Args:
            agent_state: agent 的当前状态，格式由具体环境定义
        """
        pass
    
    def reset(self) -> None:
        """
        重置相机状态。
        
        可选方法，在环境 reset 时调用。默认不做任何操作。
        """
        pass


class Camera2D(BaseCamera):
    """
    2D 相机 - 多边形视野区域。
    
    通过中心位置和相对顶点定义多边形视野。
    支持任意形状的多边形，如矩形、扇形、三角形等。
    
    Attributes:
        position: 相机位置 (x, y)
        relative_vertices: 相对于 position 的多边形顶点
    
    See test/test_interaction_ascii.py and test/test_interaction_2d.py for usage examples.
    """
    
    def __init__(
        self,
        position: Tuple[float, float] = (0, 0),
        relative_vertices: Optional[List[Tuple[float, float]]] = None,
        width: float = 100,
        height: float = 100
    ):
        """
        初始化 2D 相机。
        
        Args:
            position: 相机中心位置 (x, y)
            relative_vertices: 相对于 position 的多边形顶点列表
                              如果为 None，则使用 width/height 创建矩形
            width: 矩形视野宽度（仅当 relative_vertices 为 None 时使用）
            height: 矩形视野高度（仅当 relative_vertices 为 None 时使用）
        """
        self._position = (float(position[0]), float(position[1]))
        
        # 形状元数据：用于渲染器进行精确遮罩
        self._shape_info: Dict[str, Any] = {}
        
        if relative_vertices is not None:
            self._relative_vertices = [
                (float(v[0]), float(v[1])) for v in relative_vertices
            ]
            # 自定义顶点时默认为多边形类型
            self._shape_info = {
                "shape_type": SHAPE_POLYGON
            }
        else:
            # 默认矩形视野
            hw, hh = width / 2, height / 2
            self._relative_vertices = [
                (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
            ]
            self._shape_info = {
                "shape_type": SHAPE_RECTANGLE,
                "width": float(width),
                "height": float(height)
            }
        
        self._view: Optional[ViewVolume2D] = None
        self._rebuild_view()
    
    @property
    def dimension(self) -> int:
        return 2
    
    @property
    def position(self) -> Tuple[float, float]:
        """返回相机位置"""
        return self._position
    
    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        """设置相机位置并重建视野"""
        self._position = (float(value[0]), float(value[1]))
        self._rebuild_view()
    
    @property
    def relative_vertices(self) -> List[Tuple[float, float]]:
        """返回相对顶点列表"""
        return self._relative_vertices.copy()
    
    @relative_vertices.setter
    def relative_vertices(self, value: List[Tuple[float, float]]) -> None:
        """设置相对顶点并重建视野"""
        self._relative_vertices = [(float(v[0]), float(v[1])) for v in value]
        self._rebuild_view()
    
    def _rebuild_view(self) -> None:
        """重建视野多边形"""
        # 将相对顶点转换为绝对坐标
        absolute_vertices = [
            (self._position[0] + v[0], self._position[1] + v[1])
            for v in self._relative_vertices
        ]
        # 传递形状信息给 ViewVolume2D
        shape_info = self.get_shape_info() if hasattr(self, '_shape_info') else None
        self._view = ViewVolume2D(absolute_vertices, shape_info=shape_info)
    
    @property
    def view(self) -> ViewVolume2D:
        return self._view
    
    def update(self, agent_state: Any) -> None:
        """
        根据 agent 状态更新相机位置。
        
        Args:
            agent_state: 支持以下格式：
                - dict: 包含 'pos' 或 'position' 键
                - tuple/list: 直接作为 (x, y) 位置
                - 对象: 包含 pos 或 position 属性
        """
        new_pos = self._extract_position(agent_state)
        if new_pos is not None:
            self.position = new_pos
    
    def _extract_position(self, agent_state: Any) -> Optional[Tuple[float, float]]:
        """从 agent_state 提取位置"""
        if agent_state is None:
            return None
        
        if isinstance(agent_state, dict):
            pos = agent_state.get('pos') or agent_state.get('position')
            if pos is not None:
                return (float(pos[0]), float(pos[1]))
        elif isinstance(agent_state, (tuple, list)):
            if len(agent_state) >= 2:
                return (float(agent_state[0]), float(agent_state[1]))
        elif hasattr(agent_state, 'pos'):
            pos = agent_state.pos
            return (float(pos[0]), float(pos[1]))
        elif hasattr(agent_state, 'position'):
            pos = agent_state.position
            return (float(pos[0]), float(pos[1]))
        
        return None
    
    def reset(self) -> None:
        """重置相机到原点"""
        self._position = (0.0, 0.0)
        self._rebuild_view()
    
    def set_rectangular_view(self, width: float, height: float) -> None:
        """设置矩形视野"""
        hw, hh = width / 2, height / 2
        self._shape_info = {
            "shape_type": SHAPE_RECTANGLE,
            "width": float(width),
            "height": float(height)
        }
        self.relative_vertices = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
    
    def set_circular_view(self, radius: float, segments: int = 16) -> None:
        """
        设置圆形视野（近似多边形）。
        
        Args:
            radius: 圆形半径
            segments: 多边形分段数（越多越接近圆形）
        """
        self._shape_info = {
            "shape_type": SHAPE_CIRCLE,
            "radius": float(radius),
            "segments": segments
        }
        vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, y))
        self.relative_vertices = vertices
    
    def set_sector_view(
        self,
        radius: float,
        angle_start: float,
        angle_end: float,
        segments: int = 8
    ) -> None:
        """
        设置扇形视野。
        
        Args:
            radius: 扇形半径
            angle_start: 起始角度（度），0 度为正上方
            angle_end: 结束角度（度）
            segments: 弧线分段数
        """
        self._shape_info = {
            "shape_type": SHAPE_SECTOR,
            "radius": float(radius),
            "angle_start": float(angle_start),
            "angle_end": float(angle_end),
            "segments": segments
        }
        vertices = [(0, 0)]  # 扇形中心
        
        for i in range(segments + 1):
            t = i / segments
            angle = math.radians(angle_start + t * (angle_end - angle_start))
            # 注意：数学坐标系中 0 度是正右方，这里调整为正上方
            x = radius * math.sin(angle)
            y = radius * math.cos(angle)
            vertices.append((x, y))
        
        self.relative_vertices = vertices
    
    def set_ring_view(
        self,
        outer_radius: float,
        inner_radius: float,
        segments: int = 32
    ) -> None:
        """
        设置环形视野（圆环）。
        
        Args:
            outer_radius: 外圆半径
            inner_radius: 内圆半径（盲区）
            segments: 多边形分段数
        """
        if inner_radius >= outer_radius:
            raise ValueError("inner_radius 必须小于 outer_radius")
        if inner_radius < 0:
            raise ValueError("inner_radius 必须为非负数")
        
        self._shape_info = {
            "shape_type": SHAPE_RING,
            "outer_radius": float(outer_radius),
            "inner_radius": float(inner_radius),
            "segments": segments
        }
        
        # 环形需要特殊处理：外圆 + 内圆（逆时针）形成带孔多边形
        # 这里我们用外圆顶点来近似，可见性检测需要额外处理内圆
        # 对于多边形遮罩，渲染器会使用 shape_info 进行精确处理
        vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
            vertices.append((x, y))
        self.relative_vertices = vertices
    
    def get_shape_info(self) -> Dict[str, Any]:
        """
        获取当前视野的形状元数据。
        
        用于渲染器进行精确的形状遮罩裁剪。
        
        Returns:
            包含形状类型和参数的字典，例如：
            - 矩形: {"shape_type": "rectangle", "width": 100, "height": 80}
            - 圆形: {"shape_type": "circle", "radius": 50, "center": (x, y)}
            - 扇形: {"shape_type": "sector", "radius": 50, "angle_start": -30, "angle_end": 30}
            - 环形: {"shape_type": "ring", "outer_radius": 100, "inner_radius": 30}
            - 多边形: {"shape_type": "polygon"}
        """
        # 返回形状信息的副本，并添加当前中心位置
        info = self._shape_info.copy()
        info["center"] = self._position
        return info


class Camera3D(BaseCamera):
    """
    3D 相机 - 视锥体视野区域。
    
    通过位置、方向和视锥体参数定义 3D 视野。
    支持透视投影，自动计算视锥体的 8 个顶点和 6 个平面。
    
    Attributes:
        position: 相机位置 (x, y, z)
        direction: 观察方向 (dx, dy, dz)
        fov: 垂直视野角度（度）
        aspect: 宽高比
        near: 近裁剪面距离
        far: 远裁剪面距离
    
    See test/test_interaction_3d.py for usage examples.
    """
    
    def __init__(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        direction: Tuple[float, float, float] = (0, 0, -1),
        fov: float = 60.0,
        aspect: float = 16/9,
        near: float = 0.1,
        far: float = 100.0,
        up: Tuple[float, float, float] = (0, 1, 0)
    ):
        """
        初始化 3D 相机。
        
        Args:
            position: 相机位置 (x, y, z)
            direction: 观察方向（会被归一化）
            fov: 垂直视野角度（度）
            aspect: 宽高比 (width / height)
            near: 近裁剪面距离
            far: 远裁剪面距离
            up: 上方向向量
        """
        self._position = tuple(float(v) for v in position)
        self._direction = self._normalize(direction)
        self._up = tuple(float(v) for v in up)
        self._fov = float(fov)
        self._aspect = float(aspect)
        self._near = float(near)
        self._far = float(far)
        
        self._view: Optional[ViewVolume3D] = None
        self._rebuild_view()
    
    def _normalize(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """归一化向量"""
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length > 1e-10:
            return (v[0]/length, v[1]/length, v[2]/length)
        return (0, 0, -1)
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def position(self) -> Tuple[float, float, float]:
        return self._position
    
    @position.setter
    def position(self, value: Tuple[float, float, float]) -> None:
        self._position = tuple(float(v) for v in value)
        self._rebuild_view()
    
    @property
    def direction(self) -> Tuple[float, float, float]:
        return self._direction
    
    @direction.setter
    def direction(self, value: Tuple[float, float, float]) -> None:
        self._direction = self._normalize(value)
        self._rebuild_view()
    
    @property
    def fov(self) -> float:
        return self._fov
    
    @fov.setter
    def fov(self, value: float) -> None:
        self._fov = float(value)
        self._rebuild_view()
    
    @property
    def aspect(self) -> float:
        return self._aspect
    
    @aspect.setter
    def aspect(self, value: float) -> None:
        self._aspect = float(value)
        self._rebuild_view()
    
    @property
    def near(self) -> float:
        return self._near
    
    @near.setter
    def near(self, value: float) -> None:
        self._near = float(value)
        self._rebuild_view()
    
    @property
    def far(self) -> float:
        return self._far
    
    @far.setter
    def far(self, value: float) -> None:
        self._far = float(value)
        self._rebuild_view()
    
    @property
    def up(self) -> Tuple[float, float, float]:
        return self._up
    
    @up.setter
    def up(self, value: Tuple[float, float, float]) -> None:
        self._up = tuple(float(v) for v in value)
        self._rebuild_view()
    
    def _rebuild_view(self) -> None:
        """重建视锥体"""
        self._view = ViewVolume3D(
            position=self._position,
            direction=self._direction,
            fov=self._fov,
            aspect=self._aspect,
            near=self._near,
            far=self._far,
            up=self._up
        )
    
    @property
    def view(self) -> ViewVolume3D:
        return self._view
    
    def update(self, agent_state: Any) -> None:
        """
        根据 agent 状态更新相机。
        
        Args:
            agent_state: 支持以下格式：
                - dict: 包含 'pos'/'position' 和可选的 'direction'/'rotation' 键
                - tuple/list: 直接作为 (x, y, z) 位置
                - 对象: 包含 pos/position 和 direction/rotation 属性
        """
        if agent_state is None:
            return
        
        new_pos = None
        new_dir = None
        
        if isinstance(agent_state, dict):
            # 提取位置
            pos = agent_state.get('pos') or agent_state.get('position')
            if pos is not None and len(pos) >= 3:
                new_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            
            # 提取方向
            direction = agent_state.get('direction')
            if direction is not None and len(direction) >= 3:
                new_dir = (float(direction[0]), float(direction[1]), float(direction[2]))
            
            # 或者从旋转角度计算方向
            rotation = agent_state.get('rotation')
            if rotation is not None and new_dir is None:
                new_dir = self._rotation_to_direction(rotation)
                
        elif isinstance(agent_state, (tuple, list)):
            if len(agent_state) >= 3:
                new_pos = (float(agent_state[0]), float(agent_state[1]), float(agent_state[2]))
                
        elif hasattr(agent_state, 'pos') or hasattr(agent_state, 'position'):
            pos = getattr(agent_state, 'pos', None) or getattr(agent_state, 'position', None)
            if pos is not None and len(pos) >= 3:
                new_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            
            direction = getattr(agent_state, 'direction', None)
            if direction is not None and len(direction) >= 3:
                new_dir = (float(direction[0]), float(direction[1]), float(direction[2]))
        
        # 更新相机参数
        need_rebuild = False
        if new_pos is not None:
            self._position = new_pos
            need_rebuild = True
        if new_dir is not None:
            self._direction = self._normalize(new_dir)
            need_rebuild = True
        
        if need_rebuild:
            self._rebuild_view()
    
    def _rotation_to_direction(self, rotation: Any) -> Optional[Tuple[float, float, float]]:
        """从旋转参数计算观察方向"""
        if isinstance(rotation, (tuple, list)) and len(rotation) >= 2:
            # 假设是 (yaw, pitch) 或 (yaw, pitch, roll)
            yaw = math.radians(float(rotation[0]))
            pitch = math.radians(float(rotation[1]))
            
            # 从欧拉角计算方向向量
            dx = math.cos(pitch) * math.sin(yaw)
            dy = math.sin(pitch)
            dz = -math.cos(pitch) * math.cos(yaw)
            
            return (dx, dy, dz)
        return None
    
    def reset(self) -> None:
        """重置相机到默认状态"""
        self._position = (0.0, 0.0, 0.0)
        self._direction = (0.0, 0.0, -1.0)
        self._rebuild_view()
    
    def look_at(self, target: Tuple[float, float, float]) -> None:
        """
        将相机朝向目标点。
        
        Args:
            target: 目标点坐标 (x, y, z)
        """
        dx = target[0] - self._position[0]
        dy = target[1] - self._position[1]
        dz = target[2] - self._position[2]
        self.direction = (dx, dy, dz)
    
    def get_render_params(self) -> dict:
        """
        获取传递给渲染器的参数字典。
        
        Returns:
            包含 camera_position, camera_target, fov 等的字典
        """
        # 计算目标点
        target = (
            self._position[0] + self._direction[0] * 10,
            self._position[1] + self._direction[1] * 10,
            self._position[2] + self._direction[2] * 10
        )
        return {
            "camera_position": list(self._position),
            "camera_target": list(target),
            "fov": self._fov,
            "near": self._near,
            "far": self._far,
            "aspect": self._aspect
        }


def create_camera(
    dimension: int,
    **kwargs
) -> BaseCamera:
    """
    创建相机的工厂函数。
    
    Args:
        dimension: 维度 (2 或 3)
        **kwargs: 传递给具体相机类的参数
        
    Returns:
        Camera2D 或 Camera3D 实例
    """
    if dimension == 2:
        return Camera2D(**kwargs)
    else:
        return Camera3D(**kwargs)
