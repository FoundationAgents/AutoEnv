# ============================================================
# SEMANTIC VIEW DATA MODELS (Pydantic)
# Purpose: Standard data format between Env and Renderer
# ============================================================

from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional, Union, Tuple, Literal


class Material3D(BaseModel):
    """
    PBR (Physically Based Rendering) 材质配置。
    
    用于 3D 渲染时定义物体的外观属性，基于物理的渲染材质模型。
    
    Attributes:
        base_color: RGBA 基础颜色，范围 [0, 1]
        metallic: 金属度，0 = 非金属，1 = 完全金属
        roughness: 粗糙度，0 = 光滑镜面，1 = 完全粗糙
        emissive: 自发光颜色 RGB，范围 [0, 1]
        texture_path: 可选的漫反射纹理贴图路径
        normal_map_path: 可选的法线贴图路径
        alpha_mode: 透明度模式
    
    Example:
        >>> # 金属材质
        >>> metal = Material3D(
        ...     base_color=(0.8, 0.8, 0.9, 1.0),
        ...     metallic=1.0,
        ...     roughness=0.2
        ... )
        >>> 
        >>> # 发光材质
        >>> glow = Material3D(
        ...     base_color=(0.2, 0.8, 0.2, 1.0),
        ...     emissive=(0.0, 1.0, 0.0)
        ... )
    """
    
    base_color: Tuple[float, float, float, float] = Field(
        default=(0.5, 0.5, 0.5, 1.0),
        description="RGBA 基础颜色，范围 [0, 1]"
    )
    metallic: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="金属度，0 = 非金属，1 = 完全金属"
    )
    roughness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="粗糙度，0 = 光滑镜面，1 = 完全粗糙"
    )
    emissive: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="自发光颜色 RGB，范围 [0, 1]"
    )
    texture_path: Optional[str] = Field(
        default=None,
        description="漫反射纹理贴图路径"
    )
    normal_map_path: Optional[str] = Field(
        default=None,
        description="法线贴图路径"
    )
    alpha_mode: Literal["opaque", "blend", "mask"] = Field(
        default="opaque",
        description="透明度模式: opaque(不透明), blend(混合), mask(遮罩)"
    )


class LightConfig(BaseModel):
    """
    3D 场景光源配置。
    
    支持多种光源类型：环境光、方向光、点光源、聚光灯。
    
    Attributes:
        type: 光源类型
        color: RGB 颜色，范围 [0, 1]
        intensity: 光照强度
        position: 光源位置（点光源、聚光灯使用）
        direction: 光照方向（方向光、聚光灯使用）
        inner_cone_angle: 聚光灯内锥角（度）
        outer_cone_angle: 聚光灯外锥角（度）
    
    Example:
        >>> # 方向光（类似太阳光）
        >>> sun = LightConfig(
        ...     type="directional",
        ...     color=(1.0, 0.95, 0.9),
        ...     intensity=1.0,
        ...     direction=(-1, -1, -1)
        ... )
        >>> 
        >>> # 点光源
        >>> lamp = LightConfig(
        ...     type="point",
        ...     color=(1.0, 0.8, 0.6),
        ...     intensity=2.0,
        ...     position=(5, 3, 5)
        ... )
    """
    
    type: Literal["ambient", "directional", "point", "spot"] = Field(
        default="directional",
        description="光源类型"
    )
    color: Tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="RGB 颜色，范围 [0, 1]"
    )
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        description="光照强度"
    )
    position: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="光源位置（点光源、聚光灯使用）"
    )
    direction: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="光照方向（方向光、聚光灯使用）"
    )
    inner_cone_angle: float = Field(
        default=30.0,
        description="聚光灯内锥角（度）"
    )
    outer_cone_angle: float = Field(
        default=45.0,
        description="聚光灯外锥角（度）"
    )


class Transform(BaseModel):
    """
    物体 3D 变形属性。
    
    2D 渲染时使用 rotation_z 进行平面旋转，rotation_x/y 产生透视效果。
    3D 渲染时使用完整的欧拉角旋转。
    
    Attributes:
        rotation_x: 绕 X 轴旋转角度（度），2D 中产生上下倾斜透视
        rotation_y: 绕 Y 轴旋转角度（度），2D 中产生左右倾斜透视
        rotation_z: 绕 Z 轴旋转角度（度），2D 中为平面内旋转
        scale_x: X 轴缩放倍数
        scale_y: Y 轴缩放倍数
        scale_z: Z 轴缩放倍数（仅 3D 渲染使用）
    
    Example:
        >>> transform = Transform(rotation_z=45, scale_x=2.0)
        >>> # 物体旋转 45 度，水平方向放大 2 倍
    """
    
    # 旋转（欧拉角，单位：度）
    rotation_x: float = Field(default=0.0, description="绕 X 轴旋转角度（度）")
    rotation_y: float = Field(default=0.0, description="绕 Y 轴旋转角度（度）")
    rotation_z: float = Field(default=0.0, description="绕 Z 轴旋转角度（度），2D 主旋转")
    
    # 缩放（各轴独立）
    scale_x: float = Field(default=1.0, ge=0.0, description="X 轴缩放倍数")
    scale_y: float = Field(default=1.0, ge=0.0, description="Y 轴缩放倍数")
    scale_z: float = Field(default=1.0, ge=0.0, description="Z 轴缩放倍数（3D 专用）")


class ObjectData(BaseModel):
    """
    可见物体的标准化结构。
    
    定义了渲染一个物体所需的全部信息，包括位置、尺寸、变形和层级。
    支持 2D 和 3D 渲染的所有必要属性。
    
    Attributes:
        id: 物体 ID，用于匹配素材文件（如 "player" 对应 "player.png"）
        pos: 坐标位置，支持 2D (x, y) 或 3D (x, y, z)
        size: 像素尺寸 (width, height)，None 时使用素材原始尺寸（2D 专用）
        transform: 变形属性（旋转、缩放）
        z_index: 渲染层级，数值大的在上层绘制
        
        # 3D 渲染专用字段
        geometry: 程序几何体类型 (box/sphere/cylinder/capsule/cone/plane)
        model_path: 外部 3D 模型文件路径 (.gltf/.glb/.obj/.stl)
        material: PBR 材质配置
        scale_3d: 3D 缩放 (sx, sy, sz)，优先于 transform 中的缩放
    
    Example:
        >>> # 2D 物体
        >>> obj_2d = ObjectData(
        ...     id="player",
        ...     pos=(100, 200),
        ...     size=(64, 64),
        ...     transform=Transform(rotation_z=30),
        ...     z_index=10
        ... )
        >>> 
        >>> # 3D 程序几何体
        >>> obj_3d = ObjectData(
        ...     id="sphere_1",
        ...     pos=(0, 1, 0),
        ...     geometry="sphere",
        ...     material=Material3D(base_color=(1, 0, 0, 1), metallic=0.8)
        ... )
        >>> 
        >>> # 3D 外部模型
        >>> obj_model = ObjectData(
        ...     id="building",
        ...     pos=(5, 0, 0),
        ...     model_path="assets/building.gltf"
        ... )
    """
    
    id: str = Field(..., description="物体 ID，用于匹配素材文件")
    pos: Union[Tuple[float, float], Tuple[float, float, float]] = Field(
        ..., description="坐标位置 (x, y) 或 (x, y, z)"
    )
    size: Optional[Tuple[int, int]] = Field(
        default=None, description="像素尺寸 (width, height)，None 时使用素材原始尺寸（2D 专用）"
    )
    transform: Optional[Transform] = Field(
        default=None, description="变形属性（旋转、缩放）"
    )
    z_index: int = Field(default=0, description="渲染层级，数值大的在上层")
    
    # 3D 渲染专用字段
    geometry: Optional[Literal[
        "box", "sphere", "cylinder", "capsule", "cone", "plane", "torus"
    ]] = Field(
        default=None,
        description="程序几何体类型，指定后将生成对应的基础几何体"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="外部 3D 模型文件路径 (.gltf/.glb/.obj/.stl)"
    )
    material: Optional[Material3D] = Field(
        default=None,
        description="PBR 材质配置"
    )
    scale_3d: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="3D 缩放 (sx, sy, sz)"
    )
    
    model_config = {"extra": "allow"}  # 允许自定义额外字段


class SemanticView(BaseModel):
    """
    Env 与 Renderer 之间的标准数据格式。
    
    由环境生成，传递给渲染器进行可视化。包含视野区域、可见物体列表和元数据。
    支持 2D 和 3D 渲染场景配置。
    
    Attributes:
        view_region: Camera.view 原样传递，供 Renderer 做裁剪/相机设置
            - 2D: {"x": 0, "y": 0, "width": 800, "height": 600}
            - 3D: {"camera_position": [10,10,10], "camera_target": [0,0,0], "fov": 60}
        objects: 可见物体列表，支持 ObjectData 或兼容的 Dict 格式
        lights: 3D 场景光源配置列表
        metadata: 可选的额外元数据（如步数、分数、全局状态等）
    
    Example:
        >>> # 2D 场景
        >>> view_2d = SemanticView(
        ...     view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        ...     objects=[
        ...         ObjectData(id="player", pos=(100, 200)),
        ...         {"id": "wall", "pos": (50, 50)}
        ...     ]
        ... )
        >>> 
        >>> # 3D 场景
        >>> view_3d = SemanticView(
        ...     view_region={
        ...         "camera_position": [10, 10, 10],
        ...         "camera_target": [0, 0, 0],
        ...         "fov": 60
        ...     },
        ...     objects=[
        ...         ObjectData(
        ...             id="sphere_1",
        ...             pos=(0, 1, 0),
        ...             geometry="sphere",
        ...             material=Material3D(base_color=(1, 0, 0, 1))
        ...         )
        ...     ],
        ...     lights=[
        ...         LightConfig(type="directional", direction=(-1, -1, -1)),
        ...         LightConfig(type="ambient", intensity=0.3)
        ...     ]
        ... )
    """
    
    view_region: Any = Field(..., description="Camera.view 原样传递，给 Renderer 做裁剪/相机设置")
    objects: List[Union[ObjectData, Dict[str, Any]]] = Field(
        default_factory=list,
        description="可见物体列表"
    )
    lights: Optional[List[Union[LightConfig, Dict[str, Any]]]] = Field(
        default=None,
        description="3D 场景光源配置列表"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="额外元数据（步数、分数等）"
    )
    
    def get_normalized_objects(self) -> List[ObjectData]:
        """
        将所有物体转换为 ObjectData 格式。
        
        对于 Dict 格式的物体，会自动转换为 ObjectData。
        
        Returns:
            标准化的 ObjectData 列表
        """
        result = []
        for obj in self.objects:
            if isinstance(obj, ObjectData):
                result.append(obj)
            else:
                # 兼容旧的 Dict 格式
                result.append(ObjectData(**obj))
        return result
    
    def get_object_by_id(self, obj_id: str) -> Optional[Union[ObjectData, Dict[str, Any]]]:
        """
        根据 ID 获取物体。
        
        Args:
            obj_id: 物体 ID
            
        Returns:
            物体对象，如果不存在则返回 None
        """
        for obj in self.objects:
            if isinstance(obj, ObjectData):
                if obj.id == obj_id:
                    return obj
            elif obj.get("id") == obj_id:
                return obj
        return None
    
    def get_objects_by_type(self, type_prefix: str) -> List[Union[ObjectData, Dict[str, Any]]]:
        """
        根据 ID 前缀获取同类物体。
        
        Args:
            type_prefix: ID 前缀，如 "wall" 匹配 "wall_1", "wall_2" 等
            
        Returns:
            匹配的物体列表
        """
        result = []
        for obj in self.objects:
            if isinstance(obj, ObjectData):
                if obj.id.startswith(type_prefix):
                    result.append(obj)
            elif obj.get("id", "").startswith(type_prefix):
                result.append(obj)
        return result
    
    def __len__(self) -> int:
        """返回可见物体数量"""
        return len(self.objects)
