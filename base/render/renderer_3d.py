# ============================================================
# 3D RENDERER (PyRender + Trimesh)
# Purpose: Render SemanticView to 3D image using PyRender
# ============================================================

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    pyrender = None

from base.render.base_renderer import BaseRenderer, RenderConfig
from base.env.semantic_view import SemanticView, ObjectData, Material3D, LightConfig


class Renderer3D(BaseRenderer):
    """
    3D 渲染器 - 基于 PyRender + Trimesh。
    
    将 SemanticView 渲染为 3D 图像，支持：
    - 程序几何体：box, sphere, cylinder, capsule, cone, plane, torus
    - 外部模型加载：.gltf, .glb, .obj, .stl
    - PBR 材质：金属度、粗糙度、基础颜色、纹理
    - 多种光源：环境光、方向光、点光源、聚光灯
    - 离屏渲染（无需显示器）
    
    依赖：
    - pyrender: 3D 渲染引擎
    - trimesh: 3D 网格处理
    - numpy: 数值计算
    - Pillow: 图像处理
    
    Example:
        >>> config = RenderConfig(resolution=(800, 600))
        >>> renderer = Renderer3D(config)
        >>> 
        >>> semantic_view = SemanticView(
        ...     view_region={
        ...         "camera_position": [10, 10, 10],
        ...         "camera_target": [0, 0, 0],
        ...         "fov": 60
        ...     },
        ...     objects=[
        ...         {
        ...             "id": "sphere_1",
        ...             "pos": (0, 1, 0),
        ...             "geometry": "sphere",
        ...             "material": {"base_color": [1, 0, 0, 1], "metallic": 0.5}
        ...         }
        ...     ]
        ... )
        >>> image = renderer.render(semantic_view)
        >>> image.save("output_3d.png")
    """
    
    # 几何体生成函数映射
    GEOMETRY_CREATORS = {
        "box": lambda: trimesh.creation.box(extents=[1, 1, 1]),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.5),
        "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1.0),
        "capsule": lambda: trimesh.creation.capsule(radius=0.3, height=0.6),
        "cone": lambda: trimesh.creation.cone(radius=0.5, height=1.0),
        "plane": lambda: trimesh.creation.box(extents=[10, 0.01, 10]),
        "torus": lambda: trimesh.creation.torus(major_radius=0.5, minor_radius=0.2),
    }
    
    # 默认颜色映射（根据 ID 前缀）
    DEFAULT_COLORS = {
        "player": (0.0, 1.0, 0.0, 1.0),
        "enemy": (1.0, 0.0, 0.0, 1.0),
        "wall": (0.5, 0.5, 0.5, 1.0),
        "floor": (0.3, 0.3, 0.3, 1.0),
        "treasure": (1.0, 0.84, 0.0, 1.0),
        "water": (0.25, 0.25, 1.0, 0.8),
        "fire": (1.0, 0.27, 0.0, 1.0),
        "goal": (0.0, 1.0, 1.0, 1.0),
    }
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        初始化 3D 渲染器。
        
        Args:
            config: 渲染配置
            
        Raises:
            ImportError: 如果必要的依赖未安装
        """
        if not NUMPY_AVAILABLE:
            raise ImportError(
                "Renderer3D requires numpy. "
                "Install with: pip install numpy"
            )
        if not PIL_AVAILABLE:
            raise ImportError(
                "Renderer3D requires PIL/Pillow. "
                "Install with: pip install Pillow"
            )
        if not TRIMESH_AVAILABLE:
            raise ImportError(
                "Renderer3D requires trimesh. "
                "Install with: pip install trimesh"
            )
        if not PYRENDER_AVAILABLE:
            raise ImportError(
                "Renderer3D requires pyrender. "
                "Install with: pip install pyrender"
            )
        
        super().__init__(config)
        self._renderer: Optional[pyrender.OffscreenRenderer] = None
        self._model_cache: Dict[str, trimesh.Trimesh] = {}
    
    def _get_renderer(self) -> "pyrender.OffscreenRenderer":
        """获取或创建离屏渲染器"""
        width, height = self.config.resolution
        if self._renderer is None:
            self._renderer = pyrender.OffscreenRenderer(width, height)
        return self._renderer
    
    def render(self, semantic_view: SemanticView) -> "Image.Image":
        """
        渲染 SemanticView 为 3D 图像。
        
        Args:
            semantic_view: 语义视图数据
            
        Returns:
            PIL Image 对象
        """
        # 创建场景
        scene = pyrender.Scene(
            bg_color=self._parse_background_color(),
            ambient_light=[0.3, 0.3, 0.3]
        )
        
        # 添加物体
        for obj in semantic_view.objects:
            self._add_object_to_scene(scene, obj)
        
        # 设置光源
        self._setup_lights(scene, semantic_view)
        
        # 设置相机
        camera_node = self._setup_camera(scene, semantic_view)
        
        # 渲染
        renderer = self._get_renderer()
        color, depth = renderer.render(scene)
        
        # 转换为 PIL Image
        image = Image.fromarray(color)
        return image
    
    def render_with_camera(
        self,
        semantic_view: SemanticView,
        camera_position: Tuple[float, float, float],
        camera_target: Tuple[float, float, float] = (0, 0, 0),
        fov: float = 60
    ) -> "Image.Image":
        """
        使用自定义相机参数渲染。
        
        Args:
            semantic_view: 语义视图数据
            camera_position: 相机位置 (x, y, z)
            camera_target: 相机目标点 (x, y, z)
            fov: 视野角度（度）
            
        Returns:
            PIL Image 对象
        """
        # 创建临时的 view_region
        original_view_region = semantic_view.view_region
        
        # 修改 view_region
        if isinstance(semantic_view.view_region, dict):
            new_view_region = dict(semantic_view.view_region)
        else:
            new_view_region = {}
        
        new_view_region["camera_position"] = list(camera_position)
        new_view_region["camera_target"] = list(camera_target)
        new_view_region["fov"] = fov
        
        # 临时修改
        semantic_view.view_region = new_view_region
        
        try:
            return self.render(semantic_view)
        finally:
            semantic_view.view_region = original_view_region
    
    def _parse_background_color(self) -> List[float]:
        """解析背景颜色为 RGBA 列表"""
        bg = self.config.background
        
        if hasattr(bg, 'color'):
            color = bg.color
            if isinstance(color, str) and color.startswith("#"):
                color = color[1:]
                r = int(color[0:2], 16) / 255.0
                g = int(color[2:4], 16) / 255.0
                b = int(color[4:6], 16) / 255.0
                return [r, g, b, 1.0]
        
        return [0.1, 0.1, 0.15, 1.0]  # 默认深色背景
    
    def _add_object_to_scene(
        self,
        scene: "pyrender.Scene",
        obj: Union[ObjectData, Dict[str, Any]]
    ) -> None:
        """将物体添加到场景"""
        # 标准化为 dict
        if isinstance(obj, ObjectData):
            obj_dict = obj.model_dump()
        else:
            obj_dict = obj
        
        obj_id = obj_dict.get("id", "unknown")
        pos = obj_dict.get("pos", (0, 0, 0))
        
        # 获取网格
        mesh = self._get_mesh_for_object(obj_dict)
        if mesh is None:
            return
        
        # 创建材质
        material = self._create_material(obj_dict)
        
        # 创建 PyRender 网格
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # 计算变换矩阵
        transform = self._compute_transform(obj_dict)
        
        # 添加到场景
        scene.add(pr_mesh, pose=transform)
    
    def _get_mesh_for_object(self, obj: Dict[str, Any]) -> Optional["trimesh.Trimesh"]:
        """获取物体的网格"""
        obj_id = obj.get("id", "unknown")
        
        # 1. 检查是否有外部模型路径
        model_path = obj.get("model_path")
        if model_path:
            return self._load_model(model_path)
        
        # 2. 检查是否指定了几何体类型
        geometry = obj.get("geometry")
        if geometry and geometry in self.GEOMETRY_CREATORS:
            return self.GEOMETRY_CREATORS[geometry]()
        
        # 3. 根据 ID 前缀推断几何体
        base_type = obj_id.split("_")[0].lower()
        if base_type in self.GEOMETRY_CREATORS:
            return self.GEOMETRY_CREATORS[base_type]()
        
        # 4. 默认使用 box
        return self.GEOMETRY_CREATORS["box"]()
    
    def _load_model(self, path: str) -> Optional["trimesh.Trimesh"]:
        """加载外部 3D 模型"""
        # 检查缓存
        if path in self._model_cache:
            return self._model_cache[path].copy()
        
        # 构建完整路径
        if not os.path.isabs(path):
            full_path = os.path.join(self.config.asset_path, path)
        else:
            full_path = path
        
        if not os.path.exists(full_path):
            print(f"Warning: Model file not found: {full_path}")
            return None
        
        try:
            # 加载模型
            loaded = trimesh.load(full_path)
            
            # 处理场景（多个网格）
            if isinstance(loaded, trimesh.Scene):
                # 合并所有网格
                meshes = []
                for name, geom in loaded.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    return None
            else:
                mesh = loaded
            
            # 缓存
            self._model_cache[path] = mesh
            return mesh.copy()
            
        except Exception as e:
            print(f"Warning: Failed to load model {full_path}: {e}")
            return None
    
    def _create_material(self, obj: Dict[str, Any]) -> "pyrender.MetallicRoughnessMaterial":
        """创建 PBR 材质"""
        obj_id = obj.get("id", "unknown")
        material_config = obj.get("material") or {}
        
        # 处理 Material3D 对象
        if isinstance(material_config, Material3D):
            material_config = material_config.model_dump()
        elif not isinstance(material_config, dict):
            material_config = {}
        
        # 获取基础颜色
        if material_config and "base_color" in material_config:
            base_color = material_config["base_color"]
            if len(base_color) == 3:
                base_color = list(base_color) + [1.0]
        else:
            # 根据 ID 前缀获取默认颜色
            base_type = obj_id.split("_")[0].lower()
            base_color = self.DEFAULT_COLORS.get(base_type, self._id_to_color(obj_id))
        
        # 获取 PBR 参数
        metallic = material_config.get("metallic", 0.0)
        roughness = material_config.get("roughness", 0.5)
        emissive = material_config.get("emissive", [0.0, 0.0, 0.0])
        alpha_mode = material_config.get("alpha_mode", "OPAQUE")
        
        # 加载纹理
        base_color_texture = None
        texture_path = material_config.get("texture_path")
        if texture_path:
            base_color_texture = self._load_texture(texture_path)
        
        # 创建材质
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=list(base_color),
            metallicFactor=metallic,
            roughnessFactor=roughness,
            emissiveFactor=list(emissive),
            baseColorTexture=base_color_texture,
            alphaMode=alpha_mode.upper()
        )
        
        return material
    
    def _load_texture(self, path: str) -> Optional["pyrender.Texture"]:
        """加载纹理"""
        if not os.path.isabs(path):
            full_path = os.path.join(self.config.asset_path, path)
        else:
            full_path = path
        
        if not os.path.exists(full_path):
            return None
        
        try:
            image = Image.open(full_path)
            return pyrender.Texture(source=image)
        except Exception as e:
            print(f"Warning: Failed to load texture {full_path}: {e}")
            return None
    
    def _compute_transform(self, obj: Dict[str, Any]) -> np.ndarray:
        """计算物体的变换矩阵"""
        pos = obj.get("pos", (0, 0, 0))
        
        # 处理位置
        if len(pos) == 2:
            x, y = pos
            z = 0
        else:
            x, y, z = pos
        
        # 创建平移矩阵
        transform = np.eye(4)
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        
        # 处理缩放
        scale = obj.get("scale_3d")
        if scale:
            sx, sy, sz = scale
            scale_matrix = np.diag([sx, sy, sz, 1.0])
            transform = transform @ scale_matrix
        
        # 处理旋转（从 transform 字段）
        obj_transform = obj.get("transform")
        if obj_transform:
            if isinstance(obj_transform, dict):
                rot_x = math.radians(obj_transform.get("rotation_x", 0))
                rot_y = math.radians(obj_transform.get("rotation_y", 0))
                rot_z = math.radians(obj_transform.get("rotation_z", 0))
            else:
                rot_x = math.radians(getattr(obj_transform, "rotation_x", 0))
                rot_y = math.radians(getattr(obj_transform, "rotation_y", 0))
                rot_z = math.radians(getattr(obj_transform, "rotation_z", 0))
            
            # 应用旋转
            if rot_x != 0 or rot_y != 0 or rot_z != 0:
                rotation_matrix = self._euler_to_matrix(rot_x, rot_y, rot_z)
                transform[:3, :3] = rotation_matrix @ transform[:3, :3]
        
        return transform
    
    def _euler_to_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """欧拉角转旋转矩阵 (XYZ 顺序)"""
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        
        # Rotation matrix (XYZ order)
        return np.array([
            [cy * cz, -cy * sz, sy],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
            [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
        ])
    
    def _setup_lights(self, scene: "pyrender.Scene", semantic_view: SemanticView) -> None:
        """设置场景光源"""
        lights = semantic_view.lights
        
        if not lights:
            # 默认光源配置
            self._add_default_lights(scene)
            return
        
        for light_config in lights:
            if isinstance(light_config, dict):
                self._add_light_from_config(scene, light_config)
            elif isinstance(light_config, LightConfig):
                self._add_light_from_config(scene, light_config.model_dump())
    
    def _add_default_lights(self, scene: "pyrender.Scene") -> None:
        """添加默认光源"""
        # 主方向光
        directional_light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],
            intensity=3.0
        )
        light_pose = self._look_at_matrix(
            eye=[5, 10, 5],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        scene.add(directional_light, pose=light_pose)
        
        # 补光
        fill_light = pyrender.DirectionalLight(
            color=[0.8, 0.85, 1.0],
            intensity=1.0
        )
        fill_pose = self._look_at_matrix(
            eye=[-3, 5, -3],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
        scene.add(fill_light, pose=fill_pose)
    
    def _add_light_from_config(self, scene: "pyrender.Scene", config: Dict[str, Any]) -> None:
        """根据配置添加光源"""
        light_type = config.get("type", "directional")
        color = config.get("color", [1.0, 1.0, 1.0])
        intensity = config.get("intensity", 1.0)
        
        if light_type == "ambient":
            # 环境光通过场景设置
            scene.ambient_light = np.array(color) * intensity
            return
        
        elif light_type == "directional":
            light = pyrender.DirectionalLight(color=color, intensity=intensity)
            direction = config.get("direction", [-1, -1, -1])
            # 计算光源朝向
            pose = self._direction_to_pose(direction)
            
        elif light_type == "point":
            light = pyrender.PointLight(color=color, intensity=intensity)
            position = config.get("position", [0, 5, 0])
            pose = np.eye(4)
            pose[0, 3] = position[0]
            pose[1, 3] = position[1]
            pose[2, 3] = position[2]
            
        elif light_type == "spot":
            inner_angle = math.radians(config.get("inner_cone_angle", 30))
            outer_angle = math.radians(config.get("outer_cone_angle", 45))
            light = pyrender.SpotLight(
                color=color,
                intensity=intensity,
                innerConeAngle=inner_angle,
                outerConeAngle=outer_angle
            )
            position = config.get("position", [0, 5, 0])
            direction = config.get("direction", [0, -1, 0])
            pose = self._spot_light_pose(position, direction)
        else:
            return
        
        scene.add(light, pose=pose)
    
    def _direction_to_pose(self, direction: List[float]) -> np.ndarray:
        """将方向向量转换为光源姿态矩阵"""
        direction = np.array(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        
        # 光源位于原点，指向 direction
        pose = np.eye(4)
        
        # 计算旋转，使 -Z 轴指向 direction
        z_axis = -direction
        
        # 选择一个不平行的向量作为参考
        if abs(z_axis[1]) < 0.9:
            up = np.array([0, 1, 0])
        else:
            up = np.array([1, 0, 0])
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        
        return pose
    
    def _spot_light_pose(self, position: List[float], direction: List[float]) -> np.ndarray:
        """计算聚光灯姿态矩阵"""
        pose = self._direction_to_pose(direction)
        pose[0, 3] = position[0]
        pose[1, 3] = position[1]
        pose[2, 3] = position[2]
        return pose
    
    def _setup_camera(self, scene: "pyrender.Scene", semantic_view: SemanticView) -> "pyrender.Node":
        """设置相机"""
        view_region = semantic_view.view_region
        
        # 解析相机参数
        if isinstance(view_region, dict):
            cam_pos = view_region.get("camera_position", [10, 10, 10])
            cam_target = view_region.get("camera_target", [0, 0, 0])
            fov = view_region.get("fov", 60)
        else:
            cam_pos = [10, 10, 10]
            cam_target = [0, 0, 0]
            fov = 60
        
        # 创建透视相机
        width, height = self.config.resolution
        aspect_ratio = width / height
        
        camera = pyrender.PerspectiveCamera(
            yfov=math.radians(fov),
            aspectRatio=aspect_ratio
        )
        
        # 计算相机姿态
        camera_pose = self._look_at_matrix(
            eye=cam_pos,
            target=cam_target,
            up=[0, 1, 0]
        )
        
        return scene.add(camera, pose=camera_pose)
    
    def _look_at_matrix(
        self,
        eye: List[float],
        target: List[float],
        up: List[float]
    ) -> np.ndarray:
        """计算 look-at 变换矩阵"""
        eye = np.array(eye, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        up = np.array(up, dtype=np.float64)
        
        # 计算相机坐标系
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建变换矩阵
        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye
        
        return pose
    
    def _id_to_color(self, obj_id: str) -> Tuple[float, float, float, float]:
        """根据 ID 生成确定性颜色"""
        if not obj_id:
            return (0.5, 0.5, 0.5, 1.0)
        
        hash_val = hash(obj_id)
        r = max(((hash_val & 0xFF0000) >> 16) / 255.0, 0.25)
        g = max(((hash_val & 0x00FF00) >> 8) / 255.0, 0.25)
        b = max((hash_val & 0x0000FF) / 255.0, 0.25)
        
        return (r, g, b, 1.0)
    
    def _load_asset_impl(self, obj_id: str) -> Optional[str]:
        """加载素材实现"""
        for ext in [".gltf", ".glb", ".obj", ".stl"]:
            path = self.get_asset_path(obj_id, ext)
            if os.path.exists(path):
                return path
        return None
    
    def cleanup(self) -> None:
        """清理渲染器资源"""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
        self._model_cache.clear()
        self.clear_cache()
    
    def __del__(self):
        """析构函数"""
        self.cleanup()


class Renderer3DFallback(BaseRenderer):
    """
    3D 渲染器的降级实现。
    
    当 PyRender 不可用时，提供简单的 2D 俯视图作为降级方案。
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        if not PIL_AVAILABLE:
            raise ImportError("Renderer3DFallback requires PIL/Pillow")
        super().__init__(config)
    
    def render(self, semantic_view: SemanticView) -> "Image.Image":
        """渲染降级的 2D 俯视图"""
        width, height = self.config.resolution
        
        # 解析背景颜色
        bg_color = self._parse_color(self._get_background_color())
        
        # 创建图像
        image = Image.new("RGB", (width, height), bg_color)
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        
        # 绘制网格
        grid_color = (60, 60, 60)
        step = 40
        for x in range(0, width, step):
            draw.line([(x, 0), (x, height)], fill=grid_color)
        for y in range(0, height, step):
            draw.line([(0, y), (width, y)], fill=grid_color)
        
        # 绘制物体
        center_x, center_y = width // 2, height // 2
        scale = 30
        
        for obj in semantic_view.objects:
            if isinstance(obj, ObjectData):
                obj = obj.model_dump()
            
            pos = obj.get("pos", [0, 0, 0])
            obj_id = obj.get("id", "")
            
            # 提取位置
            if isinstance(pos, (list, tuple)):
                x = pos[0] if len(pos) > 0 else 0
                z = pos[2] if len(pos) > 2 else (pos[1] if len(pos) > 1 else 0)
            else:
                x, z = 0, 0
            
            # 转换为屏幕坐标
            screen_x = center_x + int(x * scale)
            screen_y = center_y + int(z * scale)
            
            # 绘制物体
            color = self._id_to_color(obj_id)
            size = 15
            draw.ellipse(
                [screen_x - size, screen_y - size, screen_x + size, screen_y + size],
                fill=color,
                outline=(255, 255, 255)
            )
        
        # 添加降级提示
        draw.text((10, 10), "3D Fallback (PyRender Unavailable)", fill=(200, 200, 200))
        
        return image
    
    def _get_background_color(self) -> str:
        """获取背景颜色"""
        bg = self.config.background
        if hasattr(bg, 'color'):
            return bg.color
        return "#1a1a2e"
    
    def _parse_color(self, color: str) -> Tuple[int, int, int]:
        if color.startswith("#"):
            color = color[1:]
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            return (r, g, b)
        return (0, 0, 0)
    
    def _id_to_color(self, obj_id: str) -> Tuple[int, int, int]:
        if not obj_id:
            return (128, 128, 128)
        hash_val = hash(obj_id)
        r = max((hash_val & 0xFF0000) >> 16, 64)
        g = max((hash_val & 0x00FF00) >> 8, 64)
        b = max(hash_val & 0x0000FF, 64)
        return (r, g, b)
    
    def _load_asset_impl(self, obj_id: str) -> None:
        return None


def create_renderer(config: Optional[RenderConfig] = None) -> BaseRenderer:
    """
    创建 3D 渲染器的工厂函数。
    
    如果 PyRender 可用，返回 Renderer3D；否则返回 Renderer3DFallback。
    
    Args:
        config: 渲染配置
        
    Returns:
        3D 渲染器实例
    """
    if PYRENDER_AVAILABLE and TRIMESH_AVAILABLE:
        return Renderer3D(config)
    else:
        return Renderer3DFallback(config)
