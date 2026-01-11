# ============================================================
# 3D渲染器基础功能测试
# 测试Renderer3D的程序几何体、外部模型、PBR材质、多光照等功能
# ============================================================

"""
测试内容：
    1. 程序几何体测试：box, sphere, cylinder, capsule, cone, plane, torus
    2. 外部模型加载测试：test_bugatti.obj, Tree1_test.obj（带scale_3d配置）
    3. PBR材质测试：base_color, metallic, roughness, emissive, alpha_mode
    4. 多光照测试：ambient, directional, point, spot
    5. 相机参数测试：不同fov, position, target
    6. 视野遮罩测试：圆形、扇形、环形遮罩
"""

import sys
import os
import math

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from base.render.renderer_3d import Renderer3D, create_renderer, PYRENDER_AVAILABLE
from base.render.base_renderer import RenderConfig, BackgroundColor
from base.env.semantic_view import SemanticView, ObjectData, Transform, Material3D, LightConfig


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "3d")
TEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")


def ensure_output_dirs():
    """确保输出目录存在"""
    subdirs = ["geometry", "models", "materials", "lighting"]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def save_image(image, subdir: str, filename: str):
    """保存图像到文件"""
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    image.save(path)
    print(f"    保存: {path}")


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_3d_available():
    """检查3D渲染依赖是否可用"""
    if not PYRENDER_AVAILABLE:
        print("\n  ⚠ PyRender不可用，将使用降级渲染器")
        print("    安装方法: pip install pyrender trimesh")
        return False
    return True


# ============================================================
# 测试1: 程序几何体测试
# ============================================================
def test_geometry():
    """测试所有程序几何体类型"""
    print_separator("测试1: 程序几何体测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#1a1a2e")
    )
    renderer = create_renderer(config)
    
    # 所有支持的几何体类型
    geometries = ["box", "sphere", "cylinder", "capsule", "cone", "plane", "torus"]
    
    # 1.1 单个几何体展示
    print("\n  [1.1] 单个几何体展示")
    for geom in geometries:
        objects = [
            ObjectData(
                id=f"{geom}_center",
                pos=(0, 0.5, 0),
                geometry=geom,
                material=Material3D(
                    base_color=(0.7, 0.3, 0.3, 1.0),
                    metallic=0.3,
                    roughness=0.5
                )
            ),
            # 地面参考
            ObjectData(
                id="ground",
                pos=(0, -0.01, 0),
                geometry="plane",
                scale_3d=(5, 1, 5),
                material=Material3D(
                    base_color=(0.3, 0.3, 0.35, 1.0),
                    roughness=0.8
                )
            )
        ]
        
        semantic_view = SemanticView(
            view_region={
                "camera_position": [3, 3, 3],
                "camera_target": [0, 0.5, 0],
                "fov": 60
            },
            objects=objects,
            lights=[
                LightConfig(type="directional", direction=(-1, -1, -1), intensity=2.0),
                LightConfig(type="ambient", intensity=0.3)
            ]
        )
        
        image = renderer.render(semantic_view)
        save_image(image, "geometry", f"single_{geom}.png")
    print(f"    ✓ 生成{len(geometries)}种几何体单独展示")
    
    # 1.2 所有几何体组合展示
    print("\n  [1.2] 所有几何体组合展示")
    all_objects = []
    
    # 排列几何体
    positions = [
        (-3, 0.5, -2), (-1, 0.5, -2), (1, 0.5, -2), (3, 0.5, -2),
        (-2, 0.5, 1), (0, 0.5, 1), (2, 0.5, 1)
    ]
    
    colors = [
        (0.9, 0.2, 0.2, 1.0),   # 红
        (0.2, 0.9, 0.2, 1.0),   # 绿
        (0.2, 0.2, 0.9, 1.0),   # 蓝
        (0.9, 0.9, 0.2, 1.0),   # 黄
        (0.9, 0.2, 0.9, 1.0),   # 紫
        (0.2, 0.9, 0.9, 1.0),   # 青
        (0.9, 0.6, 0.2, 1.0),   # 橙
    ]
    
    for i, (geom, pos, color) in enumerate(zip(geometries, positions, colors)):
        all_objects.append(ObjectData(
            id=f"{geom}_{i}",
            pos=pos,
            geometry=geom,
            material=Material3D(
                base_color=color,
                metallic=0.4,
                roughness=0.4
            )
        ))
    
    # 添加地面
    all_objects.append(ObjectData(
        id="ground",
        pos=(0, -0.01, 0),
        geometry="plane",
        scale_3d=(10, 1, 8),
        material=Material3D(
            base_color=(0.25, 0.25, 0.3, 1.0),
            roughness=0.9
        )
    ))
    
    semantic_view_all = SemanticView(
        view_region={
            "camera_position": [0, 8, 10],
            "camera_target": [0, 0, 0],
            "fov": 50
        },
        objects=all_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -2, -1), intensity=2.5),
            LightConfig(type="directional", direction=(1, -1, 1), intensity=1.0, color=(0.8, 0.9, 1.0)),
            LightConfig(type="ambient", intensity=0.3)
        ]
    )
    
    image_all = renderer.render(semantic_view_all)
    save_image(image_all, "geometry", "all_geometries.png")
    print("    ✓ 所有几何体组合展示完成")
    
    # 1.3 几何体缩放测试
    print("\n  [1.3] 几何体缩放测试")
    scale_objects = []
    scales = [
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (1.5, 1.5, 1.5),
        (2.0, 0.5, 0.5),  # 横向拉伸
        (0.5, 2.0, 0.5),  # 纵向拉伸
    ]
    
    for i, scale in enumerate(scales):
        scale_objects.append(ObjectData(
            id=f"box_scale_{i}",
            pos=(-4 + i * 2, scale[1] / 2, 0),
            geometry="box",
            scale_3d=scale,
            material=Material3D(
                base_color=(0.5 + i * 0.1, 0.7 - i * 0.1, 0.3, 1.0),
                roughness=0.5
            )
        ))
    
    scale_objects.append(ObjectData(
        id="ground",
        pos=(0, -0.01, 0),
        geometry="plane",
        scale_3d=(12, 1, 6),
        material=Material3D(base_color=(0.3, 0.3, 0.35, 1.0))
    ))
    
    semantic_view_scale = SemanticView(
        view_region={
            "camera_position": [0, 5, 8],
            "camera_target": [0, 0.5, 0],
            "fov": 55
        },
        objects=scale_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -1, -1), intensity=2.0),
            LightConfig(type="ambient", intensity=0.3)
        ]
    )
    
    image_scale = renderer.render(semantic_view_scale)
    save_image(image_scale, "geometry", "scale_comparison.png")
    print("    ✓ 几何体缩放测试完成")


# ============================================================
# 测试2: 外部模型加载测试
# ============================================================
def test_external_models():
    """测试外部3D模型加载"""
    print_separator("测试2: 外部模型加载测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(1024, 768),
        background=BackgroundColor(color="#0f0f1a")
    )
    renderer = create_renderer(config)
    
    # 模型配置（包含缩放因子）
    models = [
        {
            "name": "bugatti",
            "path": os.path.join(TEST_MODELS_DIR, "test_bugatti.obj"),
            "scale": (1.0, 1.0, 1.0),  # 保持原始尺寸
            "pos": (0, 0, 0),
            "camera_distance": 15,
        },
        {
            "name": "tree",
            "path": os.path.join(TEST_MODELS_DIR, "Tree1_test.obj"),
            "scale": (0.5, 0.5, 0.5),  # 模型适中
            "pos": (0, 0, 0),
            "camera_distance": 10,
        }
    ]
    
    # 2.1 单独展示每个模型
    print("\n  [2.1] 单独模型展示")
    for model_info in models:
        if not os.path.exists(model_info["path"]):
            print(f"    ⚠ 模型文件不存在: {model_info['path']}")
            continue
        
        objects = [
            ObjectData(
                id=model_info["name"],
                pos=model_info["pos"],
                model_path=model_info["path"],
                scale_3d=model_info["scale"],
                material=Material3D(
                    base_color=(0.8, 0.8, 0.8, 1.0),
                    metallic=0.3,
                    roughness=0.6
                )
            ),
            # 地面
            ObjectData(
                id="ground",
                pos=(0, -0.1, 0),
                geometry="plane",
                scale_3d=(50, 1, 50),
                material=Material3D(
                    base_color=(0.2, 0.22, 0.25, 1.0),
                    roughness=0.9
                )
            )
        ]
        
        dist = model_info["camera_distance"]
        semantic_view = SemanticView(
            view_region={
                "camera_position": [dist, dist * 0.8, dist],
                "camera_target": [0, 0, 0],
                "fov": 45
            },
            objects=objects,
            lights=[
                LightConfig(type="directional", direction=(-1, -1.5, -1), intensity=3.0),
                LightConfig(type="directional", direction=(1, -0.5, 0.5), intensity=1.5, color=(0.9, 0.95, 1.0)),
                LightConfig(type="ambient", intensity=0.4)
            ]
        )
        
        image = renderer.render(semantic_view)
        save_image(image, "models", f"{model_info['name']}_single.png")
    print("    ✓ 单独模型展示完成")
    
    # 2.2 模型组合场景
    print("\n  [2.2] 模型组合场景")
    scene_objects = [
        # 中心车辆
        ObjectData(
            id="vehicle",
            pos=(0, 0, 0),
            model_path=os.path.join(TEST_MODELS_DIR, "test_bugatti.obj"),
            scale_3d=(1.0, 1.0, 1.0),
            transform=Transform(rotation_y=30)
        ),
        # 周围的树
        ObjectData(
            id="tree_0",
            pos=(-4, 0, -3),
            model_path=os.path.join(TEST_MODELS_DIR, "Tree1_test.obj"),
            scale_3d=(0.4, 0.4, 0.4)
        ),
        ObjectData(
            id="tree_1",
            pos=(4, 0, -3),
            model_path=os.path.join(TEST_MODELS_DIR, "Tree1_test.obj"),
            scale_3d=(0.5, 0.5, 0.5)
        ),
        ObjectData(
            id="tree_2",
            pos=(-3, 0, 4),
            model_path=os.path.join(TEST_MODELS_DIR, "Tree1_test.obj"),
            scale_3d=(0.35, 0.35, 0.35)
        ),
        # 地面
        ObjectData(
            id="ground",
            pos=(0, -0.05, 0),
            geometry="plane",
            scale_3d=(40, 1, 40),
            material=Material3D(
                base_color=(0.15, 0.3, 0.15, 1.0),  # 草地绿
                roughness=0.95
            )
        )
    ]
    
    semantic_view_scene = SemanticView(
        view_region={
            "camera_position": [20, 15, 25],
            "camera_target": [0, 0, 0],
            "fov": 50
        },
        objects=scene_objects,
        lights=[
            LightConfig(type="directional", direction=(-0.5, -1, -0.5), intensity=2.5, color=(1.0, 0.98, 0.95)),
            LightConfig(type="ambient", intensity=0.4, color=(0.6, 0.7, 0.9))
        ]
    )
    
    image_scene = renderer.render(semantic_view_scene)
    save_image(image_scene, "models", "combined_scene.png")
    print("    ✓ 模型组合场景完成")
    
    # 2.3 不同视角展示同一模型
    print("\n  [2.3] 不同视角展示")
    camera_angles = [
        ("front", [0, 5, 20]),
        ("side", [20, 5, 0]),
        ("top", [0, 25, 0.1]),
        ("diagonal", [15, 10, 15]),
    ]
    
    for angle_name, cam_pos in camera_angles:
        objects = [
            ObjectData(
                id="bugatti",
                pos=(0, 0, 0),
                model_path=os.path.join(TEST_MODELS_DIR, "test_bugatti.obj"),
                scale_3d=(1.0, 1.0, 1.0)
            ),
            ObjectData(
                id="ground",
                pos=(0, -0.05, 0),
                geometry="plane",
                scale_3d=(40, 1, 40),
                material=Material3D(base_color=(0.25, 0.25, 0.3, 1.0))
            )
        ]
        
        semantic_view_angle = SemanticView(
            view_region={
                "camera_position": cam_pos,
                "camera_target": [0, 0, 0],
                "fov": 50
            },
            objects=objects,
            lights=[
                LightConfig(type="directional", direction=(-1, -1, -1), intensity=2.5),
                LightConfig(type="ambient", intensity=0.3)
            ]
        )
        
        image_angle = renderer.render(semantic_view_angle)
        save_image(image_angle, "models", f"angle_{angle_name}.png")
    print(f"    ✓ 生成{len(camera_angles)}种视角展示")


# ============================================================
# 测试3: PBR材质测试
# ============================================================
def test_materials():
    """测试PBR材质系统"""
    print_separator("测试3: PBR材质测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#101018")
    )
    renderer = create_renderer(config)
    
    # 3.1 金属度测试 (metallic)
    print("\n  [3.1] 金属度测试")
    metallic_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    metallic_objects = []
    
    for i, metallic in enumerate(metallic_values):
        metallic_objects.append(ObjectData(
            id=f"sphere_metallic_{i}",
            pos=(-4 + i * 2, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.8, 0.2, 0.2, 1.0),
                metallic=metallic,
                roughness=0.3
            )
        ))
    
    metallic_objects.append(ObjectData(
        id="ground",
        pos=(0, 0, 0),
        geometry="plane",
        scale_3d=(12, 1, 6),
        material=Material3D(base_color=(0.2, 0.2, 0.25, 1.0), roughness=0.9)
    ))
    
    semantic_view_metallic = SemanticView(
        view_region={
            "camera_position": [0, 4, 8],
            "camera_target": [0, 0.5, 0],
            "fov": 45
        },
        objects=metallic_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -1, -0.5), intensity=3.0),
            LightConfig(type="point", position=(3, 3, 3), intensity=50.0, color=(1.0, 0.95, 0.9)),
            LightConfig(type="ambient", intensity=0.2)
        ]
    )
    
    image_metallic = renderer.render(semantic_view_metallic)
    save_image(image_metallic, "materials", "metallic_gradient.png")
    print("    ✓ 金属度测试完成")
    
    # 3.2 粗糙度测试 (roughness)
    print("\n  [3.2] 粗糙度测试")
    roughness_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    roughness_objects = []
    
    for i, roughness in enumerate(roughness_values):
        roughness_objects.append(ObjectData(
            id=f"sphere_roughness_{i}",
            pos=(-4 + i * 2, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.9, 0.9, 0.9, 1.0),
                metallic=0.9,
                roughness=roughness
            )
        ))
    
    roughness_objects.append(ObjectData(
        id="ground",
        pos=(0, 0, 0),
        geometry="plane",
        scale_3d=(12, 1, 6),
        material=Material3D(base_color=(0.15, 0.15, 0.2, 1.0), roughness=0.5)
    ))
    
    semantic_view_roughness = SemanticView(
        view_region={
            "camera_position": [0, 4, 8],
            "camera_target": [0, 0.5, 0],
            "fov": 45
        },
        objects=roughness_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -1, -0.5), intensity=3.0),
            LightConfig(type="point", position=(0, 5, 5), intensity=80.0),
            LightConfig(type="ambient", intensity=0.15)
        ]
    )
    
    image_roughness = renderer.render(semantic_view_roughness)
    save_image(image_roughness, "materials", "roughness_gradient.png")
    print("    ✓ 粗糙度测试完成")
    
    # 3.3 自发光测试 (emissive)
    print("\n  [3.3] 自发光测试")
    emissive_objects = [
        # 普通物体
        ObjectData(
            id="normal_sphere",
            pos=(-2, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.3, 0.3, 0.3, 1.0),
                roughness=0.5
            )
        ),
        # 发光物体 - 红色
        ObjectData(
            id="glow_red",
            pos=(0, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.5, 0.1, 0.1, 1.0),
                emissive=(1.0, 0.0, 0.0),
                roughness=0.3
            )
        ),
        # 发光物体 - 绿色
        ObjectData(
            id="glow_green",
            pos=(2, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.1, 0.5, 0.1, 1.0),
                emissive=(0.0, 1.0, 0.0),
                roughness=0.3
            )
        ),
        # 发光物体 - 蓝色
        ObjectData(
            id="glow_blue",
            pos=(4, 0.6, 0),
            geometry="sphere",
            material=Material3D(
                base_color=(0.1, 0.1, 0.5, 1.0),
                emissive=(0.0, 0.0, 1.0),
                roughness=0.3
            )
        ),
        ObjectData(
            id="ground",
            pos=(0, 0, 0),
            geometry="plane",
            scale_3d=(12, 1, 6),
            material=Material3D(base_color=(0.1, 0.1, 0.12, 1.0), roughness=0.9)
        )
    ]
    
    semantic_view_emissive = SemanticView(
        view_region={
            "camera_position": [1, 3, 6],
            "camera_target": [1, 0.5, 0],
            "fov": 50
        },
        objects=emissive_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -1, -1), intensity=0.5),  # 较暗以突出发光
            LightConfig(type="ambient", intensity=0.1)
        ]
    )
    
    image_emissive = renderer.render(semantic_view_emissive)
    save_image(image_emissive, "materials", "emissive_colors.png")
    print("    ✓ 自发光测试完成")
    
    # 3.4 颜色变化测试
    print("\n  [3.4] 基础颜色测试")
    colors = [
        ((1.0, 0.0, 0.0, 1.0), "red"),
        ((0.0, 1.0, 0.0, 1.0), "green"),
        ((0.0, 0.0, 1.0, 1.0), "blue"),
        ((1.0, 1.0, 0.0, 1.0), "yellow"),
        ((1.0, 0.0, 1.0, 1.0), "magenta"),
        ((0.0, 1.0, 1.0, 1.0), "cyan"),
        ((1.0, 0.5, 0.0, 1.0), "orange"),
        ((0.5, 0.0, 1.0, 1.0), "purple"),
    ]
    
    color_objects = []
    for i, (color, name) in enumerate(colors):
        row = i // 4
        col = i % 4
        color_objects.append(ObjectData(
            id=f"sphere_{name}",
            pos=(-3 + col * 2, 0.6 + row * 1.5, -row * 2),
            geometry="sphere",
            material=Material3D(
                base_color=color,
                metallic=0.3,
                roughness=0.4
            )
        ))
    
    color_objects.append(ObjectData(
        id="ground",
        pos=(0, 0, 0),
        geometry="plane",
        scale_3d=(12, 1, 8),
        material=Material3D(base_color=(0.2, 0.2, 0.25, 1.0))
    ))
    
    semantic_view_colors = SemanticView(
        view_region={
            "camera_position": [0, 6, 8],
            "camera_target": [0, 0.5, -1],
            "fov": 50
        },
        objects=color_objects,
        lights=[
            LightConfig(type="directional", direction=(-1, -1.5, -1), intensity=2.5),
            LightConfig(type="ambient", intensity=0.35)
        ]
    )
    
    image_colors = renderer.render(semantic_view_colors)
    save_image(image_colors, "materials", "base_colors.png")
    print("    ✓ 基础颜色测试完成")


# ============================================================
# 测试4: 多光照测试
# ============================================================
def test_lighting():
    """测试多种光源类型"""
    print_separator("测试4: 多光照测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#050508")
    )
    renderer = create_renderer(config)
    
    # 基础场景物体
    def create_base_objects():
        return [
            ObjectData(
                id="center_sphere",
                pos=(0, 0.8, 0),
                geometry="sphere",
                material=Material3D(base_color=(0.9, 0.9, 0.9, 1.0), metallic=0.1, roughness=0.3)
            ),
            ObjectData(
                id="left_box",
                pos=(-2, 0.5, 0),
                geometry="box",
                material=Material3D(base_color=(0.8, 0.3, 0.3, 1.0), roughness=0.5)
            ),
            ObjectData(
                id="right_cylinder",
                pos=(2, 0.5, 0),
                geometry="cylinder",
                material=Material3D(base_color=(0.3, 0.3, 0.8, 1.0), roughness=0.5)
            ),
            ObjectData(
                id="back_cone",
                pos=(0, 0.5, -2),
                geometry="cone",
                material=Material3D(base_color=(0.3, 0.8, 0.3, 1.0), roughness=0.5)
            ),
            ObjectData(
                id="ground",
                pos=(0, 0, 0),
                geometry="plane",
                scale_3d=(10, 1, 10),
                material=Material3D(base_color=(0.3, 0.3, 0.35, 1.0), roughness=0.8)
            )
        ]
    
    view_region = {
        "camera_position": [5, 5, 5],
        "camera_target": [0, 0.5, 0],
        "fov": 50
    }
    
    # 4.1 仅环境光
    print("\n  [4.1] 仅环境光")
    semantic_ambient = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            LightConfig(type="ambient", intensity=0.8, color=(1.0, 1.0, 1.0))
        ]
    )
    image_ambient = renderer.render(semantic_ambient)
    save_image(image_ambient, "lighting", "ambient_only.png")
    print("    ✓ 环境光测试完成")
    
    # 4.2 方向光
    print("\n  [4.2] 方向光")
    semantic_directional = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            LightConfig(type="directional", direction=(-1, -1, -1), intensity=3.0, color=(1.0, 0.95, 0.9)),
            LightConfig(type="ambient", intensity=0.15)
        ]
    )
    image_directional = renderer.render(semantic_directional)
    save_image(image_directional, "lighting", "directional.png")
    print("    ✓ 方向光测试完成")
    
    # 4.3 点光源
    print("\n  [4.3] 点光源")
    semantic_point = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            LightConfig(type="point", position=(0, 3, 0), intensity=100.0, color=(1.0, 0.9, 0.8)),
            LightConfig(type="ambient", intensity=0.1)
        ]
    )
    image_point = renderer.render(semantic_point)
    save_image(image_point, "lighting", "point.png")
    print("    ✓ 点光源测试完成")
    
    # 4.4 聚光灯
    print("\n  [4.4] 聚光灯")
    semantic_spot = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            LightConfig(
                type="spot",
                position=(0, 5, 3),
                direction=(0, -1, -0.5),
                intensity=150.0,
                inner_cone_angle=15,
                outer_cone_angle=30,
                color=(1.0, 1.0, 0.9)
            ),
            LightConfig(type="ambient", intensity=0.08)
        ]
    )
    image_spot = renderer.render(semantic_spot)
    save_image(image_spot, "lighting", "spotlight.png")
    print("    ✓ 聚光灯测试完成")
    
    # 4.5 多光源组合
    print("\n  [4.5] 多光源组合")
    semantic_multi = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            # 主光 - 暖色方向光
            LightConfig(type="directional", direction=(-1, -1, -0.5), intensity=2.0, color=(1.0, 0.9, 0.8)),
            # 补光 - 冷色方向光
            LightConfig(type="directional", direction=(1, -0.5, 0.5), intensity=1.0, color=(0.7, 0.8, 1.0)),
            # 红色点光源
            LightConfig(type="point", position=(-3, 2, 0), intensity=30.0, color=(1.0, 0.2, 0.1)),
            # 蓝色点光源
            LightConfig(type="point", position=(3, 2, 0), intensity=30.0, color=(0.1, 0.3, 1.0)),
            # 环境光
            LightConfig(type="ambient", intensity=0.2)
        ]
    )
    image_multi = renderer.render(semantic_multi)
    save_image(image_multi, "lighting", "multi_lights.png")
    print("    ✓ 多光源组合测试完成")
    
    # 4.6 不同颜色的光
    print("\n  [4.6] 彩色光照")
    semantic_colored = SemanticView(
        view_region=view_region,
        objects=create_base_objects(),
        lights=[
            LightConfig(type="point", position=(-3, 3, 2), intensity=60.0, color=(1.0, 0.0, 0.0)),
            LightConfig(type="point", position=(3, 3, 2), intensity=60.0, color=(0.0, 1.0, 0.0)),
            LightConfig(type="point", position=(0, 3, -3), intensity=60.0, color=(0.0, 0.0, 1.0)),
            LightConfig(type="ambient", intensity=0.1)
        ]
    )
    image_colored = renderer.render(semantic_colored)
    save_image(image_colored, "lighting", "colored_lights.png")
    print("    ✓ 彩色光照测试完成")


# ============================================================
# 测试5: 相机参数测试
# ============================================================
def test_camera_params():
    """测试不同相机参数的效果"""
    print_separator("测试5: 相机参数测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#15151f")
    )
    renderer = create_renderer(config)
    
    # 测试场景
    def create_scene_objects():
        objects = []
        # 中心物体
        objects.append(ObjectData(
            id="center",
            pos=(0, 0.5, 0),
            geometry="box",
            material=Material3D(base_color=(0.8, 0.3, 0.3, 1.0))
        ))
        # 周围物体
        for i in range(6):
            angle = i * 60
            rad = math.radians(angle)
            x = 3 * math.cos(rad)
            z = 3 * math.sin(rad)
            objects.append(ObjectData(
                id=f"sphere_{i}",
                pos=(x, 0.5, z),
                geometry="sphere",
                material=Material3D(base_color=(0.3, 0.5 + i * 0.08, 0.8, 1.0))
            ))
        # 地面
        objects.append(ObjectData(
            id="ground",
            pos=(0, 0, 0),
            geometry="plane",
            scale_3d=(12, 1, 12),
            material=Material3D(base_color=(0.25, 0.25, 0.3, 1.0))
        ))
        return objects
    
    lights = [
        LightConfig(type="directional", direction=(-1, -1, -1), intensity=2.5),
        LightConfig(type="ambient", intensity=0.3)
    ]
    
    # 5.1 不同FOV
    print("\n  [5.1] 不同FOV测试")
    fov_values = [30, 45, 60, 90, 120]
    for fov in fov_values:
        semantic_view = SemanticView(
            view_region={
                "camera_position": [8, 6, 8],
                "camera_target": [0, 0, 0],
                "fov": fov
            },
            objects=create_scene_objects(),
            lights=lights
        )
        image = renderer.render(semantic_view)
        save_image(image, "geometry", f"fov_{fov}.png")
    print(f"    ✓ 生成{len(fov_values)}种FOV效果")
    
    # 5.2 不同相机距离
    print("\n  [5.2] 不同相机距离")
    distances = [5, 10, 15, 20]
    for dist in distances:
        semantic_view = SemanticView(
            view_region={
                "camera_position": [dist, dist * 0.75, dist],
                "camera_target": [0, 0, 0],
                "fov": 50
            },
            objects=create_scene_objects(),
            lights=lights
        )
        image = renderer.render(semantic_view)
        save_image(image, "geometry", f"distance_{dist}.png")
    print(f"    ✓ 生成{len(distances)}种距离效果")
    
    # 5.3 不同相机高度
    print("\n  [5.3] 不同相机高度")
    heights = [2, 5, 10, 15]
    for height in heights:
        semantic_view = SemanticView(
            view_region={
                "camera_position": [8, height, 8],
                "camera_target": [0, 0, 0],
                "fov": 50
            },
            objects=create_scene_objects(),
            lights=lights
        )
        image = renderer.render(semantic_view)
        save_image(image, "geometry", f"height_{height}.png")
    print(f"    ✓ 生成{len(heights)}种高度效果")


# ============================================================
# 测试6: 视野遮罩测试
# ============================================================
def test_view_masks():
    """测试3D渲染的视野遮罩后处理"""
    print_separator("测试6: 视野遮罩测试")
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#1a1a2e")
    )
    renderer = create_renderer(config)
    
    # 创建场景
    objects = [
        ObjectData(
            id="center",
            pos=(0, 0.8, 0),
            geometry="sphere",
            material=Material3D(base_color=(0.9, 0.3, 0.3, 1.0))
        ),
        ObjectData(
            id="left",
            pos=(-2, 0.5, 0),
            geometry="box",
            material=Material3D(base_color=(0.3, 0.9, 0.3, 1.0))
        ),
        ObjectData(
            id="right",
            pos=(2, 0.5, 0),
            geometry="cylinder",
            material=Material3D(base_color=(0.3, 0.3, 0.9, 1.0))
        ),
        ObjectData(
            id="ground",
            pos=(0, 0, 0),
            geometry="plane",
            scale_3d=(10, 1, 10),
            material=Material3D(base_color=(0.25, 0.25, 0.3, 1.0))
        )
    ]
    
    lights = [
        LightConfig(type="directional", direction=(-1, -1, -1), intensity=2.5),
        LightConfig(type="ambient", intensity=0.3)
    ]
    
    base_view = {
        "camera_position": [5, 5, 5],
        "camera_target": [0, 0.5, 0],
        "fov": 50
    }
    
    # 6.1 圆形遮罩
    print("\n  [6.1] 圆形遮罩")
    view_circle = base_view.copy()
    view_circle["shape_type"] = "circle"
    view_circle["center"] = (400, 300)
    view_circle["radius"] = 200
    
    semantic_circle = SemanticView(view_region=view_circle, objects=objects, lights=lights)
    image_circle = renderer.render(semantic_circle)
    save_image(image_circle, "geometry", "mask_circle.png")
    print("    ✓ 圆形遮罩完成")
    
    # 6.2 扇形遮罩
    print("\n  [6.2] 扇形遮罩")
    view_sector = base_view.copy()
    view_sector["shape_type"] = "sector"
    view_sector["center"] = (400, 300)
    view_sector["radius"] = 280
    view_sector["angle_start"] = -45
    view_sector["angle_end"] = 45
    
    semantic_sector = SemanticView(view_region=view_sector, objects=objects, lights=lights)
    image_sector = renderer.render(semantic_sector)
    save_image(image_sector, "geometry", "mask_sector.png")
    print("    ✓ 扇形遮罩完成")
    
    # 6.3 环形遮罩
    print("\n  [6.3] 环形遮罩")
    view_ring = base_view.copy()
    view_ring["shape_type"] = "ring"
    view_ring["center"] = (400, 300)
    view_ring["outer_radius"] = 250
    view_ring["inner_radius"] = 100
    
    semantic_ring = SemanticView(view_region=view_ring, objects=objects, lights=lights)
    image_ring = renderer.render(semantic_ring)
    save_image(image_ring, "geometry", "mask_ring.png")
    print("    ✓ 环形遮罩完成")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有3D渲染器测试"""
    print("\n" + "=" * 70)
    print("  3D渲染器基础功能测试")
    print("=" * 70)
    print("""
测试内容：
  1. 程序几何体测试：box, sphere, cylinder等
  2. 外部模型加载测试：bugatti, tree模型
  3. PBR材质测试：metallic, roughness, emissive等
  4. 多光照测试：ambient, directional, point, spot
  5. 相机参数测试：FOV, 距离, 高度
  6. 视野遮罩测试：圆形, 扇形, 环形
    """)
    
    ensure_output_dirs()
    
    has_3d = check_3d_available()
    
    try:
        test_geometry()
        test_external_models()
        test_materials()
        test_lighting()
        test_camera_params()
        test_view_masks()
        
        print("\n" + "=" * 70)
        print("  所有3D渲染器测试完成！")
        print(f"  输出目录: {OUTPUT_DIR}")
        if not has_3d:
            print("  注意：使用了降级渲染器（PyRender不可用）")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n  测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

