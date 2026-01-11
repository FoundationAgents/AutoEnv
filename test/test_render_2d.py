# ============================================================
# 2D渲染器基础功能测试
# 测试Renderer2D的素材加载、视野形状、变换效果、背景模式等功能
# ============================================================

"""
测试内容：
    1. 基础渲染测试：使用test_pic素材和默认占位符
    2. 视野形状遮罩测试：矩形、圆形、扇形、环形、多边形
    3. Transform变换测试：旋转、缩放、透视效果
    4. 背景模式测试：纯色、图片背景各种模式
    5. 素材映射测试：精确匹配和前缀匹配
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from base.render.renderer_2d import Renderer2D
from base.render.base_renderer import RenderConfig, BackgroundColor, BackgroundImage
from base.env.semantic_view import SemanticView, ObjectData, Transform


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "2d")
TEST_PIC_DIR = os.path.join(os.path.dirname(__file__), "test_pic")


def ensure_output_dirs():
    """确保输出目录存在"""
    subdirs = ["basic", "view_shapes", "transforms", "backgrounds"]
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


# ============================================================
# 测试1: 基础渲染测试
# ============================================================
def test_basic_rendering():
    """测试基础渲染功能"""
    print_separator("测试1: 基础渲染测试")
    
    # 1.1 使用默认占位符渲染
    print("\n  [1.1] 默认占位符渲染")
    config_placeholder = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#2c3e50")
    )
    renderer = Renderer2D(config_placeholder)
    
    # 创建不同类型的物体（无对应素材，使用占位符）
    objects = [
        ObjectData(id="player", pos=(100, 100), size=(64, 64), z_index=100),
        ObjectData(id="enemy_0", pos=(300, 150), size=(48, 48), z_index=50),
        ObjectData(id="enemy_1", pos=(500, 200), size=(48, 48), z_index=50),
        ObjectData(id="wall_0", pos=(50, 300), size=(100, 32), z_index=10),
        ObjectData(id="wall_1", pos=(200, 350), size=(150, 32), z_index=10),
        ObjectData(id="treasure", pos=(600, 400), size=(40, 40), z_index=30),
        ObjectData(id="goal", pos=(700, 500), size=(56, 56), z_index=20),
    ]
    
    semantic_view = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects
    )
    
    image = renderer.render(semantic_view)
    print(f"    画布尺寸: 800x600")
    print(f"    物体数量: {len(objects)}")
    print(f"    输出尺寸: {image.size}")
    save_image(image, "basic", "placeholder_rendering.png")
    print("    ✓ 默认占位符渲染完成")
    
    # 1.2 使用test_pic素材渲染
    print("\n  [1.2] 使用test_pic素材渲染")
    config_assets = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(1024, 768),
        background=BackgroundColor(color="#1a1a2e"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",  # 马素材
            "treasure_": "—Pngtree—3d texture golden crown_7253018",  # 皇冠素材
            "obstacle_1": "1",
            "obstacle_2": "2", 
            "obstacle_3": "3",
        }
    )
    renderer_assets = Renderer2D(config_assets)
    
    objects_with_assets = [
        ObjectData(id="player", pos=(200, 300), size=(128, 128), z_index=100),
        ObjectData(id="treasure_0", pos=(500, 200), size=(80, 80), z_index=50),
        ObjectData(id="treasure_1", pos=(700, 400), size=(80, 80), z_index=50),
        ObjectData(id="obstacle_1", pos=(150, 500), size=(64, 64), z_index=30),
        ObjectData(id="obstacle_2", pos=(400, 550), size=(64, 64), z_index=30),
        ObjectData(id="obstacle_3", pos=(650, 600), size=(64, 64), z_index=30),
        ObjectData(id="unknown_item", pos=(850, 300), size=(48, 48), z_index=20),
    ]
    
    semantic_view_assets = SemanticView(
        view_region={"x": 0, "y": 0, "width": 1024, "height": 768},
        objects=objects_with_assets
    )
    
    image_assets = renderer_assets.render(semantic_view_assets)
    print(f"    画布尺寸: 1024x768")
    print(f"    素材映射: player→马, treasure_*→皇冠, obstacle_N→数字")
    print(f"    输出尺寸: {image_assets.size}")
    save_image(image_assets, "basic", "with_assets.png")
    print("    ✓ test_pic素材渲染完成")
    
    # 1.3 带覆盖文字渲染
    print("\n  [1.3] 带覆盖文字渲染")
    image_overlay = renderer_assets.render_with_overlay(
        semantic_view_assets,
        overlay_text="2D Render Test | Step: 0 | Objects: 7",
        overlay_position=(10, 10),
        overlay_font_size=16
    )
    save_image(image_overlay, "basic", "with_overlay.png")
    print("    ✓ 带覆盖文字渲染完成")
    
    # 1.4 z_index层级渲染
    print("\n  [1.4] z_index层级渲染")
    objects_layered = [
        # 底层 - 大的背景元素
        ObjectData(id="floor", pos=(400, 300), size=(300, 200), z_index=0),
        # 中层 - 部分重叠的物体
        ObjectData(id="treasure_0", pos=(380, 280), size=(100, 100), z_index=50),
        ObjectData(id="obstacle_1", pos=(450, 320), size=(80, 80), z_index=30),
        # 顶层 - 玩家总是在最上面
        ObjectData(id="player", pos=(400, 300), size=(120, 120), z_index=100),
    ]
    
    semantic_view_layered = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_layered
    )
    
    config_layered = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#34495e"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure_": "—Pngtree—3d texture golden crown_7253018",
            "obstacle_1": "1",
        }
    )
    renderer_layered = Renderer2D(config_layered)
    
    image_layered = renderer_layered.render(semantic_view_layered)
    save_image(image_layered, "basic", "z_index_layers.png")
    print("    ✓ z_index层级渲染完成")


# ============================================================
# 测试2: 视野形状遮罩测试
# ============================================================
def test_view_shapes():
    """测试不同视野形状的遮罩效果"""
    print_separator("测试2: 视野形状遮罩测试")
    
    config = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#2d3436"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure_": "—Pngtree—3d texture golden crown_7253018",
            "obstacle_": "1",
        }
    )
    renderer = Renderer2D(config)
    
    # 创建分布在画布各处的物体
    base_objects = [
        ObjectData(id="player", pos=(400, 300), size=(80, 80), z_index=100),
        ObjectData(id="treasure_0", pos=(200, 150), size=(60, 60), z_index=50),
        ObjectData(id="treasure_1", pos=(600, 150), size=(60, 60), z_index=50),
        ObjectData(id="treasure_2", pos=(200, 450), size=(60, 60), z_index=50),
        ObjectData(id="treasure_3", pos=(600, 450), size=(60, 60), z_index=50),
        ObjectData(id="obstacle_0", pos=(100, 300), size=(50, 50), z_index=30),
        ObjectData(id="obstacle_1", pos=(700, 300), size=(50, 50), z_index=30),
        ObjectData(id="obstacle_2", pos=(400, 100), size=(50, 50), z_index=30),
        ObjectData(id="obstacle_3", pos=(400, 500), size=(50, 50), z_index=30),
    ]
    
    # 2.1 矩形视野
    print("\n  [2.1] 矩形视野遮罩")
    semantic_rect = SemanticView(
        view_region={
            "shape_type": "rectangle",
            "x": 150,
            "y": 100,
            "width": 500,
            "height": 400
        },
        objects=base_objects
    )
    image_rect = renderer.render(semantic_rect)
    save_image(image_rect, "view_shapes", "rectangle.png")
    print("    ✓ 矩形视野遮罩完成")
    
    # 2.2 圆形视野
    print("\n  [2.2] 圆形视野遮罩")
    semantic_circle = SemanticView(
        view_region={
            "shape_type": "circle",
            "center": (400, 300),
            "radius": 200
        },
        objects=base_objects
    )
    image_circle = renderer.render(semantic_circle)
    save_image(image_circle, "view_shapes", "circle.png")
    print("    ✓ 圆形视野遮罩完成")
    
    # 2.3 扇形视野
    print("\n  [2.3] 扇形视野遮罩")
    semantic_sector = SemanticView(
        view_region={
            "shape_type": "sector",
            "center": (400, 300),
            "radius": 250,
            "angle_start": -60,
            "angle_end": 60
        },
        objects=base_objects
    )
    image_sector = renderer.render(semantic_sector)
    save_image(image_sector, "view_shapes", "sector.png")
    print("    ✓ 扇形视野遮罩完成")
    
    # 2.4 环形视野
    print("\n  [2.4] 环形视野遮罩")
    semantic_ring = SemanticView(
        view_region={
            "shape_type": "ring",
            "center": (400, 300),
            "outer_radius": 250,
            "inner_radius": 100
        },
        objects=base_objects
    )
    image_ring = renderer.render(semantic_ring)
    save_image(image_ring, "view_shapes", "ring.png")
    print("    ✓ 环形视野遮罩完成")
    
    # 2.5 多边形视野（三角形）
    print("\n  [2.5] 多边形视野遮罩（三角形）")
    semantic_polygon = SemanticView(
        view_region={
            "shape_type": "polygon",
            "vertices": [
                (400, 50),   # 顶点
                (150, 500),  # 左下
                (650, 500),  # 右下
            ]
        },
        objects=base_objects
    )
    image_polygon = renderer.render(semantic_polygon)
    save_image(image_polygon, "view_shapes", "polygon_triangle.png")
    print("    ✓ 多边形视野遮罩完成")
    
    # 2.6 不同半径的圆形视野对比
    print("\n  [2.6] 不同半径圆形视野对比")
    radii = [100, 150, 200, 250]
    for radius in radii:
        semantic_r = SemanticView(
            view_region={
                "shape_type": "circle",
                "center": (400, 300),
                "radius": radius
            },
            objects=base_objects
        )
        image_r = renderer.render(semantic_r)
        save_image(image_r, "view_shapes", f"circle_radius_{radius}.png")
    print(f"    ✓ 生成{len(radii)}种不同半径的圆形视野")
    
    # 2.7 不同角度的扇形视野对比
    print("\n  [2.7] 不同角度扇形视野对比")
    angles = [(0, 45), (-30, 30), (-90, 90), (0, 180)]
    for start, end in angles:
        semantic_a = SemanticView(
            view_region={
                "shape_type": "sector",
                "center": (400, 300),
                "radius": 220,
                "angle_start": start,
                "angle_end": end
            },
            objects=base_objects
        )
        image_a = renderer.render(semantic_a)
        save_image(image_a, "view_shapes", f"sector_{start}_{end}.png")
    print(f"    ✓ 生成{len(angles)}种不同角度的扇形视野")


# ============================================================
# 测试3: Transform变换测试
# ============================================================
def test_transforms():
    """测试各种Transform变换效果"""
    print_separator("测试3: Transform变换测试")
    
    config = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#1e272e"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure": "—Pngtree—3d texture golden crown_7253018",
        }
    )
    renderer = Renderer2D(config)
    
    # 3.1 rotation_z平面旋转测试
    print("\n  [3.1] rotation_z平面旋转测试")
    rotation_angles = [0, 30, 45, 60, 90, 120, 180]
    objects_rotation = []
    
    for i, angle in enumerate(rotation_angles):
        x = 100 + i * 100
        objects_rotation.append(ObjectData(
            id="player",
            pos=(x, 300),
            size=(80, 80),
            transform=Transform(rotation_z=angle),
            z_index=10
        ))
    
    semantic_rotation = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_rotation
    )
    
    image_rotation = renderer.render_with_overlay(
        semantic_rotation,
        overlay_text=f"rotation_z: {rotation_angles}"
    )
    save_image(image_rotation, "transforms", "rotation_z.png")
    print("    ✓ rotation_z平面旋转测试完成")
    
    # 3.2 scale缩放测试
    print("\n  [3.2] scale缩放测试")
    scales = [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (1.5, 0.75), (0.75, 1.5)]
    objects_scale = []
    
    for i, (sx, sy) in enumerate(scales):
        col = i % 3
        row = i // 3
        x = 150 + col * 250
        y = 200 + row * 200
        objects_scale.append(ObjectData(
            id="treasure",
            pos=(x, y),
            size=(60, 60),  # 基础尺寸
            transform=Transform(scale_x=sx, scale_y=sy),
            z_index=10
        ))
    
    semantic_scale = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_scale
    )
    
    image_scale = renderer.render_with_overlay(
        semantic_scale,
        overlay_text=f"scales: {scales}"
    )
    save_image(image_scale, "transforms", "scale.png")
    print("    ✓ scale缩放测试完成")
    
    # 3.3 rotation_x透视效果（上下倾斜）
    print("\n  [3.3] rotation_x透视效果")
    rotation_x_angles = [-45, -30, -15, 0, 15, 30, 45]
    objects_rx = []
    
    for i, angle in enumerate(rotation_x_angles):
        x = 60 + i * 100
        objects_rx.append(ObjectData(
            id="player",
            pos=(x, 300),
            size=(80, 80),
            transform=Transform(rotation_x=angle),
            z_index=10
        ))
    
    semantic_rx = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_rx
    )
    
    image_rx = renderer.render_with_overlay(
        semantic_rx,
        overlay_text=f"rotation_x: {rotation_x_angles}"
    )
    save_image(image_rx, "transforms", "rotation_x_perspective.png")
    print("    ✓ rotation_x透视效果测试完成")
    
    # 3.4 rotation_y透视效果（左右倾斜）
    print("\n  [3.4] rotation_y透视效果")
    rotation_y_angles = [-45, -30, -15, 0, 15, 30, 45]
    objects_ry = []
    
    for i, angle in enumerate(rotation_y_angles):
        x = 60 + i * 100
        objects_ry.append(ObjectData(
            id="player",
            pos=(x, 300),
            size=(80, 80),
            transform=Transform(rotation_y=angle),
            z_index=10
        ))
    
    semantic_ry = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_ry
    )
    
    image_ry = renderer.render_with_overlay(
        semantic_ry,
        overlay_text=f"rotation_y: {rotation_y_angles}"
    )
    save_image(image_ry, "transforms", "rotation_y_perspective.png")
    print("    ✓ rotation_y透视效果测试完成")
    
    # 3.5 组合变换测试
    print("\n  [3.5] 组合变换测试")
    combined_transforms = [
        Transform(rotation_z=45, scale_x=1.2, scale_y=1.2),
        Transform(rotation_x=20, rotation_z=30),
        Transform(rotation_y=25, rotation_z=-15),
        Transform(rotation_x=15, rotation_y=15, rotation_z=30),
        Transform(scale_x=0.8, scale_y=1.5, rotation_z=60),
    ]
    
    objects_combined = []
    for i, transform in enumerate(combined_transforms):
        col = i % 3
        row = i // 3
        x = 150 + col * 250
        y = 200 + row * 250
        objects_combined.append(ObjectData(
            id="treasure",
            pos=(x, y),
            size=(100, 100),
            transform=transform,
            z_index=10
        ))
    
    semantic_combined = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_combined
    )
    
    image_combined = renderer.render_with_overlay(
        semantic_combined,
        overlay_text="Combined transforms: rotation + scale"
    )
    save_image(image_combined, "transforms", "combined.png")
    print("    ✓ 组合变换测试完成")


# ============================================================
# 测试4: 背景模式测试
# ============================================================
def test_backgrounds():
    """测试不同的背景模式"""
    print_separator("测试4: 背景模式测试")
    
    # 测试物体
    test_objects = [
        ObjectData(id="player", pos=(400, 300), size=(100, 100), z_index=100),
        ObjectData(id="treasure", pos=(200, 200), size=(60, 60), z_index=50),
        ObjectData(id="treasure", pos=(600, 400), size=(60, 60), z_index=50),
    ]
    
    semantic_view = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=test_objects
    )
    
    # 4.1 纯色背景测试
    print("\n  [4.1] 纯色背景测试")
    colors = [
        ("#1a1a2e", "dark_blue"),
        ("#2d3436", "charcoal"),
        ("#0984e3", "bright_blue"),
        ("#00b894", "green"),
        ("#d63031", "red"),
        ("#fdcb6e", "yellow"),
    ]
    
    for color_hex, color_name in colors:
        config = RenderConfig(
            asset_path=TEST_PIC_DIR,
            resolution=(800, 600),
            background=BackgroundColor(color=color_hex),
            asset_mapping={
                "player": "—Pngtree—2026 new year  year_23419394",
                "treasure": "—Pngtree—3d texture golden crown_7253018",
            }
        )
        renderer = Renderer2D(config)
        image = renderer.render(semantic_view)
        save_image(image, "backgrounds", f"color_{color_name}.png")
    print(f"    ✓ 生成{len(colors)}种纯色背景")
    
    # 4.2 图片背景测试 - 各种模式
    print("\n  [4.2] 图片背景测试")
    
    # 使用test_pic中的图片作为背景
    bg_image_path = os.path.join(TEST_PIC_DIR, "1.png")
    
    if os.path.exists(bg_image_path):
        modes = ["stretch", "tile", "center", "cover", "contain"]
        
        for mode in modes:
            config_bg = RenderConfig(
                asset_path=TEST_PIC_DIR,
                resolution=(800, 600),
                background=BackgroundImage(path=bg_image_path, mode=mode),
                asset_mapping={
                    "player": "—Pngtree—2026 new year  year_23419394",
                    "treasure": "—Pngtree—3d texture golden crown_7253018",
                }
            )
            renderer_bg = Renderer2D(config_bg)
            image_bg = renderer_bg.render_with_overlay(
                semantic_view,
                overlay_text=f"Background mode: {mode}"
            )
            save_image(image_bg, "backgrounds", f"image_{mode}.png")
        print(f"    ✓ 生成{len(modes)}种图片背景模式")
    else:
        print(f"    ⚠ 背景图片不存在: {bg_image_path}")
    
    # 4.3 透明度背景测试
    print("\n  [4.3] 透明度背景测试")
    config_alpha = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#1a1a2e80"),  # 带透明度
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure": "—Pngtree—3d texture golden crown_7253018",
        }
    )
    renderer_alpha = Renderer2D(config_alpha)
    image_alpha = renderer_alpha.render(semantic_view)
    save_image(image_alpha, "backgrounds", "color_with_alpha.png")
    print("    ✓ 透明度背景测试完成")


# ============================================================
# 测试5: 素材映射测试
# ============================================================
def test_asset_mapping():
    """测试素材映射功能"""
    print_separator("测试5: 素材映射测试")
    
    # 5.1 精确匹配测试
    print("\n  [5.1] 精确匹配测试")
    config_exact = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#2c3e50"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "crown": "—Pngtree—3d texture golden crown_7253018",
            "num1": "1",
            "num2": "2",
            "num3": "3",
        }
    )
    renderer_exact = Renderer2D(config_exact)
    
    objects_exact = [
        ObjectData(id="player", pos=(150, 300), size=(100, 100)),
        ObjectData(id="crown", pos=(350, 300), size=(80, 80)),
        ObjectData(id="num1", pos=(500, 300), size=(60, 60)),
        ObjectData(id="num2", pos=(600, 300), size=(60, 60)),
        ObjectData(id="num3", pos=(700, 300), size=(60, 60)),
    ]
    
    semantic_exact = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_exact
    )
    
    image_exact = renderer_exact.render_with_overlay(
        semantic_exact,
        overlay_text="Exact matching: player, crown, num1, num2, num3"
    )
    save_image(image_exact, "basic", "asset_exact_match.png")
    print("    ✓ 精确匹配测试完成")
    
    # 5.2 前缀匹配测试
    print("\n  [5.2] 前缀匹配测试")
    config_prefix = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#2c3e50"),
        asset_mapping={
            "horse_": "—Pngtree—2026 new year  year_23419394",  # horse_0, horse_1等
            "crown_": "—Pngtree—3d texture golden crown_7253018",  # crown_0, crown_1等
        }
    )
    renderer_prefix = Renderer2D(config_prefix)
    
    objects_prefix = [
        ObjectData(id="horse_0", pos=(100, 200), size=(80, 80)),
        ObjectData(id="horse_1", pos=(100, 350), size=(80, 80)),
        ObjectData(id="horse_2", pos=(100, 500), size=(80, 80)),
        ObjectData(id="crown_gold", pos=(300, 200), size=(70, 70)),
        ObjectData(id="crown_silver", pos=(300, 350), size=(70, 70)),
        ObjectData(id="crown_bronze", pos=(300, 500), size=(70, 70)),
        ObjectData(id="unknown_item", pos=(550, 350), size=(60, 60)),  # 无匹配，用占位符
    ]
    
    semantic_prefix = SemanticView(
        view_region={"x": 0, "y": 0, "width": 800, "height": 600},
        objects=objects_prefix
    )
    
    image_prefix = renderer_prefix.render_with_overlay(
        semantic_prefix,
        overlay_text="Prefix matching: horse_* -> horse, crown_* -> crown"
    )
    save_image(image_prefix, "basic", "asset_prefix_match.png")
    print("    ✓ 前缀匹配测试完成")
    
    # 5.3 缓存测试
    print("\n  [5.3] 素材缓存测试")
    # 加载相同素材多次，验证缓存
    renderer_prefix.clear_cache()
    print(f"    清空缓存后: {len(renderer_prefix._asset_cache)}个素材")
    
    for i in range(5):
        renderer_prefix.load_asset("horse_0")
    print(f"    加载horse_0五次后: {len(renderer_prefix._asset_cache)}个素材（应为1，说明使用了缓存）")
    
    renderer_prefix.load_asset("crown_0")
    renderer_prefix.load_asset("unknown")
    print(f"    加载更多素材后: {len(renderer_prefix._asset_cache)}个素材")
    print("    ✓ 素材缓存测试完成")


# ============================================================
# 测试6: 综合场景测试
# ============================================================
def test_complex_scene():
    """测试复杂的综合场景"""
    print_separator("测试6: 综合场景测试")
    
    config = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(1024, 768),
        background=BackgroundColor(color="#0f0f23"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure_": "—Pngtree—3d texture golden crown_7253018",
            "marker_1": "1",
            "marker_2": "2",
            "marker_3": "3",
        }
    )
    renderer = Renderer2D(config)
    
    objects = []
    
    # 玩家（带旋转）
    objects.append(ObjectData(
        id="player",
        pos=(512, 384),
        size=(120, 120),
        transform=Transform(rotation_z=15, scale_x=1.2, scale_y=1.2),
        z_index=100
    ))
    
    # 围绕玩家的宝藏（带各种变换）
    import math
    for i in range(8):
        angle = i * 45
        rad = math.radians(angle)
        radius = 200
        x = 512 + radius * math.cos(rad)
        y = 384 + radius * math.sin(rad)
        
        objects.append(ObjectData(
            id=f"treasure_{i}",
            pos=(x, y),
            size=(60, 60),
            transform=Transform(rotation_z=angle, scale_x=0.8 + i * 0.05, scale_y=0.8 + i * 0.05),
            z_index=50 + i
        ))
    
    # 标记物（带透视效果）
    markers = [
        ("marker_1", (150, 150), Transform(rotation_x=20)),
        ("marker_2", (870, 150), Transform(rotation_y=20)),
        ("marker_3", (512, 650), Transform(rotation_x=-15, rotation_y=-15)),
    ]
    
    for marker_id, pos, transform in markers:
        objects.append(ObjectData(
            id=marker_id,
            pos=pos,
            size=(80, 80),
            transform=transform,
            z_index=30
        ))
    
    # 背景装饰物
    for i in range(20):
        x = (i * 50) % 1024
        y = (i * 37) % 768
        objects.append(ObjectData(
            id=f"bg_element_{i}",
            pos=(x, y),
            size=(30, 30),
            z_index=5
        ))
    
    semantic_view = SemanticView(
        view_region={
            "shape_type": "circle",
            "center": (512, 384),
            "radius": 350
        },
        objects=objects
    )
    
    image = renderer.render_with_overlay(
        semantic_view,
        overlay_text=f"Complex Scene | Objects: {len(objects)} | Circular View"
    )
    save_image(image, "basic", "complex_scene.png")
    print(f"    ✓ 综合场景测试完成 (物体数量: {len(objects)})")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有2D渲染器测试"""
    print("\n" + "=" * 70)
    print("  2D渲染器基础功能测试")
    print("=" * 70)
    print("""
测试内容：
  1. 基础渲染测试：占位符、素材加载、覆盖文字
  2. 视野形状遮罩测试：矩形、圆形、扇形、环形、多边形
  3. Transform变换测试：旋转、缩放、透视
  4. 背景模式测试：纯色、图片背景
  5. 素材映射测试：精确匹配、前缀匹配
  6. 综合场景测试
    """)
    
    ensure_output_dirs()
    
    try:
        test_basic_rendering()
        test_view_shapes()
        test_transforms()
        test_backgrounds()
        test_asset_mapping()
        test_complex_scene()
        
        print("\n" + "=" * 70)
        print("  所有2D渲染器测试完成！")
        print(f"  输出目录: {OUTPUT_DIR}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n  测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

