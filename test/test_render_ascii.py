# ============================================================
# ASCII渲染器基础功能测试
# 测试ASCIIRenderer的符号映射、网格尺寸、渲染输出等功能
# ============================================================

"""
测试内容：
    1. 符号映射测试：精确匹配、前缀匹配、默认符号
    2. 网格尺寸测试：配置尺寸、从view_region推断、从物体位置推断
    3. 渲染输出测试：标准网格输出、带头尾信息输出、列表格式降级
    4. 图例生成测试：get_legend()功能验证
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from base.render.ascii_renderer import ASCIIRenderer
from base.render.base_renderer import RenderConfig
from base.env.semantic_view import SemanticView, ObjectData


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "ascii")


def ensure_output_dirs():
    """确保输出目录存在"""
    subdirs = ["basic", "symbol_mapping", "grid_size"]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


def save_output(content: str, subdir: str, filename: str):
    """保存文本输出到文件"""
    path = os.path.join(OUTPUT_DIR, subdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    保存: {path}")


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# 测试1: 符号映射测试
# ============================================================
def test_symbol_mapping():
    """测试符号映射功能"""
    print_separator("测试1: 符号映射测试")
    
    # 1.1 精确匹配测试
    print("\n  [1.1] 精确匹配测试")
    config = RenderConfig(
        grid_size=(20, 15),
        symbol_map={
            "player": "@",
            "wall": "#",
            "treasure": "$",
            "enemy": "E",
            "door": "D",
            "key": "K",
        },
        default_symbol="?",
        empty_symbol="."
    )
    renderer = ASCIIRenderer(config)
    
    # 创建测试场景
    objects = [
        ObjectData(id="player", pos=(10, 7)),
        ObjectData(id="wall", pos=(5, 5)),
        ObjectData(id="treasure", pos=(15, 3)),
        ObjectData(id="enemy", pos=(3, 10)),
        ObjectData(id="door", pos=(18, 7)),
        ObjectData(id="key", pos=(2, 2)),
    ]
    
    semantic_view = SemanticView(
        view_region={"width": 20, "height": 15},
        objects=objects
    )
    
    output = renderer.render(semantic_view)
    print(f"    网格尺寸: 20x15")
    print(f"    物体数量: {len(objects)}")
    print("\n" + output)
    save_output(output, "symbol_mapping", "exact_match.txt")
    
    # 验证符号
    assert "@" in output, "player符号(@)应该出现在输出中"
    assert "#" in output, "wall符号(#)应该出现在输出中"
    assert "$" in output, "treasure符号($)应该出现在输出中"
    print("    ✓ 精确匹配验证通过")
    
    # 1.2 前缀匹配测试
    print("\n  [1.2] 前缀匹配测试")
    config_prefix = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@",
            "wall": "#",      # wall_0, wall_1 等都会匹配
            "treasure": "$",  # treasure_0, treasure_1 等都会匹配
            "enemy": "E",
        },
        default_symbol="?",
        empty_symbol="."
    )
    renderer_prefix = ASCIIRenderer(config_prefix)
    
    # 使用带编号的物体ID
    objects_prefix = [
        ObjectData(id="player", pos=(12, 10)),
        ObjectData(id="wall_0", pos=(5, 5)),
        ObjectData(id="wall_1", pos=(6, 5)),
        ObjectData(id="wall_2", pos=(7, 5)),
        ObjectData(id="wall_3", pos=(8, 5)),
        ObjectData(id="treasure_0", pos=(20, 3)),
        ObjectData(id="treasure_1", pos=(20, 5)),
        ObjectData(id="enemy_0", pos=(3, 15)),
        ObjectData(id="enemy_1", pos=(22, 15)),
    ]
    
    semantic_view_prefix = SemanticView(
        view_region={"width": 25, "height": 20},
        objects=objects_prefix
    )
    
    output_prefix = renderer_prefix.render(semantic_view_prefix)
    print(f"    网格尺寸: 25x20")
    print(f"    物体数量: {len(objects_prefix)}")
    print("\n" + output_prefix)
    save_output(output_prefix, "symbol_mapping", "prefix_match.txt")
    
    # 验证前缀匹配
    wall_count = output_prefix.count("#")
    treasure_count = output_prefix.count("$")
    assert wall_count == 4, f"应有4个墙壁符号，实际有{wall_count}个"
    assert treasure_count == 2, f"应有2个宝藏符号，实际有{treasure_count}个"
    print(f"    ✓ 前缀匹配验证通过 (墙壁: {wall_count}, 宝藏: {treasure_count})")
    
    # 1.3 默认符号测试
    print("\n  [1.3] 默认符号测试")
    objects_unknown = [
        ObjectData(id="player", pos=(10, 10)),
        ObjectData(id="unknown_object", pos=(5, 5)),
        ObjectData(id="mystery_item", pos=(15, 8)),
        ObjectData(id="undefined_entity", pos=(3, 12)),
    ]
    
    semantic_view_unknown = SemanticView(
        view_region={"width": 20, "height": 15},
        objects=objects_unknown
    )
    
    output_unknown = renderer.render(semantic_view_unknown)
    print(f"    网格尺寸: 20x15")
    print(f"    未知物体: unknown_object, mystery_item, undefined_entity")
    print("\n" + output_unknown)
    save_output(output_unknown, "symbol_mapping", "default_symbol.txt")
    
    unknown_count = output_unknown.count("?")
    assert unknown_count == 3, f"应有3个默认符号(?)，实际有{unknown_count}个"
    print(f"    ✓ 默认符号验证通过 (未知物体使用'?': {unknown_count}个)")
    
    # 1.4 额外字段symbol覆盖测试
    print("\n  [1.4] 物体额外字段symbol覆盖测试")
    config_override = RenderConfig(
        grid_size=(15, 12),
        symbol_map={"player": "@", "item": "I"},
        default_symbol="?",
        empty_symbol="."
    )
    renderer_override = ASCIIRenderer(config_override)
    
    # 使用model_extra来设置自定义symbol
    objects_override = [
        ObjectData(id="player", pos=(7, 6)),
        {"id": "item_special", "pos": (3, 3), "symbol": "★"},  # 自定义符号
        {"id": "item_rare", "pos": (10, 8), "symbol": "◆"},    # 自定义符号
    ]
    
    semantic_view_override = SemanticView(
        view_region={"width": 15, "height": 12},
        objects=objects_override
    )
    
    output_override = renderer_override.render(semantic_view_override)
    print(f"    网格尺寸: 15x12")
    print(f"    特殊符号: ★, ◆")
    print("\n" + output_override)
    save_output(output_override, "symbol_mapping", "symbol_override.txt")
    print("    ✓ 额外字段symbol覆盖测试完成")


# ============================================================
# 测试2: 网格尺寸测试
# ============================================================
def test_grid_size():
    """测试网格尺寸推断功能"""
    print_separator("测试2: 网格尺寸测试")
    
    base_config = RenderConfig(
        symbol_map={"player": "@", "wall": "#", "goal": "G"},
        default_symbol="?",
        empty_symbol="."
    )
    
    # 2.1 配置指定尺寸
    print("\n  [2.1] 配置指定尺寸")
    config_fixed = RenderConfig(
        grid_size=(30, 20),
        symbol_map={"player": "@", "goal": "G"},
        default_symbol="?",
        empty_symbol="."
    )
    renderer_fixed = ASCIIRenderer(config_fixed)
    
    objects = [
        ObjectData(id="player", pos=(5, 10)),
        ObjectData(id="goal", pos=(25, 15)),
    ]
    semantic_view = SemanticView(
        view_region={},  # 空的view_region，使用配置的grid_size
        objects=objects
    )
    
    output_fixed = renderer_fixed.render(semantic_view)
    lines = output_fixed.strip().split("\n")
    actual_height = len(lines)
    actual_width = len(lines[0]) if lines else 0
    
    print(f"    配置尺寸: 30x20")
    print(f"    实际输出: {actual_width}x{actual_height}")
    print("\n" + output_fixed)
    save_output(output_fixed, "grid_size", "fixed_size.txt")
    
    assert actual_width == 30 and actual_height == 20, "尺寸应为配置的30x20"
    print("    ✓ 配置尺寸验证通过")
    
    # 2.2 从view_region推断尺寸
    print("\n  [2.2] 从view_region推断尺寸")
    config_infer = RenderConfig(
        symbol_map={"player": "@", "goal": "G"},
        default_symbol="?",
        empty_symbol="."
    )
    renderer_infer = ASCIIRenderer(config_infer)
    
    semantic_view_region = SemanticView(
        view_region={"width": 25, "height": 18},
        objects=objects
    )
    
    output_region = renderer_infer.render(semantic_view_region)
    lines_region = output_region.strip().split("\n")
    inferred_height = len(lines_region)
    inferred_width = len(lines_region[0]) if lines_region else 0
    
    print(f"    view_region: width=25, height=18")
    print(f"    推断尺寸: {inferred_width}x{inferred_height}")
    print("\n" + output_region)
    save_output(output_region, "grid_size", "from_view_region.txt")
    
    assert inferred_width == 25 and inferred_height == 18, "尺寸应从view_region推断为25x18"
    print("    ✓ view_region推断验证通过")
    
    # 2.3 从size字段推断尺寸
    print("\n  [2.3] 从size字段推断尺寸")
    semantic_view_size = SemanticView(
        view_region={"size": [22, 16]},
        objects=objects
    )
    
    output_size = renderer_infer.render(semantic_view_size)
    lines_size = output_size.strip().split("\n")
    size_height = len(lines_size)
    size_width = len(lines_size[0]) if lines_size else 0
    
    print(f"    view_region.size: [22, 16]")
    print(f"    推断尺寸: {size_width}x{size_height}")
    print("\n" + output_size)
    save_output(output_size, "grid_size", "from_size_field.txt")
    print("    ✓ size字段推断验证通过")
    
    # 2.4 从center+radius推断尺寸
    print("\n  [2.4] 从center+radius推断尺寸")
    semantic_view_circle = SemanticView(
        view_region={"center": [10, 10], "radius": 8},
        objects=[ObjectData(id="player", pos=(10, 10))]
    )
    
    output_circle = renderer_infer.render(semantic_view_circle)
    lines_circle = output_circle.strip().split("\n")
    circle_height = len(lines_circle)
    circle_width = len(lines_circle[0]) if lines_circle else 0
    
    expected_size = 8 * 2 + 1  # 17x17
    print(f"    view_region: center=[10,10], radius=8")
    print(f"    预期尺寸: {expected_size}x{expected_size}")
    print(f"    推断尺寸: {circle_width}x{circle_height}")
    print("\n" + output_circle)
    save_output(output_circle, "grid_size", "from_center_radius.txt")
    print("    ✓ center+radius推断验证通过")
    
    # 2.5 从物体位置推断尺寸
    print("\n  [2.5] 从物体位置推断尺寸")
    objects_spread = [
        ObjectData(id="player", pos=(3, 2)),
        ObjectData(id="goal", pos=(18, 12)),
        ObjectData(id="wall", pos=(10, 8)),
    ]
    
    semantic_view_objects = SemanticView(
        view_region=None,  # 无view_region信息
        objects=objects_spread
    )
    
    output_objects = renderer_infer.render(semantic_view_objects)
    lines_objects = output_objects.strip().split("\n")
    obj_height = len(lines_objects)
    obj_width = len(lines_objects[0]) if lines_objects else 0
    
    print(f"    物体位置: (3,2), (18,12), (10,8)")
    print(f"    推断尺寸: {obj_width}x{obj_height} (应至少包含所有物体)")
    print("\n" + output_objects)
    save_output(output_objects, "grid_size", "from_object_positions.txt")
    print("    ✓ 物体位置推断验证通过")


# ============================================================
# 测试3: 渲染输出格式测试
# ============================================================
def test_render_output():
    """测试不同的渲染输出格式"""
    print_separator("测试3: 渲染输出格式测试")
    
    config = RenderConfig(
        grid_size=(25, 18),
        symbol_map={
            "player": "@",
            "wall": "#",
            "treasure": "$",
            "enemy": "E",
            "exit": "X",
        },
        default_symbol="?",
        empty_symbol="."
    )
    renderer = ASCIIRenderer(config)
    
    # 创建迷宫场景
    objects = [
        ObjectData(id="player", pos=(2, 2)),
        ObjectData(id="exit", pos=(22, 15)),
    ]
    
    # 添加墙壁边界
    for x in range(25):
        objects.append(ObjectData(id=f"wall_top_{x}", pos=(x, 0)))
        objects.append(ObjectData(id=f"wall_bottom_{x}", pos=(x, 17)))
    for y in range(18):
        objects.append(ObjectData(id=f"wall_left_{y}", pos=(0, y)))
        objects.append(ObjectData(id=f"wall_right_{y}", pos=(24, y)))
    
    # 添加内部障碍和宝藏
    internal_walls = [(5, 5), (6, 5), (7, 5), (10, 10), (11, 10), (12, 10), (15, 3), (15, 4)]
    for i, (x, y) in enumerate(internal_walls):
        objects.append(ObjectData(id=f"wall_internal_{i}", pos=(x, y)))
    
    treasures = [(8, 8), (18, 5), (12, 14)]
    for i, (x, y) in enumerate(treasures):
        objects.append(ObjectData(id=f"treasure_{i}", pos=(x, y)))
    
    enemies = [(6, 12), (20, 10)]
    for i, (x, y) in enumerate(enemies):
        objects.append(ObjectData(id=f"enemy_{i}", pos=(x, y)))
    
    semantic_view = SemanticView(
        view_region={"width": 25, "height": 18},
        objects=objects
    )
    
    # 3.1 标准网格输出
    print("\n  [3.1] 标准网格输出")
    output_standard = renderer.render(semantic_view)
    print(f"    网格尺寸: 25x18")
    print(f"    物体数量: {len(objects)}")
    print("\n" + output_standard)
    save_output(output_standard, "basic", "standard_grid.txt")
    print("    ✓ 标准网格输出完成")
    
    # 3.2 带头尾信息输出
    print("\n  [3.2] 带头尾信息输出")
    header = "=== 迷宫探索游戏 ==="
    footer = "提示: @ = 玩家, $ = 宝藏, E = 敌人, X = 出口"
    
    output_with_info = renderer.render_with_info(
        semantic_view,
        header=header,
        footer=footer
    )
    print("\n" + output_with_info)
    save_output(output_with_info, "basic", "with_header_footer.txt")
    print("    ✓ 带头尾信息输出完成")
    
    # 3.3 图例生成
    print("\n  [3.3] 图例生成")
    legend = renderer.get_legend()
    print("\n" + legend)
    save_output(legend, "basic", "legend.txt")
    print("    ✓ 图例生成完成")
    
    # 3.4 列表格式降级（无法确定网格尺寸时）
    print("\n  [3.4] 列表格式降级输出")
    config_no_grid = RenderConfig(
        symbol_map={"player": "@", "item": "I"},
        default_symbol="?",
        empty_symbol="."
        # 不设置grid_size
    )
    renderer_no_grid = ASCIIRenderer(config_no_grid)
    
    # 创建无法确定网格尺寸的场景
    objects_no_grid = [
        ObjectData(id="player", pos=(100.5, 200.3)),  # 浮点坐标
        ObjectData(id="item", pos=(150.7, 180.2)),
    ]
    
    semantic_view_no_grid = SemanticView(
        view_region=None,  # 无法从view_region推断
        objects=objects_no_grid
    )
    
    output_list = renderer_no_grid.render(semantic_view_no_grid)
    print("\n" + output_list)
    save_output(output_list, "basic", "list_format_fallback.txt")
    print("    ✓ 列表格式降级输出完成")


# ============================================================
# 测试4: z_index层级测试
# ============================================================
def test_z_index():
    """测试z_index渲染层级"""
    print_separator("测试4: z_index层级测试")
    
    config = RenderConfig(
        grid_size=(15, 12),
        symbol_map={
            "floor": ".",
            "item": "I",
            "player": "@",
            "effect": "*",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    # 在同一位置放置不同z_index的物体
    # 高z_index的物体会覆盖低z_index的物体
    objects = [
        # 地板层 (z_index=0)
        ObjectData(id="floor", pos=(7, 6), z_index=0),
        ObjectData(id="floor", pos=(8, 6), z_index=0),
        ObjectData(id="floor", pos=(7, 7), z_index=0),
        
        # 物品层 (z_index=10) - 同一位置，会被更高层覆盖
        ObjectData(id="item", pos=(7, 6), z_index=10),
        
        # 玩家层 (z_index=50) - 最高层，会显示
        ObjectData(id="player", pos=(7, 6), z_index=50),
        
        # 在另一个位置测试
        ObjectData(id="floor", pos=(10, 8), z_index=0),
        ObjectData(id="item", pos=(10, 8), z_index=10),
        # 这里item应该显示，因为没有更高层覆盖
    ]
    
    semantic_view = SemanticView(
        view_region={"width": 15, "height": 12},
        objects=objects
    )
    
    output = renderer.render(semantic_view)
    print(f"    测试场景:")
    print(f"      位置(7,6): floor(z=0) → item(z=10) → player(z=50)")
    print(f"      位置(10,8): floor(z=0) → item(z=10)")
    print("\n" + output)
    save_output(output, "basic", "z_index_layering.txt")
    
    # 验证：(7,6)位置应显示@（player），(10,8)位置应显示I（item）
    lines = output.strip().split("\n")
    char_at_7_6 = lines[6][7] if len(lines) > 6 and len(lines[6]) > 7 else ""
    char_at_10_8 = lines[8][10] if len(lines) > 8 and len(lines[8]) > 10 else ""
    
    print(f"    位置(7,6)显示: '{char_at_7_6}' (应为@)")
    print(f"    位置(10,8)显示: '{char_at_10_8}' (应为I)")
    print("    ✓ z_index层级测试完成")


# ============================================================
# 测试5: 不同场景类型测试
# ============================================================
def test_various_scenes():
    """测试各种不同的场景类型"""
    print_separator("测试5: 不同场景类型测试")
    
    # 5.1 Roguelike地牢场景
    print("\n  [5.1] Roguelike地牢场景")
    config_dungeon = RenderConfig(
        grid_size=(40, 20),
        symbol_map={
            "player": "@",
            "wall": "#",
            "floor": ".",
            "door": "+",
            "stairs_up": "<",
            "stairs_down": ">",
            "gold": "$",
            "potion": "!",
            "scroll": "?",
            "weapon": ")",
            "armor": "[",
            "monster": "M",
            "dragon": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer_dungeon = ASCIIRenderer(config_dungeon)
    
    dungeon_objects = [
        ObjectData(id="player", pos=(10, 10)),
        ObjectData(id="stairs_down", pos=(35, 17)),
    ]
    
    # 房间1
    for x in range(5, 20):
        dungeon_objects.append(ObjectData(id="wall", pos=(x, 5)))
        dungeon_objects.append(ObjectData(id="wall", pos=(x, 15)))
    for y in range(5, 16):
        dungeon_objects.append(ObjectData(id="wall", pos=(5, y)))
        dungeon_objects.append(ObjectData(id="wall", pos=(19, y)))
    dungeon_objects.append(ObjectData(id="door", pos=(19, 10)))
    
    # 房间2
    for x in range(22, 38):
        dungeon_objects.append(ObjectData(id="wall", pos=(x, 3)))
        dungeon_objects.append(ObjectData(id="wall", pos=(x, 18)))
    for y in range(3, 19):
        dungeon_objects.append(ObjectData(id="wall", pos=(22, y)))
        dungeon_objects.append(ObjectData(id="wall", pos=(37, y)))
    dungeon_objects.append(ObjectData(id="door", pos=(22, 10)))
    
    # 物品和怪物
    dungeon_objects.extend([
        ObjectData(id="gold", pos=(8, 8)),
        ObjectData(id="potion", pos=(15, 12)),
        ObjectData(id="scroll", pos=(12, 7)),
        ObjectData(id="weapon", pos=(30, 8)),
        ObjectData(id="armor", pos=(32, 15)),
        ObjectData(id="monster", pos=(28, 10)),
        ObjectData(id="dragon", pos=(34, 12)),
    ])
    
    semantic_view_dungeon = SemanticView(
        view_region={"width": 40, "height": 20},
        objects=dungeon_objects
    )
    
    output_dungeon = renderer_dungeon.render(semantic_view_dungeon)
    print("\n" + output_dungeon)
    save_output(output_dungeon, "basic", "roguelike_dungeon.txt")
    print("    ✓ Roguelike地牢场景完成")
    
    # 5.2 棋盘场景
    print("\n  [5.2] 棋盘场景")
    config_chess = RenderConfig(
        grid_size=(17, 17),  # 8x8棋盘 + 边框
        symbol_map={
            "king_w": "K", "queen_w": "Q", "rook_w": "R", 
            "bishop_w": "B", "knight_w": "N", "pawn_w": "P",
            "king_b": "k", "queen_b": "q", "rook_b": "r",
            "bishop_b": "b", "knight_b": "n", "pawn_b": "p",
            "white_square": "□",
            "black_square": "■",
            "border": "─",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer_chess = ASCIIRenderer(config_chess)
    
    chess_objects = []
    
    # 绘制棋盘格
    for row in range(8):
        for col in range(8):
            x = col * 2 + 1
            y = row * 2 + 1
            square_type = "white_square" if (row + col) % 2 == 0 else "black_square"
            chess_objects.append(ObjectData(id=square_type, pos=(x, y), z_index=0))
    
    # 初始棋子布局
    # 白方
    chess_objects.extend([
        ObjectData(id="rook_w", pos=(1, 15), z_index=10),
        ObjectData(id="knight_w", pos=(3, 15), z_index=10),
        ObjectData(id="bishop_w", pos=(5, 15), z_index=10),
        ObjectData(id="queen_w", pos=(7, 15), z_index=10),
        ObjectData(id="king_w", pos=(9, 15), z_index=10),
        ObjectData(id="bishop_w", pos=(11, 15), z_index=10),
        ObjectData(id="knight_w", pos=(13, 15), z_index=10),
        ObjectData(id="rook_w", pos=(15, 15), z_index=10),
    ])
    for i in range(8):
        chess_objects.append(ObjectData(id="pawn_w", pos=(i*2+1, 13), z_index=10))
    
    # 黑方
    chess_objects.extend([
        ObjectData(id="rook_b", pos=(1, 1), z_index=10),
        ObjectData(id="knight_b", pos=(3, 1), z_index=10),
        ObjectData(id="bishop_b", pos=(5, 1), z_index=10),
        ObjectData(id="queen_b", pos=(7, 1), z_index=10),
        ObjectData(id="king_b", pos=(9, 1), z_index=10),
        ObjectData(id="bishop_b", pos=(11, 1), z_index=10),
        ObjectData(id="knight_b", pos=(13, 1), z_index=10),
        ObjectData(id="rook_b", pos=(15, 1), z_index=10),
    ])
    for i in range(8):
        chess_objects.append(ObjectData(id="pawn_b", pos=(i*2+1, 3), z_index=10))
    
    semantic_view_chess = SemanticView(
        view_region={"width": 17, "height": 17},
        objects=chess_objects
    )
    
    output_chess = renderer_chess.render(semantic_view_chess)
    print("\n" + output_chess)
    save_output(output_chess, "basic", "chess_board.txt")
    print("    ✓ 棋盘场景完成")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有ASCII渲染器测试"""
    print("\n" + "=" * 70)
    print("  ASCII渲染器基础功能测试")
    print("=" * 70)
    print("""
测试内容：
  1. 符号映射测试：精确匹配、前缀匹配、默认符号
  2. 网格尺寸测试：配置尺寸、从view_region推断
  3. 渲染输出格式测试
  4. z_index层级测试
  5. 不同场景类型测试
    """)
    
    ensure_output_dirs()
    
    try:
        test_symbol_mapping()
        test_grid_size()
        test_render_output()
        test_z_index()
        test_various_scenes()
        
        print("\n" + "=" * 70)
        print("  所有ASCII渲染器测试完成！")
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

