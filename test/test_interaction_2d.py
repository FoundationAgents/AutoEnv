# ============================================================
# 2D模式完整交互测试
# 测试CameraEnv + Camera2D + Renderer2D的完整交互流程
# 包含复杂状态机、观测副作用、动画效果等功能
# ============================================================

"""
测试场景：收集游戏
    - player: 玩家 (使用 2026 new year 蛇年图片，红色蛇形图案，带金色高亮边框)
    - treasure_*: 宝藏（金色皇冠素材）
    - obstacle_*: 障碍物（数字素材 1.png, 2.png, 3.png）
    - decoration_*: 装饰物（默认占位符）
    
状态机：
    - IDLE → SPOTTED（被观测到）
    - SPOTTED → ACTIVE（开始动画）
    - ACTIVE → COLLECTED/DESTROYED
    
副作用：
    - 物体被观测后添加旋转动画
    - 收集物品后产生缩放效果
    - 物体变形：障碍物类型切换
    - z_index动态变化

输出文件夹说明：
    - exploration/  : 基础探索功能测试 - 玩家移动、视野跟随、不同视野形状
    - animation/    : 动画效果测试 - 物体旋转动画、z_index浮动效果
    - effects/      : 特效测试 - 收集动画（缩小消失）、障碍物变形（1→2→3循环）
    - state_machine/: 完整游戏会话 - 50步随机游戏模拟，展示状态机完整流程
"""

import sys
import os
import math
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.env.base_env import CameraEnv
from base.env.base_camera import Camera2D
from base.env.base_observation import ObservationPolicy
from base.env.semantic_view import SemanticView, ObjectData, Transform
from base.env.view_volume import ViewVolume2D
from base.render.renderer_2d import Renderer2D
from base.render.base_renderer import RenderConfig, BackgroundColor


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "interaction", "2d")
TEST_PIC_DIR = os.path.join(os.path.dirname(__file__), "test_pic")


def ensure_output_dirs():
    """确保输出目录存在"""
    subdirs = ["exploration", "animation", "state_machine", "effects"]
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
# 物体状态枚举
# ============================================================
class ObjectState(Enum):
    IDLE = "idle"               # 初始状态
    SPOTTED = "spotted"         # 被发现
    ACTIVE = "active"           # 激活（动画中）
    COLLECTED = "collected"     # 已收集
    DESTROYED = "destroyed"     # 已销毁
    TRANSFORMING = "transforming"  # 变形中


# ============================================================
# 观察策略
# ============================================================
class GameObservationPolicy(ObservationPolicy):
    """游戏观察策略"""
    
    def __call__(self, env_state: Dict[str, Any], t: int):
        return {
            "step": t,
            "player": env_state.get("player", {}),
            "score": env_state.get("score", 0),
            "collected": env_state.get("collected", [])
        }


# ============================================================
# 2D收集游戏环境
# ============================================================
class CollectionGameEnv(CameraEnv):
    """
    2D收集游戏环境
    
    特性：
    - 物体状态机（IDLE → SPOTTED → ACTIVE → COLLECTED）
    - 观测副作用（旋转动画、缩放效果）
    - 物体变形（障碍物类型切换）
    - 动态z_index
    """
    
    def __init__(
        self,
        world_width: int = 1000,
        world_height: int = 800,
        view_width: int = 400,
        view_height: int = 300
    ):
        """
        初始化收集游戏环境
        
        Args:
            world_width: 世界宽度（像素）
            world_height: 世界高度（像素）
            view_width: 视野宽度
            view_height: 视野高度
        """
        # 创建2D相机 - 矩形视野
        camera = Camera2D(
            position=(world_width // 2, world_height // 2),
            width=view_width,
            height=view_height
        )
        
        super().__init__(
            env_id=1,
            obs_policy=GameObservationPolicy(),
            camera=camera
        )
        
        self.world_width = world_width
        self.world_height = world_height
        
        self.configs = {
            "termination": {"max_steps": 100}
        }
        
        # 物体状态追踪
        self._object_states: Dict[str, ObjectState] = {}
        self._object_animations: Dict[str, Dict[str, Any]] = {}  # 动画状态
        self._spotted_time: Dict[str, int] = {}  # 被发现的时间
        
        self._init_world()
    
    def _init_world(self):
        """初始化世界"""
        self._t = 0
        self._object_states.clear()
        self._object_animations.clear()
        self._spotted_time.clear()
        
        player_pos = (self.world_width // 2, self.world_height // 2)
        
        self._state = {
            "player": {
                "pos": player_pos,
                "size": (120, 120),  # 较大尺寸便于识别
                "direction": (1, 0),
                "speed": 50
            },
            "score": 0,
            "collected": [],
            "objects": self._create_game_objects()
        }
        
        # 初始化所有物体状态
        for obj in self._state["objects"]:
            self._object_states[obj["id"]] = ObjectState.IDLE
            self._object_animations[obj["id"]] = {
                "rotation_z": 0,
                "scale": 1.0,
                "z_offset": 0,
            }
    
    def _create_game_objects(self) -> List[Dict[str, Any]]:
        """创建游戏物体"""
        objects = []
        
        # 玩家 - 使用较大尺寸以便突出显示
        objects.append({
            "id": "player",
            "type": "player",
            "pos": (self.world_width // 2, self.world_height // 2),
            "size": (120, 120),  # 比其他物体更大，便于识别
            "z_index": 100,      # 最高层级，始终在最前
        })
        
        # 宝藏 - 分布在世界各处
        treasure_positions = [
            (150, 150), (850, 150), (150, 650), (850, 650),
            (500, 100), (500, 700), (100, 400), (900, 400),
            (300, 300), (700, 500),
        ]
        for i, pos in enumerate(treasure_positions):
            objects.append({
                "id": f"treasure_{i}",
                "type": "treasure",
                "pos": pos,
                "size": (60, 60),
                "z_index": 50,
                "value": 100 + i * 10,
            })
        
        # 障碍物 - 可变形的障碍
        obstacle_configs = [
            {"pos": (250, 200), "variant": 1},
            {"pos": (750, 200), "variant": 2},
            {"pos": (250, 600), "variant": 3},
            {"pos": (750, 600), "variant": 1},
            {"pos": (400, 400), "variant": 2},
            {"pos": (600, 400), "variant": 3},
        ]
        for i, config in enumerate(obstacle_configs):
            objects.append({
                "id": f"obstacle_{i}",
                "type": "obstacle",
                "pos": config["pos"],
                "size": (64, 64),
                "z_index": 30,
                "variant": config["variant"],
                "next_variant": (config["variant"] % 3) + 1,  # 循环变形
            })
        
        # 装饰物（背景元素）
        for i in range(15):
            x = (i * 73) % self.world_width
            y = (i * 53) % self.world_height
            objects.append({
                "id": f"decoration_{i}",
                "type": "decoration",
                "pos": (x, y),
                "size": (32, 32),
                "z_index": 5,
            })
        
        return objects
    
    # ============================================================
    # CameraEnv 抽象方法实现
    # ============================================================
    
    def _dsl_config(self):
        pass
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        """重置环境"""
        self._init_world()
        self.camera.reset()
        player_pos = self._state["player"]["pos"]
        self.camera.position = player_pos
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        return self._state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        self._init_world()
        return "collection_world"
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """状态转移"""
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        if action_name == "move":
            self._handle_move(params)
        elif action_name == "collect":
            self._handle_collect()
        elif action_name == "interact":
            self._handle_interact()
        
        # 更新动画状态
        self._update_animations()
        
        # 同步玩家物体位置
        self._sync_player_object()
        
        # 更新相机跟随
        self.camera.update(self._state["player"])
        
        return self._state
    
    def _handle_move(self, params: Dict[str, Any]):
        """处理移动"""
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        
        player = self._state["player"]
        speed = player["speed"]
        
        new_x = player["pos"][0] + dx * speed
        new_y = player["pos"][1] + dy * speed
        
        # 边界限制
        half_w = player["size"][0] // 2
        half_h = player["size"][1] // 2
        new_x = max(half_w, min(self.world_width - half_w, new_x))
        new_y = max(half_h, min(self.world_height - half_h, new_y))
        
        player["pos"] = (new_x, new_y)
        if dx != 0 or dy != 0:
            player["direction"] = (dx, dy)
    
    def _handle_collect(self):
        """处理收集物品"""
        player_pos = self._state["player"]["pos"]
        player_size = self._state["player"]["size"]
        
        for obj in self._state["objects"]:
            if obj["type"] != "treasure":
                continue
            
            state = self._object_states.get(obj["id"])
            if state not in [ObjectState.SPOTTED, ObjectState.ACTIVE]:
                continue
            
            # 检查距离
            dist = math.sqrt(
                (obj["pos"][0] - player_pos[0]) ** 2 +
                (obj["pos"][1] - player_pos[1]) ** 2
            )
            
            if dist < (player_size[0] + obj["size"][0]) / 2:
                self._object_states[obj["id"]] = ObjectState.COLLECTED
                self._state["collected"].append(obj["id"])
                self._state["score"] += obj.get("value", 100)
                
                # 触发收集动画
                self._object_animations[obj["id"]]["collecting"] = True
                self._last_action_result = f"Collected {obj['id']}"
    
    def _handle_interact(self):
        """处理交互（触发障碍物变形）"""
        player_pos = self._state["player"]["pos"]
        player_size = self._state["player"]["size"]
        
        for obj in self._state["objects"]:
            if obj["type"] != "obstacle":
                continue
            
            state = self._object_states.get(obj["id"])
            if state not in [ObjectState.SPOTTED, ObjectState.ACTIVE]:
                continue
            
            # 检查距离
            dist = math.sqrt(
                (obj["pos"][0] - player_pos[0]) ** 2 +
                (obj["pos"][1] - player_pos[1]) ** 2
            )
            
            if dist < 100:  # 交互范围
                # 开始变形
                self._object_states[obj["id"]] = ObjectState.TRANSFORMING
                old_variant = obj["variant"]
                obj["variant"] = obj["next_variant"]
                obj["next_variant"] = (obj["variant"] % 3) + 1
                
                self._last_action_result = f"Transformed {obj['id']}: {old_variant} → {obj['variant']}"
    
    def _update_animations(self):
        """更新所有物体的动画状态"""
        for obj in self._state["objects"]:
            obj_id = obj["id"]
            state = self._object_states.get(obj_id, ObjectState.IDLE)
            anim = self._object_animations.get(obj_id, {})
            
            if state == ObjectState.SPOTTED:
                # 被发现后开始缓慢旋转
                anim["rotation_z"] = (anim.get("rotation_z", 0) + 2) % 360
                
                # 转换到ACTIVE状态
                spotted_at = self._spotted_time.get(obj_id, self._t)
                if self._t - spotted_at >= 3:  # 3步后变为ACTIVE
                    self._object_states[obj_id] = ObjectState.ACTIVE
            
            elif state == ObjectState.ACTIVE:
                # 激活后快速旋转和脉动
                anim["rotation_z"] = (anim.get("rotation_z", 0) + 8) % 360
                
                # 缩放脉动效果
                phase = self._t * 0.3
                anim["scale"] = 1.0 + 0.1 * math.sin(phase)
                
                # z_index变化（浮起效果）
                anim["z_offset"] = int(5 * math.sin(phase))
            
            elif state == ObjectState.COLLECTED:
                # 收集动画：快速缩小
                current_scale = anim.get("scale", 1.0)
                anim["scale"] = max(0, current_scale - 0.15)
            
            elif state == ObjectState.TRANSFORMING:
                # 变形动画：旋转一圈后恢复
                anim["rotation_z"] = (anim.get("rotation_z", 0) + 30) % 360
                if anim["rotation_z"] < 30:  # 转完一圈
                    self._object_states[obj_id] = ObjectState.ACTIVE
            
            self._object_animations[obj_id] = anim
    
    def _sync_player_object(self):
        """同步玩家物体位置"""
        player_pos = self._state["player"]["pos"]
        for obj in self._state["objects"]:
            if obj["id"] == "player":
                obj["pos"] = player_pos
                break
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """计算奖励"""
        events = []
        reward = 0.0
        
        if self._last_action_result:
            if "Collected" in str(self._last_action_result):
                reward += 100
                events.append("item_collected")
            elif "Transformed" in str(self._last_action_result):
                reward += 10
                events.append("obstacle_transformed")
        
        return reward, events, {"score": self._state["score"]}
    
    def observe_semantic(self):
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega) -> Any:
        return omega
    
    # ============================================================
    # 观测副作用
    # ============================================================
    
    def on_observe(self, view: ViewVolume2D, visible_objects: List[ObjectData]) -> None:
        """
        观测副作用处理
        
        - 将IDLE状态的物体变为SPOTTED
        - 记录发现时间
        """
        for obj in visible_objects:
            obj_id = obj.id
            state = self._object_states.get(obj_id, ObjectState.IDLE)
            
            if state == ObjectState.IDLE:
                self._object_states[obj_id] = ObjectState.SPOTTED
                self._spotted_time[obj_id] = self._t
    
    def _get_all_objects(self) -> List[Dict[str, Any]]:
        """获取所有物体（过滤已收集/销毁的）"""
        objects = []
        for obj in self._state.get("objects", []):
            obj_id = obj["id"]
            state = self._object_states.get(obj_id, ObjectState.IDLE)
            anim = self._object_animations.get(obj_id, {})
            
            # 完全收集（缩小到0）的物体不显示
            if state == ObjectState.COLLECTED and anim.get("scale", 1.0) <= 0:
                continue
            
            objects.append(obj)
        return objects
    
    def _to_object_data(self, obj: Dict[str, Any]) -> ObjectData:
        """转换物体数据，应用动画效果"""
        obj_id = obj["id"]
        obj_type = obj["type"]
        state = self._object_states.get(obj_id, ObjectState.IDLE)
        anim = self._object_animations.get(obj_id, {})
        
        pos = obj["pos"]
        size = obj.get("size", (32, 32))
        base_z_index = obj.get("z_index", 0)
        
        # 构建Transform
        rotation_z = anim.get("rotation_z", 0)
        scale = anim.get("scale", 1.0)
        z_offset = anim.get("z_offset", 0)
        
        transform = None
        if rotation_z != 0 or scale != 1.0:
            transform = Transform(
                rotation_z=rotation_z,
                scale_x=scale,
                scale_y=scale
            )
        
        # 根据障碍物variant确定不同的id（用于素材映射）
        actual_id = obj_id
        if obj_type == "obstacle":
            variant = obj.get("variant", 1)
            actual_id = f"obstacle_{variant}"
        
        return ObjectData(
            id=actual_id,
            pos=pos,
            size=size,
            transform=transform,
            z_index=base_z_index + z_offset
        )
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        visible_count = sum(
            1 for s in self._object_states.values()
            if s != ObjectState.IDLE
        )
        return {
            "step": self._t,
            "score": self._state["score"],
            "collected": len(self._state["collected"]),
            "visible": visible_count,
            "player_pos": self._state["player"]["pos"]
        }


# ============================================================
# 测试函数
# ============================================================

def create_renderer():
    """创建配置好的渲染器"""
    config = RenderConfig(
        asset_path=TEST_PIC_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#1a1a2e"),
        asset_mapping={
            "player": "—Pngtree—2026 new year  year_23419394",
            "treasure_": "—Pngtree—3d texture golden crown_7253018",
            "obstacle_1": "1",
            "obstacle_2": "2",
            "obstacle_3": "3",
        }
    )
    return Renderer2D(config)


def test_basic_exploration():
    """测试基础探索功能"""
    print_separator("测试1: 基础探索功能")
    
    env = CollectionGameEnv(
        world_width=1000,
        world_height=800,
        view_width=400,
        view_height=300
    )
    renderer = create_renderer()
    
    env.reset()
    
    # 初始状态
    print("\n  [初始状态]")
    semantic_view = env.observe()
    game_info = env.get_game_info()
    
    image = renderer.render_with_overlay(
        semantic_view,
        overlay_text=f"Step 0 | Score: {game_info['score']} | Visible: {game_info['visible']}",
        overlay_position=(10, 10)
    )
    save_image(image, "exploration", "step_00_initial.png")
    print(f"    初始可见物体: {len(semantic_view.objects)}")
    
    # 探索移动
    movements = [
        ("右", {"dx": 1, "dy": 0}),
        ("右", {"dx": 1, "dy": 0}),
        ("下", {"dx": 0, "dy": 1}),
        ("下", {"dx": 0, "dy": 1}),
        ("左", {"dx": -1, "dy": 0}),
        ("上", {"dx": 0, "dy": -1}),
        ("右上", {"dx": 1, "dy": -1}),
        ("右下", {"dx": 1, "dy": 1}),
    ]
    
    for i, (name, params) in enumerate(movements, 1):
        state, reward, done, info = env.step({"action": "move", "params": params})
        
        semantic_view = info["semantic_view"]
        game_info = env.get_game_info()
        
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"Step {i}: {name} | Score: {game_info['score']} | Visible: {game_info['visible']}",
            overlay_position=(10, 10)
        )
        save_image(image, "exploration", f"step_{i:02d}_{name}.png")
        
        if i <= 3:
            print(f"    Step {i}: 可见物体 {len(semantic_view.objects)}")
    
    print(f"    ✓ 基础探索测试完成")


def test_state_machine_and_animation():
    """测试状态机和动画效果"""
    print_separator("测试2: 状态机和动画效果")
    
    env = CollectionGameEnv(view_width=500, view_height=400)
    renderer = create_renderer()
    
    env.reset()
    
    # 移动到宝藏附近
    print("\n  [移动到宝藏附近]")
    movements = [
        {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0},
        {"dx": 0, "dy": -1}, {"dx": 0, "dy": -1}, {"dx": 0, "dy": -1},
    ]
    
    for params in movements:
        env.step({"action": "move", "params": params})
    
    # 观察状态变化
    print(f"  玩家位置: {env._state['player']['pos']}")
    
    # 查看附近物体状态
    nearby_states = []
    for obj_id, state in env._object_states.items():
        if state != ObjectState.IDLE:
            nearby_states.append(f"{obj_id}: {state.value}")
    print(f"  发现的物体: {nearby_states[:5]}")
    
    # 记录动画帧
    for frame in range(10):
        # 每帧执行一个小移动来触发动画更新
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        
        semantic_view = env.observe()
        game_info = env.get_game_info()
        
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"Animation Frame {frame} | Visible: {game_info['visible']}",
            overlay_position=(10, 10)
        )
        save_image(image, "animation", f"anim_frame_{frame:02d}.png")
    
    print(f"    ✓ 状态机和动画测试完成，生成10帧动画")


def test_collection_effect():
    """测试收集效果"""
    print_separator("测试3: 收集效果")
    
    env = CollectionGameEnv(view_width=500, view_height=400)
    renderer = create_renderer()
    
    env.reset()
    
    print("\n  [收集物品测试]")
    
    # 直接移动玩家到宝藏位置
    treasure_pos = None
    for obj in env._state["objects"]:
        if obj["type"] == "treasure":
            treasure_pos = obj["pos"]
            break
    
    if treasure_pos:
        env._state["player"]["pos"] = (treasure_pos[0] + 30, treasure_pos[1])
        env._sync_player_object()
        env.camera.update(env._state["player"])
        
        # 先观察让宝藏变为SPOTTED
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        
        print(f"  玩家位置: {env._state['player']['pos']}")
        print(f"  宝藏位置: {treasure_pos}")
        print(f"  收集前分数: {env._state['score']}")
        
        # 渲染收集前
        semantic_view = env.observe()
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"Before Collection | Score: {env._state['score']}",
            overlay_position=(10, 10)
        )
        save_image(image, "effects", "collection_before.png")
        
        # 执行收集
        env.step({"action": "collect", "params": {}})
        print(f"  收集后分数: {env._state['score']}")
        print(f"  已收集: {env._state['collected']}")
        
        # 渲染收集动画
        for frame in range(5):
            env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
            semantic_view = env.observe()
            image = renderer.render_with_overlay(
                semantic_view,
                overlay_text=f"Collection Animation {frame} | Score: {env._state['score']}",
                overlay_position=(10, 10)
            )
            save_image(image, "effects", f"collection_anim_{frame:02d}.png")
    
    print(f"    ✓ 收集效果测试完成")


def test_obstacle_transformation():
    """测试障碍物变形"""
    print_separator("测试4: 障碍物变形")
    
    env = CollectionGameEnv(view_width=500, view_height=400)
    renderer = create_renderer()
    
    env.reset()
    
    print("\n  [障碍物变形测试]")
    
    # 找到障碍物位置
    obstacle_pos = None
    obstacle_obj = None
    for obj in env._state["objects"]:
        if obj["type"] == "obstacle":
            obstacle_pos = obj["pos"]
            obstacle_obj = obj
            break
    
    if obstacle_pos:
        # 移动到障碍物附近
        env._state["player"]["pos"] = (obstacle_pos[0] + 80, obstacle_pos[1])
        env._sync_player_object()
        env.camera.update(env._state["player"])
        
        # 观察障碍物
        for _ in range(4):
            env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        
        print(f"  障碍物位置: {obstacle_pos}")
        print(f"  变形前variant: {obstacle_obj['variant']}")
        
        # 渲染变形前
        semantic_view = env.observe()
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"Before Transform | Variant: {obstacle_obj['variant']}",
            overlay_position=(10, 10)
        )
        save_image(image, "effects", "transform_before.png")
        
        # 执行变形
        env.step({"action": "interact", "params": {}})
        print(f"  变形后variant: {obstacle_obj['variant']}")
        
        # 渲染变形动画
        for frame in range(8):
            env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
            semantic_view = env.observe()
            image = renderer.render_with_overlay(
                semantic_view,
                overlay_text=f"Transform Animation {frame} | Variant: {obstacle_obj['variant']}",
                overlay_position=(10, 10)
            )
            save_image(image, "effects", f"transform_anim_{frame:02d}.png")
    
    print(f"    ✓ 障碍物变形测试完成")


def test_different_view_shapes():
    """测试不同视野形状"""
    print_separator("测试5: 不同视野形状")
    
    renderer = create_renderer()
    
    shapes = [
        ("矩形", lambda c: c.set_rectangular_view(400, 300)),
        ("圆形", lambda c: c.set_circular_view(180, segments=32)),
        ("扇形", lambda c: c.set_sector_view(250, -60, 60, segments=16)),
        ("环形", lambda c: c.set_ring_view(200, 80, segments=32)),
    ]
    
    for shape_name, setup_fn in shapes:
        env = CollectionGameEnv(view_width=400, view_height=300)
        env.reset()
        
        # 设置视野形状
        setup_fn(env.camera)
        
        # 移动探索
        for _ in range(5):
            env.step({"action": "move", "params": {"dx": 1, "dy": 0}})
        
        semantic_view = env.observe()
        game_info = env.get_game_info()
        
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"{shape_name}视野 | Visible: {len(semantic_view.objects)}",
            overlay_position=(10, 10)
        )
        
        safe_name = shape_name.replace(" ", "_")
        save_image(image, "exploration", f"view_shape_{safe_name}.png")
        print(f"    {shape_name}: 可见物体 {len(semantic_view.objects)} 个")
    
    print(f"    ✓ 不同视野形状测试完成")


def test_full_game_session():
    """测试完整游戏会话"""
    print_separator("测试6: 完整游戏会话")
    
    env = CollectionGameEnv(view_width=450, view_height=350)
    renderer = create_renderer()
    
    env.reset()
    
    import random
    random.seed(42)
    
    total_frames = 50
    key_frames = []
    
    print("\n  [完整游戏会话模拟]")
    
    for step in range(total_frames):
        # 随机动作
        action_type = random.choice(["move", "move", "move", "collect", "interact"])
        
        if action_type == "move":
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            action = {"action": "move", "params": {"dx": dx, "dy": dy}}
        elif action_type == "collect":
            action = {"action": "collect", "params": {}}
        else:
            action = {"action": "interact", "params": {}}
        
        state, reward, done, info = env.step(action)
        
        # 每10步保存一帧
        if step % 10 == 0:
            semantic_view = info["semantic_view"]
            game_info = env.get_game_info()
            
            image = renderer.render_with_overlay(
                semantic_view,
                overlay_text=f"Step {step} | Score: {game_info['score']} | Collected: {game_info['collected']}",
                overlay_position=(10, 10)
            )
            save_image(image, "state_machine", f"game_step_{step:03d}.png")
            key_frames.append(step)
        
        if done:
            print(f"  游戏结束于Step {step}")
            break
    
    game_info = env.get_game_info()
    print(f"\n  最终分数: {game_info['score']}")
    print(f"  收集物品: {game_info['collected']}个")
    print(f"  关键帧: {key_frames}")
    print(f"    ✓ 完整游戏会话测试完成")


def test_z_index_dynamics():
    """测试z_index动态变化"""
    print_separator("测试7: z_index动态变化")
    
    env = CollectionGameEnv(view_width=500, view_height=400)
    renderer = create_renderer()
    
    env.reset()
    
    print("\n  [z_index动态变化测试]")
    
    # 移动让物体进入ACTIVE状态
    for _ in range(10):
        env.step({"action": "move", "params": {"dx": 1, "dy": 0}})
    
    # 记录z_index变化
    for frame in range(12):
        env.step({"action": "move", "params": {"dx": 0, "dy": 0}})
        
        semantic_view = env.observe()
        
        # 检查z_index变化
        z_indices = {}
        for obj in semantic_view.objects:
            if "treasure" in obj.id or "obstacle" in obj.id:
                z_indices[obj.id] = obj.z_index
        
        if frame < 3:
            print(f"  Frame {frame} z_indices: {list(z_indices.items())[:3]}")
        
        image = renderer.render_with_overlay(
            semantic_view,
            overlay_text=f"Z-Index Animation Frame {frame}",
            overlay_position=(10, 10)
        )
        save_image(image, "animation", f"z_index_frame_{frame:02d}.png")
    
    print(f"    ✓ z_index动态变化测试完成")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有2D交互测试"""
    print("\n" + "=" * 70)
    print("  2D模式完整交互测试")
    print("=" * 70)
    print("""
素材说明：
  ★ Player (玩家): 2026蛇年图片 - 红色蛇形图案，尺寸120x120，位于画面中心
  ★ Treasure (宝藏): 金色皇冠图片 - 尺寸60x60
  ★ Obstacle (障碍物): 数字图片 1/2/3 - 可变形切换

测试内容：
  1. 基础探索功能
  2. 状态机和动画效果
  3. 收集效果
  4. 障碍物变形
  5. 不同视野形状
  6. 完整游戏会话
  7. z_index动态变化

输出文件夹：
  - exploration/   : 基础探索 - 玩家移动、视野形状
  - animation/     : 动画效果 - 旋转、z_index浮动
  - effects/       : 特效 - 收集动画、障碍物变形
  - state_machine/ : 完整游戏会话模拟
    """)
    
    ensure_output_dirs()
    
    try:
        test_basic_exploration()
        test_state_machine_and_animation()
        test_collection_effect()
        test_obstacle_transformation()
        test_different_view_shapes()
        test_full_game_session()
        test_z_index_dynamics()
        
        print("\n" + "=" * 70)
        print("  所有2D交互测试完成！")
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

