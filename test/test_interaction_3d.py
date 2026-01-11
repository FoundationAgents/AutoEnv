# ============================================================
# 3D模式完整交互测试
# 测试CameraEnv + Camera3D + Renderer3D的完整交互流程
# 包含复杂状态机、材质变化、光照动态、物体生成/销毁等功能
# ============================================================

"""
测试场景：3D探索游戏
    - player: 玩家（球体）
    - vehicle: 载具（bugatti模型）
    - tree_*: 树木（Tree1模型）
    - collectible_*: 可收集物（程序几何体）
    
状态机：
    - HIDDEN → REVEALED（进入视野）
    - REVEALED → INTERACTED（接近时）
    - INTERACTED → TRANSFORMED（触发变化）
    
副作用：
    - 被观测后改变材质（roughness/metallic变化）
    - 物体生成/销毁效果
    - 几何体类型切换（box → sphere）
    - 光照动态变化（日夜循环）
    - 材质动画（发光效果脉动）
"""

import sys
import os
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.env.base_env import CameraEnv
from base.env.base_camera import Camera3D
from base.env.base_observation import ObservationPolicy
from base.env.semantic_view import SemanticView, ObjectData, Transform, Material3D, LightConfig
from base.env.view_volume import ViewVolume3D
from base.render.renderer_3d import create_renderer as create_3d_renderer, PYRENDER_AVAILABLE
from base.render.base_renderer import RenderConfig, BackgroundColor


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "interaction", "3d")
TEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), "test_models")


def ensure_output_dirs():
    """确保输出目录存在"""
    subdirs = ["exploration", "materials", "lighting", "transforms", "effects"]
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
    HIDDEN = "hidden"           # 未被发现
    REVEALED = "revealed"       # 已显示
    INTERACTED = "interacted"   # 已交互
    TRANSFORMED = "transformed" # 已变形
    COLLECTED = "collected"     # 已收集
    DESTROYED = "destroyed"     # 已销毁


# ============================================================
# 观察策略
# ============================================================
class World3DObservationPolicy(ObservationPolicy):
    """3D世界观察策略"""
    
    def __call__(self, env_state: Dict[str, Any], t: int):
        return {
            "step": t,
            "player": env_state.get("player", {}),
            "score": env_state.get("score", 0),
            "time_of_day": env_state.get("time_of_day", 0.5)
        }


# ============================================================
# 3D探索游戏环境
# ============================================================
class Exploration3DEnv(CameraEnv):
    """
    3D探索游戏环境
    
    特性：
    - 物体状态机
    - 材质动态变化
    - 日夜循环光照
    - 几何体变形
    - 发光效果脉动
    """
    
    def __init__(
        self,
        world_size: float = 50.0,
        fov: float = 60.0,
        view_distance: float = 30.0
    ):
        """
        初始化3D探索环境
        
        Args:
            world_size: 世界尺寸（单位）
            fov: 相机视野角度
            view_distance: 视野距离
        """
        # 创建3D相机
        camera = Camera3D(
            position=(0, 5, 20),
            direction=(0, -0.2, -1),
            fov=fov,
            aspect=800/600,
            near=0.1,
            far=view_distance * 2
        )
        
        super().__init__(
            env_id=1,
            obs_policy=World3DObservationPolicy(),
            camera=camera
        )
        
        self.world_size = world_size
        self.view_distance = view_distance
        
        self.configs = {
            "termination": {"max_steps": 200}
        }
        
        # 物体状态追踪
        self._object_states: Dict[str, ObjectState] = {}
        self._object_data: Dict[str, Dict[str, Any]] = {}  # 动态数据
        self._revealed_time: Dict[str, int] = {}  # 发现时间
        
        # 环境时间（用于日夜循环和动画）
        self._time_of_day = 0.5  # 0=午夜, 0.25=日出, 0.5=正午, 0.75=日落
        
        self._init_world()
    
    def _init_world(self):
        """初始化世界"""
        self._t = 0
        self._object_states.clear()
        self._object_data.clear()
        self._revealed_time.clear()
        self._time_of_day = 0.5
        
        self._state = {
            "player": {
                "pos": (0, 1, 15),
                "direction": (0, 0, -1),
                "speed": 2.0
            },
            "score": 0,
            "collected": [],
            "time_of_day": self._time_of_day,
            "objects": self._create_world_objects()
        }
        
        # 初始化物体状态
        for obj in self._state["objects"]:
            self._object_states[obj["id"]] = ObjectState.HIDDEN
            self._object_data[obj["id"]] = {
                "rotation": (0, 0, 0),
                "emissive_phase": 0,
                "material_transition": 0,
            }
    
    def _create_world_objects(self) -> List[Dict[str, Any]]:
        """创建世界物体"""
        objects = []
        
        # 玩家（球体）
        objects.append({
            "id": "player",
            "type": "player",
            "pos": (0, 1, 15),
            "geometry": "sphere",
            "scale_3d": (0.8, 0.8, 0.8),
            "material": {
                "base_color": (0.2, 0.8, 0.3, 1.0),
                "metallic": 0.3,
                "roughness": 0.4
            }
        })
        
        # 载具（bugatti模型）
        objects.append({
            "id": "vehicle",
            "type": "vehicle",
            "pos": (5, 0, 5),
            "model_path": os.path.join(TEST_MODELS_DIR, "test_bugatti.obj"),
            "scale_3d": (1.0, 1.0, 1.0),
            "rotation": (0, 45, 0),
            "material": {
                "base_color": (0.8, 0.1, 0.1, 1.0),
                "metallic": 0.9,
                "roughness": 0.2
            }
        })
        
        # 树木
        tree_positions = [
            (-8, 0, -5), (-5, 0, -10), (8, 0, -8),
            (10, 0, 0), (-10, 0, 5), (6, 0, -15),
        ]
        for i, pos in enumerate(tree_positions):
            objects.append({
                "id": f"tree_{i}",
                "type": "tree",
                "pos": pos,
                "model_path": os.path.join(TEST_MODELS_DIR, "Tree1_test.obj"),
                "scale_3d": (0.3 + i * 0.05, 0.3 + i * 0.05, 0.3 + i * 0.05),
                "rotation": (0, i * 60, 0),
            })
        
        # 可收集物（程序几何体）
        collectible_configs = [
            {"pos": (-3, 1, 0), "geometry": "box", "color": (1.0, 0.8, 0.0, 1.0)},      # 金色方块
            {"pos": (3, 1, -5), "geometry": "sphere", "color": (0.0, 0.8, 1.0, 1.0)},   # 青色球
            {"pos": (0, 1, -10), "geometry": "cylinder", "color": (1.0, 0.0, 0.8, 1.0)}, # 粉色圆柱
            {"pos": (-6, 1, -8), "geometry": "cone", "color": (0.8, 1.0, 0.0, 1.0)},    # 黄绿锥体
            {"pos": (8, 1, -12), "geometry": "torus", "color": (1.0, 0.4, 0.0, 1.0)},   # 橙色环
        ]
        for i, config in enumerate(collectible_configs):
            objects.append({
                "id": f"collectible_{i}",
                "type": "collectible",
                "pos": config["pos"],
                "geometry": config["geometry"],
                "original_geometry": config["geometry"],
                "scale_3d": (0.5, 0.5, 0.5),
                "material": {
                    "base_color": config["color"],
                    "metallic": 0.5,
                    "roughness": 0.3,
                    "emissive": (0, 0, 0)
                }
            })
        
        # 地面
        objects.append({
            "id": "ground",
            "type": "ground",
            "pos": (0, -0.05, 0),
            "geometry": "plane",
            "scale_3d": (self.world_size, 1, self.world_size),
            "material": {
                "base_color": (0.15, 0.35, 0.15, 1.0),  # 草地绿
                "roughness": 0.95
            }
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
        self.camera.position = (player_pos[0], player_pos[1] + 4, player_pos[2] + 10)
        self.camera.look_at((player_pos[0], player_pos[1], player_pos[2] - 5))
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        return self._state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        self._init_world()
        return "exploration_3d_world"
    
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
        
        # 更新时间（日夜循环）
        self._update_time()
        
        # 更新物体动画
        self._update_object_animations()
        
        # 同步玩家物体
        self._sync_player_object()
        
        # 更新相机跟随
        self._update_camera()
        
        return self._state
    
    def _handle_move(self, params: Dict[str, Any]):
        """处理移动"""
        dx = params.get("dx", 0)
        dz = params.get("dz", 0)
        
        player = self._state["player"]
        speed = player["speed"]
        
        new_x = player["pos"][0] + dx * speed
        new_z = player["pos"][2] + dz * speed
        
        # 边界限制
        half_world = self.world_size / 2
        new_x = max(-half_world, min(half_world, new_x))
        new_z = max(-half_world, min(half_world, new_z))
        
        player["pos"] = (new_x, player["pos"][1], new_z)
        
        if dx != 0 or dz != 0:
            # 归一化方向
            length = math.sqrt(dx * dx + dz * dz)
            player["direction"] = (dx / length, 0, dz / length)
    
    def _handle_collect(self):
        """处理收集物品"""
        player_pos = self._state["player"]["pos"]
        
        for obj in self._state["objects"]:
            if obj["type"] != "collectible":
                continue
            
            state = self._object_states.get(obj["id"])
            if state not in [ObjectState.REVEALED, ObjectState.INTERACTED]:
                continue
            
            # 检查距离
            dist = math.sqrt(
                (obj["pos"][0] - player_pos[0]) ** 2 +
                (obj["pos"][2] - player_pos[2]) ** 2
            )
            
            if dist < 3.0:
                self._object_states[obj["id"]] = ObjectState.COLLECTED
                self._state["collected"].append(obj["id"])
                self._state["score"] += 100
                self._last_action_result = f"Collected {obj['id']}"
    
    def _handle_interact(self):
        """处理交互（几何体变形）"""
        player_pos = self._state["player"]["pos"]
        
        for obj in self._state["objects"]:
            if obj["type"] != "collectible":
                continue
            
            state = self._object_states.get(obj["id"])
            if state not in [ObjectState.REVEALED, ObjectState.INTERACTED]:
                continue
            
            # 检查距离
            dist = math.sqrt(
                (obj["pos"][0] - player_pos[0]) ** 2 +
                (obj["pos"][2] - player_pos[2]) ** 2
            )
            
            if dist < 5.0:
                self._object_states[obj["id"]] = ObjectState.TRANSFORMED
                
                # 几何体变形
                geometry_cycle = ["box", "sphere", "cylinder", "cone", "torus"]
                current_geom = obj.get("geometry", "box")
                current_idx = geometry_cycle.index(current_geom) if current_geom in geometry_cycle else 0
                new_geom = geometry_cycle[(current_idx + 1) % len(geometry_cycle)]
                obj["geometry"] = new_geom
                
                self._last_action_result = f"Transformed {obj['id']}: {current_geom} → {new_geom}"
    
    def _update_time(self):
        """更新时间（日夜循环）"""
        # 每100步完成一个日夜循环
        self._time_of_day = (self._time_of_day + 0.01) % 1.0
        self._state["time_of_day"] = self._time_of_day
    
    def _update_object_animations(self):
        """更新物体动画"""
        for obj in self._state["objects"]:
            obj_id = obj["id"]
            state = self._object_states.get(obj_id, ObjectState.HIDDEN)
            data = self._object_data.get(obj_id, {})
            
            if state == ObjectState.REVEALED:
                # 缓慢旋转
                rx, ry, rz = data.get("rotation", (0, 0, 0))
                data["rotation"] = (rx, (ry + 2) % 360, rz)
                
                # 材质过渡
                data["material_transition"] = min(1.0, data.get("material_transition", 0) + 0.1)
            
            elif state == ObjectState.INTERACTED:
                # 快速旋转
                rx, ry, rz = data.get("rotation", (0, 0, 0))
                data["rotation"] = (rx, (ry + 10) % 360, rz)
                
                # 发光脉动
                data["emissive_phase"] = (data.get("emissive_phase", 0) + 0.2) % (2 * math.pi)
            
            elif state == ObjectState.TRANSFORMED:
                # 旋转后恢复
                rx, ry, rz = data.get("rotation", (0, 0, 0))
                data["rotation"] = ((rx + 5) % 360, (ry + 15) % 360, (rz + 3) % 360)
                
                # 一段时间后恢复到INTERACTED
                revealed_at = self._revealed_time.get(obj_id, self._t)
                if self._t - revealed_at > 5:
                    self._object_states[obj_id] = ObjectState.INTERACTED
            
            self._object_data[obj_id] = data
    
    def _sync_player_object(self):
        """同步玩家物体位置"""
        player_pos = self._state["player"]["pos"]
        for obj in self._state["objects"]:
            if obj["id"] == "player":
                obj["pos"] = player_pos
                break
    
    def _update_camera(self):
        """更新相机跟随"""
        player = self._state["player"]
        player_pos = player["pos"]
        
        # 相机在玩家后上方
        cam_x = player_pos[0]
        cam_y = player_pos[1] + 6
        cam_z = player_pos[2] + 12
        
        self.camera.position = (cam_x, cam_y, cam_z)
        self.camera.look_at((player_pos[0], player_pos[1], player_pos[2] - 5))
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """计算奖励"""
        events = []
        reward = 0.0
        
        if self._last_action_result:
            if "Collected" in str(self._last_action_result):
                reward += 100
                events.append("item_collected")
            elif "Transformed" in str(self._last_action_result):
                reward += 20
                events.append("item_transformed")
        
        return reward, events, {"score": self._state["score"]}
    
    def observe_semantic(self):
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega) -> Any:
        return omega
    
    # ============================================================
    # 观测副作用
    # ============================================================
    
    def on_observe(self, view: ViewVolume3D, visible_objects: List[ObjectData]) -> None:
        """
        观测副作用处理
        
        - HIDDEN → REVEALED
        - 记录发现时间
        - 接近时 REVEALED → INTERACTED
        """
        player_pos = self._state["player"]["pos"]
        
        for obj in visible_objects:
            obj_id = obj.id
            state = self._object_states.get(obj_id, ObjectState.HIDDEN)
            
            if state == ObjectState.HIDDEN:
                self._object_states[obj_id] = ObjectState.REVEALED
                self._revealed_time[obj_id] = self._t
            
            # 接近时激活
            elif state == ObjectState.REVEALED:
                # 获取原始物体位置
                for orig_obj in self._state["objects"]:
                    if orig_obj["id"] == obj_id:
                        dist = math.sqrt(
                            (orig_obj["pos"][0] - player_pos[0]) ** 2 +
                            (orig_obj["pos"][2] - player_pos[2]) ** 2
                        )
                        if dist < 8.0:
                            self._object_states[obj_id] = ObjectState.INTERACTED
                        break
    
    def _get_all_objects(self) -> List[Dict[str, Any]]:
        """获取所有物体（过滤已收集的）"""
        objects = []
        for obj in self._state.get("objects", []):
            obj_id = obj["id"]
            state = self._object_states.get(obj_id, ObjectState.HIDDEN)
            
            if state == ObjectState.COLLECTED:
                continue
            
            objects.append(obj)
        return objects
    
    def _to_object_data(self, obj: Dict[str, Any]) -> ObjectData:
        """转换物体数据，应用动画和材质变化"""
        obj_id = obj["id"]
        obj_type = obj["type"]
        state = self._object_states.get(obj_id, ObjectState.HIDDEN)
        data = self._object_data.get(obj_id, {})
        
        # 基础数据
        pos = obj["pos"]
        geometry = obj.get("geometry")
        model_path = obj.get("model_path")
        scale_3d = obj.get("scale_3d")
        
        # 构建Transform（应用动画旋转）
        rotation = data.get("rotation", (0, 0, 0))
        base_rotation = obj.get("rotation", (0, 0, 0))
        final_rotation = (
            base_rotation[0] + rotation[0],
            base_rotation[1] + rotation[1],
            base_rotation[2] + rotation[2]
        )
        
        transform = None
        if any(r != 0 for r in final_rotation):
            transform = Transform(
                rotation_x=final_rotation[0],
                rotation_y=final_rotation[1],
                rotation_z=final_rotation[2]
            )
        
        # 构建Material（应用状态变化）
        material = None
        base_material = obj.get("material", {})
        
        if state in [ObjectState.REVEALED, ObjectState.INTERACTED, ObjectState.TRANSFORMED]:
            # 复制基础材质
            mat_data = dict(base_material)
            
            if state == ObjectState.INTERACTED:
                # 增加金属感和发光
                mat_data["metallic"] = min(1.0, base_material.get("metallic", 0) + 0.3)
                mat_data["roughness"] = max(0.1, base_material.get("roughness", 0.5) - 0.2)
                
                # 发光脉动
                phase = data.get("emissive_phase", 0)
                intensity = 0.3 + 0.3 * math.sin(phase)
                base_color = base_material.get("base_color", (1, 1, 1, 1))
                mat_data["emissive"] = (
                    base_color[0] * intensity,
                    base_color[1] * intensity,
                    base_color[2] * intensity
                )
            
            elif state == ObjectState.TRANSFORMED:
                # 强烈发光
                base_color = base_material.get("base_color", (1, 1, 1, 1))
                mat_data["emissive"] = (base_color[0], base_color[1], base_color[2])
                mat_data["metallic"] = 1.0
            
            material = Material3D(**mat_data)
        elif base_material:
            material = Material3D(**base_material)
        
        return ObjectData(
            id=obj_id,
            pos=pos,
            geometry=geometry,
            model_path=model_path,
            scale_3d=scale_3d,
            transform=transform,
            material=material
        )
    
    def get_dynamic_lights(self) -> List[LightConfig]:
        """获取动态光照（日夜循环）"""
        t = self._time_of_day
        
        # 太阳位置（绕场景旋转）
        sun_angle = t * 2 * math.pi
        sun_height = math.sin(sun_angle)  # -1 到 1
        sun_x = math.cos(sun_angle) * 10
        sun_y = max(0.1, sun_height * 10 + 5)  # 保持在地面以上
        sun_z = -10
        
        # 太阳颜色（日出日落偏暖）
        if 0.2 < t < 0.3 or 0.7 < t < 0.8:  # 日出日落
            sun_color = (1.0, 0.7, 0.4)
            sun_intensity = 2.0
        elif 0.3 <= t <= 0.7:  # 白天
            sun_color = (1.0, 0.98, 0.95)
            sun_intensity = 3.0
        else:  # 夜晚
            sun_color = (0.3, 0.35, 0.5)
            sun_intensity = 0.5
        
        # 环境光（夜晚偏蓝）
        if t < 0.25 or t > 0.75:  # 夜晚
            ambient_color = (0.1, 0.12, 0.2)
            ambient_intensity = 0.3
        else:  # 白天
            ambient_color = (0.4, 0.45, 0.5)
            ambient_intensity = 0.4
        
        return [
            LightConfig(
                type="directional",
                direction=(-sun_x, -sun_y, sun_z),
                color=sun_color,
                intensity=sun_intensity
            ),
            LightConfig(
                type="ambient",
                color=ambient_color,
                intensity=ambient_intensity
            )
        ]
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        revealed_count = sum(
            1 for s in self._object_states.values()
            if s not in [ObjectState.HIDDEN, ObjectState.COLLECTED]
        )
        return {
            "step": self._t,
            "score": self._state["score"],
            "collected": len(self._state["collected"]),
            "revealed": revealed_count,
            "time_of_day": self._time_of_day,
            "player_pos": self._state["player"]["pos"]
        }


# ============================================================
# 测试函数
# ============================================================

def create_default_renderer():
    """创建默认配置的渲染器"""
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    return create_3d_renderer(config)


def test_basic_3d_exploration():
    """测试基础3D探索"""
    print_separator("测试1: 基础3D探索")
    
    env = Exploration3DEnv(world_size=50, fov=60, view_distance=40)
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    # 初始状态
    print("\n  [初始状态]")
    semantic_view = env.observe()
    game_info = env.get_game_info()
    
    # 添加动态光照
    semantic_view.lights = env.get_dynamic_lights()
    
    image = renderer.render(semantic_view)
    save_image(image, "exploration", "step_00_initial.png")
    print(f"    可见物体: {len(semantic_view.objects)}")
    
    # 探索移动
    movements = [
        ("前进", {"dx": 0, "dz": -1}),
        ("前进", {"dx": 0, "dz": -1}),
        ("左移", {"dx": -1, "dz": 0}),
        ("前进", {"dx": 0, "dz": -1}),
        ("右移", {"dx": 1, "dz": 0}),
        ("右移", {"dx": 1, "dz": 0}),
        ("后退", {"dx": 0, "dz": 1}),
    ]
    
    for i, (name, params) in enumerate(movements, 1):
        state, reward, done, info = env.step({"action": "move", "params": params})
        
        semantic_view = info["semantic_view"]
        semantic_view.lights = env.get_dynamic_lights()
        game_info = env.get_game_info()
        
        image = renderer.render(semantic_view)
        save_image(image, "exploration", f"step_{i:02d}_{name}.png")
        
        if i <= 3:
            print(f"    Step {i}: 可见 {len(semantic_view.objects)}, 位置 {game_info['player_pos']}")
    
    print(f"    ✓ 基础3D探索测试完成")


def test_material_changes():
    """测试材质动态变化"""
    print_separator("测试2: 材质动态变化")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    print("\n  [材质变化测试]")
    
    # 移动接近收集物
    for _ in range(5):
        env.step({"action": "move", "params": {"dx": 0, "dz": -1}})
    
    # 记录材质变化
    for frame in range(15):
        env.step({"action": "move", "params": {"dx": 0, "dz": 0}})
        
        semantic_view = env.observe()
        semantic_view.lights = env.get_dynamic_lights()
        
        # 检查状态
        states = {k: v.value for k, v in env._object_states.items() if "collectible" in k}
        
        if frame < 3:
            print(f"    Frame {frame}: {list(states.items())[:2]}")
        
        image = renderer.render(semantic_view)
        save_image(image, "materials", f"material_frame_{frame:02d}.png")
    
    print(f"    ✓ 材质动态变化测试完成")


def test_day_night_cycle():
    """测试日夜循环"""
    print_separator("测试3: 日夜循环")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    print("\n  [日夜循环测试]")
    
    # 模拟时间流逝
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]  # 午夜、日出、正午、日落、午夜
    time_names = ["midnight", "sunrise", "noon", "sunset", "midnight2"]
    
    for t, name in zip(time_points, time_names):
        # 手动设置时间
        env._time_of_day = t
        env._state["time_of_day"] = t
        
        semantic_view = env.observe()
        semantic_view.lights = env.get_dynamic_lights()
        
        print(f"    时间 {t:.2f} ({name}): 光照配置 {len(semantic_view.lights)} 个")
        
        image = renderer.render(semantic_view)
        save_image(image, "lighting", f"time_{name}.png")
    
    print(f"    ✓ 日夜循环测试完成")


def test_geometry_transformation():
    """测试几何体变形"""
    print_separator("测试4: 几何体变形")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    print("\n  [几何体变形测试]")
    
    # 移动到收集物位置
    env._state["player"]["pos"] = (-3, 1, 2)
    env._sync_player_object()
    env._update_camera()
    
    # 观察让物体变为REVEALED
    for _ in range(5):
        env.step({"action": "move", "params": {"dx": 0, "dz": 0}})
    
    # 获取collectible_0的当前几何体
    for obj in env._state["objects"]:
        if obj["id"] == "collectible_0":
            print(f"  变形前几何体: {obj.get('geometry')}")
    
    # 渲染变形前
    semantic_view = env.observe()
    semantic_view.lights = env.get_dynamic_lights()
    image = renderer.render(semantic_view)
    save_image(image, "transforms", "transform_before.png")
    
    # 执行变形
    env.step({"action": "interact", "params": {}})
    
    for obj in env._state["objects"]:
        if obj["id"] == "collectible_0":
            print(f"  变形后几何体: {obj.get('geometry')}")
    
    # 渲染变形动画
    for frame in range(8):
        env.step({"action": "move", "params": {"dx": 0, "dz": 0}})
        
        semantic_view = env.observe()
        semantic_view.lights = env.get_dynamic_lights()
        
        image = renderer.render(semantic_view)
        save_image(image, "transforms", f"transform_frame_{frame:02d}.png")
    
    print(f"    ✓ 几何体变形测试完成")


def test_collection_effect():
    """测试收集效果"""
    print_separator("测试5: 收集效果")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    print("\n  [收集效果测试]")
    
    # 移动到收集物位置
    env._state["player"]["pos"] = (-3, 1, 1)
    env._sync_player_object()
    env._update_camera()
    
    # 观察并激活
    for _ in range(6):
        env.step({"action": "move", "params": {"dx": 0, "dz": 0}})
    
    print(f"  收集前分数: {env._state['score']}")
    print(f"  收集前物体数: {len([o for o in env._state['objects'] if 'collectible' in o['id']])}")
    
    # 渲染收集前
    semantic_view = env.observe()
    semantic_view.lights = env.get_dynamic_lights()
    image = renderer.render(semantic_view)
    save_image(image, "effects", "collect_before.png")
    
    # 执行收集
    env.step({"action": "collect", "params": {}})
    
    print(f"  收集后分数: {env._state['score']}")
    print(f"  已收集: {env._state['collected']}")
    
    # 渲染收集后
    semantic_view = env.observe()
    semantic_view.lights = env.get_dynamic_lights()
    image = renderer.render(semantic_view)
    save_image(image, "effects", "collect_after.png")
    
    print(f"    ✓ 收集效果测试完成")


def test_full_3d_session():
    """测试完整3D游戏会话"""
    print_separator("测试6: 完整3D游戏会话")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    import random
    random.seed(42)
    
    total_frames = 60
    key_frames = []
    
    print("\n  [完整3D游戏会话模拟]")
    
    for step in range(total_frames):
        # 随机动作
        action_type = random.choice(["move", "move", "move", "collect", "interact"])
        
        if action_type == "move":
            dx = random.choice([-1, 0, 1])
            dz = random.choice([-1, 0, 1])
            action = {"action": "move", "params": {"dx": dx, "dz": dz}}
        elif action_type == "collect":
            action = {"action": "collect", "params": {}}
        else:
            action = {"action": "interact", "params": {}}
        
        state, reward, done, info = env.step(action)
        
        # 每10步保存一帧
        if step % 10 == 0:
            semantic_view = info["semantic_view"]
            semantic_view.lights = env.get_dynamic_lights()
            game_info = env.get_game_info()
            
            image = renderer.render(semantic_view)
            save_image(image, "exploration", f"game_step_{step:03d}.png")
            key_frames.append(step)
        
        if done:
            print(f"  游戏结束于Step {step}")
            break
    
    game_info = env.get_game_info()
    print(f"\n  最终分数: {game_info['score']}")
    print(f"  收集物品: {game_info['collected']}个")
    print(f"  关键帧: {key_frames}")
    print(f"    ✓ 完整3D游戏会话测试完成")


def test_view_masks_3d():
    """测试3D视野遮罩"""
    print_separator("测试7: 3D视野遮罩")
    
    env = Exploration3DEnv()
    
    config = RenderConfig(
        asset_path=TEST_MODELS_DIR,
        resolution=(800, 600),
        background=BackgroundColor(color="#0a0a15")
    )
    renderer = create_3d_renderer(config)
    
    env.reset()
    
    # 移动到有物体的区域
    for _ in range(3):
        env.step({"action": "move", "params": {"dx": 0, "dz": -1}})
    
    semantic_view = env.observe()
    base_lights = env.get_dynamic_lights()
    
    # 测试不同遮罩
    masks = [
        ("无遮罩", {}),
        ("圆形遮罩", {"shape_type": "circle", "center": (400, 300), "radius": 250}),
        ("扇形遮罩", {"shape_type": "sector", "center": (400, 300), "radius": 280, "angle_start": -45, "angle_end": 45}),
        ("环形遮罩", {"shape_type": "ring", "center": (400, 300), "outer_radius": 280, "inner_radius": 100}),
    ]
    
    for mask_name, mask_config in masks:
        # 复制view_region并添加遮罩
        view_region = dict(semantic_view.view_region) if isinstance(semantic_view.view_region, dict) else {}
        view_region.update(mask_config)
        
        masked_view = SemanticView(
            view_region=view_region,
            objects=semantic_view.objects,
            lights=base_lights
        )
        
        image = renderer.render(masked_view)
        safe_name = mask_name.replace(" ", "_")
        save_image(image, "effects", f"mask_{safe_name}.png")
        print(f"    {mask_name}")
    
    print(f"    ✓ 3D视野遮罩测试完成")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有3D交互测试"""
    print("\n" + "=" * 70)
    print("  3D模式完整交互测试")
    print("=" * 70)
    print("""
测试内容：
  1. 基础3D探索
  2. 材质动态变化
  3. 日夜循环
  4. 几何体变形
  5. 收集效果
  6. 完整3D游戏会话
  7. 3D视野遮罩
    """)
    
    if not PYRENDER_AVAILABLE:
        print("  ⚠ PyRender不可用，将使用降级渲染器")
        print("    安装方法: pip install pyrender trimesh")
    
    ensure_output_dirs()
    
    try:
        test_basic_3d_exploration()
        test_material_changes()
        test_day_night_cycle()
        test_geometry_transformation()
        test_collection_effect()
        test_full_3d_session()
        test_view_masks_3d()
        
        print("\n" + "=" * 70)
        print("  所有3D交互测试完成！")
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

