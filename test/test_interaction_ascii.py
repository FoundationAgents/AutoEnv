# ============================================================
# ASCII模式完整交互测试
# 测试CameraEnv + Camera2D + ASCIIRenderer的完整交互流程
# 包含状态机、观测副作用、Step更新等复杂功能
# ============================================================

"""
测试场景：迷宫探索游戏
    - player: 玩家角色
    - wall: 墙壁（不可通过）
    - treasure: 宝藏（可收集）
    - enemy: 敌人（被观测后追踪玩家）
    - key: 钥匙（用于开门）
    - door: 门（需要钥匙开启）
    
状态机：
    - HIDDEN → DISCOVERED（被观测到）
    - DISCOVERED → COLLECTED（执行collect动作）
    - door: LOCKED → UNLOCKED（收集钥匙后）
    
副作用：
    - 物体被观测到后改变符号（? → 实际符号）
    - treasure收集后消失
    - enemy被观测到后开始追踪player
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.env.base_env import CameraEnv
from base.env.base_camera import Camera2D
from base.env.base_observation import ObservationPolicy
from base.env.semantic_view import SemanticView, ObjectData
from base.env.view_volume import ViewVolume2D
from base.render.ascii_renderer import ASCIIRenderer
from base.render.base_renderer import RenderConfig


# ============================================================
# 输出配置
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output", "interaction", "ascii")


def ensure_output_dirs():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_output(content: str, filename: str):
    """保存文本输出到文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
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
    DISCOVERED = "discovered"   # 已发现
    COLLECTED = "collected"     # 已收集/已消失
    ACTIVE = "active"           # 激活状态（如敌人追踪）
    LOCKED = "locked"           # 锁定状态（门）
    UNLOCKED = "unlocked"       # 解锁状态（门）


# ============================================================
# 观察策略
# ============================================================
class MazeObservationPolicy(ObservationPolicy):
    """迷宫游戏观察策略"""
    
    def __call__(self, env_state: Dict[str, Any], t: int):
        return {
            "step": t,
            "player": env_state.get("player", {}),
            "inventory": env_state.get("inventory", []),
            "score": env_state.get("score", 0)
        }


# ============================================================
# 迷宫探索环境
# ============================================================
class MazeExplorationEnv(CameraEnv):
    """
    迷宫探索游戏环境
    
    特性：
    - 战争迷雾机制（未探索区域显示为?）
    - 物体状态机（HIDDEN → DISCOVERED → COLLECTED）
    - 敌人AI追踪
    - 钥匙开门机制
    """
    
    # 符号定义
    SYMBOLS = {
        "player": "@",
        "wall": "#",
        "treasure": "$",
        "enemy": "E",
        "key": "K",
        "door_locked": "D",
        "door_unlocked": ".",
        "floor": ".",
        "unknown": "?",
        "empty": " ",
    }
    
    def __init__(self, maze_width: int = 25, maze_height: int = 20, view_radius: int = 5):
        """
        初始化迷宫环境
        
        Args:
            maze_width: 迷宫宽度
            maze_height: 迷宫高度
            view_radius: 视野半径
        """
        # 创建2D相机 - 圆形视野
        camera = Camera2D(position=(12, 10))
        camera.set_circular_view(radius=view_radius, segments=16)
        
        super().__init__(
            env_id=1,
            obs_policy=MazeObservationPolicy(),
            camera=camera
        )
        
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.view_radius = view_radius
        
        self.configs = {
            "termination": {"max_steps": 100}
        }
        
        # 物体状态追踪
        self._object_states: Dict[str, ObjectState] = {}
        self._discovered_tiles: set = set()  # 已探索的位置
        
        self._init_maze()
    
    def _init_maze(self):
        """初始化迷宫"""
        self._t = 0
        self._object_states.clear()
        self._discovered_tiles.clear()
        
        # 玩家状态
        player_pos = (12, 10)
        
        self._state = {
            "player": {
                "pos": player_pos,
                "health": 100,
                "direction": (1, 0)
            },
            "inventory": [],
            "score": 0,
            "objects": self._create_maze_objects()
        }
        
        # 初始化所有物体状态为HIDDEN
        for obj in self._state["objects"]:
            obj_id = obj["id"]
            if obj["type"] == "wall":
                self._object_states[obj_id] = ObjectState.DISCOVERED  # 墙壁总是可见
            elif obj["type"] == "door":
                self._object_states[obj_id] = ObjectState.LOCKED
            else:
                self._object_states[obj_id] = ObjectState.HIDDEN
    
    def _create_maze_objects(self) -> List[Dict[str, Any]]:
        """创建迷宫物体"""
        objects = []
        
        # 边界墙壁
        for x in range(self.maze_width):
            objects.append(self._create_object("wall", f"wall_top_{x}", (x, 0)))
            objects.append(self._create_object("wall", f"wall_bottom_{x}", (x, self.maze_height - 1)))
        for y in range(self.maze_height):
            objects.append(self._create_object("wall", f"wall_left_{y}", (0, y)))
            objects.append(self._create_object("wall", f"wall_right_{y}", (self.maze_width - 1, y)))
        
        # 内部墙壁 - 创建迷宫结构
        internal_walls = [
            # 水平墙
            (5, 5, 8, True), (15, 5, 6, True),
            (3, 10, 5, True), (10, 10, 8, True),
            (5, 15, 10, True),
            # 垂直墙
            (5, 3, 4, False), (10, 6, 4, False),
            (15, 8, 5, False), (20, 3, 6, False),
        ]
        
        wall_id = 0
        for x, y, length, horizontal in internal_walls:
            for i in range(length):
                wx = x + i if horizontal else x
                wy = y if horizontal else y + i
                if 0 < wx < self.maze_width - 1 and 0 < wy < self.maze_height - 1:
                    objects.append(self._create_object("wall", f"wall_internal_{wall_id}", (wx, wy)))
                    wall_id += 1
        
        # 宝藏
        treasure_positions = [(3, 3), (21, 3), (3, 16), (21, 16), (12, 8)]
        for i, pos in enumerate(treasure_positions):
            objects.append(self._create_object("treasure", f"treasure_{i}", pos, value=100))
        
        # 敌人
        enemy_positions = [(8, 8), (18, 12), (6, 14)]
        for i, pos in enumerate(enemy_positions):
            objects.append(self._create_object("enemy", f"enemy_{i}", pos, tracking=False))
        
        # 钥匙
        objects.append(self._create_object("key", "key_0", (22, 8)))
        
        # 门
        objects.append(self._create_object("door", "door_0", (22, 17)))
        
        # 玩家
        objects.append(self._create_object("player", "player", (12, 10)))
        
        return objects
    
    def _create_object(self, obj_type: str, obj_id: str, pos: Tuple[int, int], **kwargs) -> Dict[str, Any]:
        """创建物体字典"""
        obj = {
            "id": obj_id,
            "type": obj_type,
            "pos": pos,
            "size": (1, 1),
            "z_index": self._get_z_index(obj_type),
        }
        obj.update(kwargs)
        return obj
    
    def _get_z_index(self, obj_type: str) -> int:
        """获取物体的z_index"""
        z_indices = {
            "floor": 0,
            "door": 5,
            "treasure": 10,
            "key": 10,
            "enemy": 20,
            "wall": 30,
            "player": 100,
        }
        return z_indices.get(obj_type, 0)
    
    # ============================================================
    # CameraEnv 抽象方法实现
    # ============================================================
    
    def _dsl_config(self):
        pass
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        """重置环境"""
        self._init_maze()
        self.camera.reset()
        player_pos = self._state["player"]["pos"]
        self.camera.position = player_pos
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        return self._state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        self._init_maze()
        return "maze_world"
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """状态转移"""
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        if action_name == "move":
            self._handle_move(params)
        elif action_name == "collect":
            self._handle_collect()
        elif action_name == "use_key":
            self._handle_use_key()
        
        # 更新敌人位置（追踪玩家）
        self._update_enemies()
        
        # 更新玩家物体位置
        self._sync_player_object()
        
        # 更新相机跟随
        self.camera.update(self._state["player"])
        
        return self._state
    
    def _handle_move(self, params: Dict[str, Any]):
        """处理移动"""
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        
        player = self._state["player"]
        new_x = player["pos"][0] + dx
        new_y = player["pos"][1] + dy
        
        # 边界检查
        if not (0 < new_x < self.maze_width - 1 and 0 < new_y < self.maze_height - 1):
            return
        
        # 碰撞检查
        if self._is_blocked(new_x, new_y):
            return
        
        player["pos"] = (new_x, new_y)
        if dx != 0 or dy != 0:
            player["direction"] = (dx, dy)
    
    def _is_blocked(self, x: int, y: int) -> bool:
        """检查位置是否被阻挡"""
        for obj in self._state["objects"]:
            if obj["pos"] == (x, y):
                if obj["type"] == "wall":
                    return True
                if obj["type"] == "door":
                    state = self._object_states.get(obj["id"], ObjectState.LOCKED)
                    if state == ObjectState.LOCKED:
                        return True
        return False
    
    def _handle_collect(self):
        """处理收集物品"""
        player_pos = self._state["player"]["pos"]
        
        for obj in self._state["objects"]:
            if obj["pos"] == player_pos and obj["type"] in ["treasure", "key"]:
                state = self._object_states.get(obj["id"])
                if state == ObjectState.DISCOVERED:
                    self._object_states[obj["id"]] = ObjectState.COLLECTED
                    self._state["inventory"].append(obj["id"])
                    
                    if obj["type"] == "treasure":
                        self._state["score"] += obj.get("value", 100)
                    
                    self._last_action_result = f"Collected {obj['id']}"
    
    def _handle_use_key(self):
        """使用钥匙开门"""
        player_pos = self._state["player"]["pos"]
        
        # 检查附近是否有锁定的门
        for obj in self._state["objects"]:
            if obj["type"] == "door":
                door_pos = obj["pos"]
                dist = abs(door_pos[0] - player_pos[0]) + abs(door_pos[1] - player_pos[1])
                if dist <= 1:  # 相邻
                    state = self._object_states.get(obj["id"])
                    if state == ObjectState.LOCKED:
                        # 检查是否有钥匙
                        if any("key" in item for item in self._state["inventory"]):
                            self._object_states[obj["id"]] = ObjectState.UNLOCKED
                            self._last_action_result = f"Unlocked {obj['id']}"
    
    def _update_enemies(self):
        """更新敌人位置（追踪玩家）"""
        player_pos = self._state["player"]["pos"]
        
        for obj in self._state["objects"]:
            if obj["type"] == "enemy":
                state = self._object_states.get(obj["id"])
                
                # 只有ACTIVE状态的敌人才追踪
                if state == ObjectState.ACTIVE:
                    ex, ey = obj["pos"]
                    px, py = player_pos
                    
                    # 简单追踪：朝玩家方向移动一格
                    dx = 1 if px > ex else (-1 if px < ex else 0)
                    dy = 1 if py > ey else (-1 if py < ey else 0)
                    
                    new_x, new_y = ex + dx, ey + dy
                    
                    # 检查是否可通行
                    if not self._is_blocked(new_x, new_y):
                        obj["pos"] = (new_x, new_y)
    
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
        
        # 收集宝藏奖励
        if self._last_action_result and "treasure" in str(self._last_action_result):
            reward += 100
            events.append("treasure_collected")
        
        # 开门奖励
        if self._last_action_result and "Unlocked" in str(self._last_action_result):
            reward += 50
            events.append("door_unlocked")
        
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
        
        - 将HIDDEN状态的物体变为DISCOVERED
        - 敌人被发现后变为ACTIVE（开始追踪）
        - 记录已探索的位置
        """
        for obj in visible_objects:
            obj_id = obj.id
            state = self._object_states.get(obj_id)
            
            if state == ObjectState.HIDDEN:
                self._object_states[obj_id] = ObjectState.DISCOVERED
            
            # 敌人特殊处理：发现后激活
            if obj_id.startswith("enemy_") and state in [ObjectState.HIDDEN, ObjectState.DISCOVERED]:
                self._object_states[obj_id] = ObjectState.ACTIVE
            
            # 记录探索位置
            pos = obj.pos
            if len(pos) >= 2:
                self._discovered_tiles.add((int(pos[0]), int(pos[1])))
    
    def _get_all_objects(self) -> List[Dict[str, Any]]:
        """获取所有物体（过滤已收集的）"""
        objects = []
        for obj in self._state.get("objects", []):
            obj_id = obj["id"]
            state = self._object_states.get(obj_id, ObjectState.HIDDEN)
            
            # 跳过已收集的物体
            if state == ObjectState.COLLECTED:
                continue
            
            objects.append(obj)
        return objects
    
    def _to_object_data(self, obj: Dict[str, Any]) -> ObjectData:
        """转换物体数据，根据状态添加符号信息"""
        obj_id = obj["id"]
        obj_type = obj["type"]
        state = self._object_states.get(obj_id, ObjectState.HIDDEN)
        pos = obj["pos"]
        
        # 根据状态确定符号
        if state == ObjectState.HIDDEN and obj_type not in ["wall", "player"]:
            symbol = self.SYMBOLS["unknown"]
        elif obj_type == "door":
            if state == ObjectState.UNLOCKED:
                symbol = self.SYMBOLS["door_unlocked"]
            else:
                symbol = self.SYMBOLS["door_locked"]
        else:
            symbol = self.SYMBOLS.get(obj_type, "?")
        
        return ObjectData(
            id=obj_id,
            pos=pos,
            size=obj.get("size"),
            z_index=obj.get("z_index", 0),
            symbol=symbol  # 添加符号到额外字段
        )
    
    def get_game_state_summary(self) -> str:
        """获取游戏状态摘要"""
        return (
            f"Step: {self._t} | "
            f"Score: {self._state['score']} | "
            f"Inventory: {self._state['inventory']} | "
            f"Player: {self._state['player']['pos']}"
        )


# ============================================================
# 测试函数
# ============================================================

def test_basic_exploration():
    """测试基础探索功能"""
    print_separator("测试1: 基础探索功能")
    
    # 创建环境
    env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=5)
    
    # 创建渲染器
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@",
            "wall": "#",
            "treasure": "$",
            "enemy": "E",
            "key": "K",
            "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    # 重置环境
    env.reset()
    
    outputs = []
    
    # 初始状态
    print("\n  [初始状态]")
    semantic_view = env.observe()
    output = renderer.render_with_info(
        semantic_view,
        header=f"=== {env.get_game_state_summary()} ===",
        footer="Legend: @ = Player, # = Wall, $ = Treasure, E = Enemy, K = Key, D = Door, ? = Unknown"
    )
    print(output)
    outputs.append(("step_00_initial", output))
    
    # 执行探索动作
    movements = [
        ("右移", {"dx": 1, "dy": 0}),
        ("右移", {"dx": 1, "dy": 0}),
        ("下移", {"dx": 0, "dy": 1}),
        ("下移", {"dx": 0, "dy": 1}),
        ("左移", {"dx": -1, "dy": 0}),
        ("上移", {"dx": 0, "dy": -1}),
    ]
    
    for i, (name, params) in enumerate(movements, 1):
        action = {"action": "move", "params": params}
        state, reward, done, info = env.step(action)
        
        semantic_view = info["semantic_view"]
        output = renderer.render_with_info(
            semantic_view,
            header=f"=== Step {i}: {name} | {env.get_game_state_summary()} ===",
            footer=f"Visible objects: {len(semantic_view.objects)}"
        )
        
        outputs.append((f"step_{i:02d}_{name}", output))
        
        if i <= 3:  # 只打印前3步
            print(f"\n  [Step {i}: {name}]")
            print(output)
    
    # 保存所有输出
    for filename, content in outputs:
        save_output(content, f"{filename}.txt")
    
    print(f"\n    ✓ 基础探索测试完成，共{len(outputs)}帧")


def test_state_machine():
    """测试物体状态机"""
    print_separator("测试2: 物体状态机")
    
    env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=6)
    
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@", "wall": "#", "treasure": "$",
            "enemy": "E", "key": "K", "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    env.reset()
    
    print("\n  [状态机转换测试]")
    
    # 移动到宝藏附近
    movements = [
        {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0},
        {"dx": 0, "dy": -1}, {"dx": 0, "dy": -1}, {"dx": 0, "dy": -1},
        {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0},
    ]
    
    for params in movements:
        env.step({"action": "move", "params": params})
    
    # 查看宝藏状态
    print(f"  移动后玩家位置: {env._state['player']['pos']}")
    
    # 找到附近的宝藏
    treasure_states = []
    for obj_id, state in env._object_states.items():
        if "treasure" in obj_id:
            treasure_states.append((obj_id, state.value))
    print(f"  宝藏状态: {treasure_states}")
    
    # 收集宝藏
    env.step({"action": "collect", "params": {}})
    
    treasure_states_after = []
    for obj_id, state in env._object_states.items():
        if "treasure" in obj_id:
            treasure_states_after.append((obj_id, state.value))
    print(f"  收集后状态: {treasure_states_after}")
    print(f"  背包: {env._state['inventory']}")
    print(f"  分数: {env._state['score']}")
    
    # 渲染当前状态
    semantic_view = env.observe()
    output = renderer.render_with_info(
        semantic_view,
        header=f"=== State Machine Test | {env.get_game_state_summary()} ===",
        footer=f"States: {dict(list(env._object_states.items())[:5])}"
    )
    print(f"\n{output}")
    save_output(output, "state_machine_test.txt")
    
    print("    ✓ 状态机测试完成")


def test_enemy_tracking():
    """测试敌人追踪功能"""
    print_separator("测试3: 敌人追踪功能")
    
    env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=8)
    
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@", "wall": "#", "treasure": "$",
            "enemy": "E", "key": "K", "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    env.reset()
    
    outputs = []
    
    print("\n  [敌人追踪测试]")
    
    # 先找到敌人
    initial_enemy_positions = {}
    for obj in env._state["objects"]:
        if obj["type"] == "enemy":
            initial_enemy_positions[obj["id"]] = obj["pos"]
    print(f"  初始敌人位置: {initial_enemy_positions}")
    
    # 移动接近敌人以发现它们
    movements = [
        {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0}, {"dx": -1, "dy": 0},
        {"dx": 0, "dy": -1}, {"dx": 0, "dy": -1},
    ]
    
    for i, params in enumerate(movements):
        state, _, _, info = env.step({"action": "move", "params": params})
        
        # 记录敌人状态
        enemy_info = []
        for obj in env._state["objects"]:
            if obj["type"] == "enemy":
                enemy_state = env._object_states.get(obj["id"], ObjectState.HIDDEN)
                enemy_info.append(f"{obj['id']}: {obj['pos']} ({enemy_state.value})")
        
        if i < 3:
            print(f"  Step {i+1}: {enemy_info}")
        
        semantic_view = info["semantic_view"]
        output = renderer.render_with_info(
            semantic_view,
            header=f"=== Enemy Tracking Step {i+1} ===",
            footer=f"Enemy states: {enemy_info}"
        )
        outputs.append((f"enemy_step_{i:02d}", output))
    
    # 继续几步观察敌人追踪
    for i in range(5, 10):
        # 随机移动
        import random
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        
        state, _, _, info = env.step({"action": "move", "params": {"dx": dx, "dy": dy}})
        
        semantic_view = info["semantic_view"]
        output = renderer.render(semantic_view)
        outputs.append((f"enemy_step_{i:02d}", output))
    
    # 保存
    for filename, content in outputs:
        save_output(content, f"{filename}.txt")
    
    print(f"    ✓ 敌人追踪测试完成，共{len(outputs)}帧")


def test_door_and_key():
    """测试门和钥匙机制"""
    print_separator("测试4: 门和钥匙机制")
    
    env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=6)
    
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@", "wall": "#", "treasure": "$",
            "enemy": "E", "key": "K", "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    env.reset()
    
    print("\n  [门和钥匙测试]")
    
    # 找到钥匙和门的位置
    key_pos = None
    door_pos = None
    for obj in env._state["objects"]:
        if obj["type"] == "key":
            key_pos = obj["pos"]
        if obj["type"] == "door":
            door_pos = obj["pos"]
    
    print(f"  钥匙位置: {key_pos}")
    print(f"  门位置: {door_pos}")
    
    # 检查门初始状态
    door_state = None
    for obj_id, state in env._object_states.items():
        if "door" in obj_id:
            door_state = state.value
    print(f"  门初始状态: {door_state}")
    
    # 模拟拿到钥匙
    env._state["inventory"].append("key_0")
    env._object_states["key_0"] = ObjectState.COLLECTED
    print(f"  收集钥匙后背包: {env._state['inventory']}")
    
    # 移动到门附近
    env._state["player"]["pos"] = (door_pos[0] - 1, door_pos[1])
    env._sync_player_object()
    
    print(f"  玩家移动到门附近: {env._state['player']['pos']}")
    
    # 尝试开门
    env.step({"action": "use_key", "params": {}})
    
    # 检查门状态
    for obj_id, state in env._object_states.items():
        if "door" in obj_id:
            print(f"  开门后状态: {state.value}")
    
    # 渲染最终状态
    semantic_view = env.observe()
    output = renderer.render_with_info(
        semantic_view,
        header=f"=== Door & Key Test | {env.get_game_state_summary()} ===",
        footer="Key collected, door unlocked!"
    )
    print(f"\n{output}")
    save_output(output, "door_key_test.txt")
    
    print("    ✓ 门和钥匙测试完成")


def test_full_game_session():
    """测试完整游戏会话"""
    print_separator("测试5: 完整游戏会话")
    
    env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=5)
    
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@", "wall": "#", "treasure": "$",
            "enemy": "E", "key": "K", "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    env.reset()
    
    outputs = []
    total_reward = 0
    
    # 模拟30步游戏
    import random
    random.seed(42)
    
    print("\n  [完整游戏会话模拟]")
    
    for step in range(30):
        # 随机动作
        action_type = random.choice(["move", "move", "move", "collect"])
        
        if action_type == "move":
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            action = {"action": "move", "params": {"dx": dx, "dy": dy}}
        else:
            action = {"action": "collect", "params": {}}
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        semantic_view = info["semantic_view"]
        
        # 每5步保存一帧
        if step % 5 == 0:
            output = renderer.render_with_info(
                semantic_view,
                header=f"=== Game Session Step {step} | Score: {env._state['score']} ===",
                footer=f"Total Reward: {total_reward:.0f} | Visible: {len(semantic_view.objects)}"
            )
            outputs.append((f"game_step_{step:02d}", output))
            
            if step == 0:
                print(output)
        
        if done:
            print(f"  游戏结束于Step {step}")
            break
    
    # 保存所有帧
    for filename, content in outputs:
        save_output(content, f"{filename}.txt")
    
    print(f"\n  最终分数: {env._state['score']}")
    print(f"  总奖励: {total_reward}")
    print(f"  背包: {env._state['inventory']}")
    print(f"    ✓ 完整游戏会话测试完成，共{len(outputs)}帧")


def test_different_view_sizes():
    """测试不同视野大小"""
    print_separator("测试6: 不同视野大小对比")
    
    view_radii = [3, 5, 8, 12]
    
    config = RenderConfig(
        grid_size=(25, 20),
        symbol_map={
            "player": "@", "wall": "#", "treasure": "$",
            "enemy": "E", "key": "K", "door": "D",
        },
        default_symbol="?",
        empty_symbol=" "
    )
    renderer = ASCIIRenderer(config)
    
    for radius in view_radii:
        env = MazeExplorationEnv(maze_width=25, maze_height=20, view_radius=radius)
        env.reset()
        
        # 移动几步探索
        movements = [
            {"dx": 1, "dy": 0}, {"dx": 1, "dy": 0},
            {"dx": 0, "dy": 1}, {"dx": 0, "dy": 1},
        ]
        for params in movements:
            env.step({"action": "move", "params": params})
        
        semantic_view = env.observe()
        
        output = renderer.render_with_info(
            semantic_view,
            header=f"=== View Radius: {radius} | Visible: {len(semantic_view.objects)} ===",
            footer=f"Player Position: {env._state['player']['pos']}"
        )
        
        save_output(output, f"view_radius_{radius}.txt")
        
        print(f"  视野半径 {radius}: 可见物体 {len(semantic_view.objects)} 个")
    
    print("    ✓ 不同视野大小测试完成")


# ============================================================
# 主函数
# ============================================================
def main():
    """运行所有ASCII交互测试"""
    print("\n" + "=" * 70)
    print("  ASCII模式完整交互测试")
    print("=" * 70)
    print("""
测试内容：
  1. 基础探索功能
  2. 物体状态机（HIDDEN → DISCOVERED → COLLECTED）
  3. 敌人追踪功能
  4. 门和钥匙机制
  5. 完整游戏会话模拟
  6. 不同视野大小对比
    """)
    
    ensure_output_dirs()
    
    try:
        test_basic_exploration()
        test_state_machine()
        test_enemy_tracking()
        test_door_and_key()
        test_full_game_session()
        test_different_view_sizes()
        
        print("\n" + "=" * 70)
        print("  所有ASCII交互测试完成！")
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

