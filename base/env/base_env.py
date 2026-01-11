# ============================================================
# BASE ENVIRONMENT CLASSES
# Purpose: Abstract base classes for implementing agentic environments
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING
from base.env.base_observation import ObservationPolicy
from base.env.semantic_view import SemanticView, ObjectData

if TYPE_CHECKING:
    from base.env.base_camera import BaseCamera
    from base.env.view_volume import ViewVolume
    from base.env.spatial_index import SpatialIndex
    from base.env.bounding_box import BoundingBox, BoundingBox2D, BoundingBox3D


class BaseEnv(ABC):
    """Defines the true state, transition, and reward."""
    def __init__(self, env_id: int):
        self.env_id = env_id # env_id means the id of this class env. 
        self._t = 0
        self._history: List = [] # past state 
        self._state = None # current state
        self.configs = None
        # Optional: store latest action side-effect/result for UI/agent feedback
        self._last_action_result: Any = None
        self._dsl_config() 

    @abstractmethod
    def _dsl_config(self): 
        """
        Load DSL configuration from YAML file.
        Expected path: worlds/{env_id}/config.yaml
        """
        pass

    @abstractmethod
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        """
        Reset environment by either loading an existing world or generating a new one.

        Args:
            mode: "load" to load from file, "generate" to generate a new world
            world_id: Used only in "load" mode. Load the world with this id.
            seed: Used only in "generate" mode. Generate a new world with this seed.

        Behavior:
            - If mode == "load": Load world state from file using world_id.
            - If mode == "generate": Generate new world using seed, then load it.
        """
        pass

    @abstractmethod
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """
        Load world state from file.
        
        Args:
            world_id: Identifier of the world file to load
            
        Returns:
            Complete world state dictionary
        """
        pass
        
    @abstractmethod  
    def _generate_world(self, seed: Optional[int] = None) -> str:
        """
        Generate complete world using generator pipeline and save to file.
        
        Args:
            seed: Random seed for reproducible generation
            
        Returns:
            world_id: Identifier of the generated world file
        """
        pass

    @abstractmethod
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        State transition function.
        Input an action dict with two key:
        - action: str, the name of action
        - params: dict, the parameters of action
        And then apply the transition to self.state
        """
        pass

    @abstractmethod
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """
        Reward Function.
        It define agent how to get a reward.
        The state can be obtained from self.state, and past state can be gained from self.history. 
        """
        pass


class ObsEnv(BaseEnv):
    """Adds observation interface: output semantic observation from true state."""

    def __init__(self, env_id, obs_policy: ObservationPolicy):
        super().__init__(env_id)
        self.obs_policy = obs_policy

    @abstractmethod
    def observe_semantic(self):
        """
        Semantic-level observation.
        The observation policy refer to the observation state, such as full, partial, radius. 
        And this function is used to transfer state to semantic obs.
        """
        pass


class SkinEnv(ObsEnv):
    """Adds rendering interface: semantic observation -> final input (X)."""

    @abstractmethod
    def render_skin(self, omega) -> Any:
        """Render the final input from semantic observation."""
        pass

    def done(self) -> bool:
        # Default: only step count; override/add conditions if needed
        return self._t >= self.configs["termination"]["max_steps"]

    def step(self, action: Dict[str, Any]):
        """
        Basic step logic for an environment
        You can modify it in anywhere you want.
        """
        # Reset last action result; transition can set it
        self._last_action_result = None
        s_next = self.transition(action)
        reward, events, rinfo = self.reward(action)
        self._t += 1
        raw_obs = self.observe_semantic()
        agent_obs = self.render_skin(raw_obs)
        if_done = self.done(s_next)
        info = {
            "raw_obs": raw_obs,
            "skinned": agent_obs,
            "events": events,
            "reward_info": rinfo,
            "last_action_result": self._last_action_result,
        }
        return s_next, reward, if_done, info


class CameraEnv(SkinEnv):
    """
    带有 Camera 视野系统的环境基类。
    
    提供统一的 2D/3D 视野检测接口，通过 ViewVolume 抽象和空间索引实现高效的可见性检测。
    
    核心功能：
    - 统一的 get_visible_objects() 接口，2D/3D 行为一致
    - 可选的空间索引（Octree/QuadTree）加速查询
    - OBB 碰撞检测支持
    
    Attributes:
        camera: Camera 实例 (Camera2D 或 Camera3D)
        spatial_index: 可选的空间索引实例
    """
    
    def __init__(
        self,
        env_id,
        camera: "BaseCamera",
        spatial_index: Optional["SpatialIndex"] = None,
        obs_policy: Optional[ObservationPolicy] = None
    ):
        """
        初始化 CameraEnv。
        
        Args:
            env_id: 环境 ID
            camera: Camera 实例，定义观察区域
            spatial_index: 可选的空间索引，用于加速大规模场景的查询
            obs_policy: 观察策略（仅用于向后兼容，CameraEnv 不使用此参数）
        """
        super().__init__(env_id, obs_policy)
        self.camera = camera
        self.spatial_index: Optional["SpatialIndex"] = spatial_index
        
        # 物体状态缓存，用于空间索引
        self._object_cache: Dict[str, Dict[str, Any]] = {}
    
    def get_visible_objects(
        self,
        view: "ViewVolume",
        spatial_index: Optional["SpatialIndex"] = None
    ) -> List[ObjectData]:
        """
        根据 ViewVolume 返回可见物体列表。
        
        这是统一的可见性检测接口，2D 和 3D 场景行为一致。
        
        检测流程：
        1. 如果有空间索引，先通过 AABB 快速筛选候选物体
        2. 使用 OBB 碰撞检测进行精确筛选
        3. 返回在视野内的物体列表
        
        Args:
            view: ViewVolume 实例（ViewVolume2D 或 ViewVolume3D）
            spatial_index: 可选的空间索引，如果为 None 则使用实例的 spatial_index
            
        Returns:
            可见物体列表 (List[ObjectData])
            
        子类实现说明：
            子类需要重写 _get_all_objects() 方法返回所有物体，
            或者重写整个 get_visible_objects() 方法自定义检测逻辑。
        """
        index = spatial_index or self.spatial_index
        
        # 获取候选物体
        if index is not None:
            # 使用空间索引快速筛选
            candidate_ids = index.query_region(view.bounds)
            candidates = self._get_objects_by_ids(candidate_ids)
        else:
            # 遍历所有物体
            candidates = self._get_all_objects()
        
        # 精确可见性检测
        visible_objects = []
        for obj in candidates:
            if self._is_object_visible(obj, view):
                visible_objects.append(self._to_object_data(obj))
        
        return visible_objects
    
    def _get_all_objects(self) -> List[Dict[str, Any]]:
        """
        获取环境中的所有物体。
        
        子类应该重写此方法，返回环境中所有可能可见的物体列表。
        每个物体应该是一个字典，至少包含以下字段：
        - 'id': 物体唯一标识符
        - 'pos': 物体位置 (x, y) 或 (x, y, z)
        
        可选字段：
        - 'size': 物体尺寸 (width, height) 或 (width, height, depth)
        - 'rotation': 物体旋转角度
        - 'type': 物体类型
        
        Returns:
            物体字典列表
        """
        # 默认实现：从 _state 中获取 objects
        if self._state is None:
            return []
        
        objects = self._state.get("objects", [])
        if isinstance(objects, dict):
            # 如果是字典格式，转换为列表
            return [{"id": k, **v} if isinstance(v, dict) else {"id": k, "data": v}
                    for k, v in objects.items()]
        return objects
    
    def _get_objects_by_ids(self, obj_ids: List[str]) -> List[Dict[str, Any]]:
        """
        根据 ID 列表获取物体。
        
        Args:
            obj_ids: 物体 ID 列表
            
        Returns:
            物体字典列表
        """
        all_objects = self._get_all_objects()
        id_set = set(obj_ids)
        return [obj for obj in all_objects if obj.get("id") in id_set]
    
    def _is_object_visible(self, obj: Dict[str, Any], view: "ViewVolume") -> bool:
        """
        检测单个物体是否在视野内。
        
        使用 OBB 碰撞检测进行精确判断。
        
        Args:
            obj: 物体字典
            view: ViewVolume 实例
            
        Returns:
            True 如果物体在视野内
        """
        pos = obj.get("pos")
        if pos is None:
            return False
        
        # 提取位置
        if view.dimension == 2:
            center = (float(pos[0]), float(pos[1]))
        else:
            center = (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]) if len(pos) > 2 else 0.0
            )
        
        # 检查是否有尺寸信息
        size = obj.get("size")
        if size is None:
            # 没有尺寸信息，使用点检测
            return view.contains_point(center)
        
        # 计算半尺寸
        if view.dimension == 2:
            half_extents = (float(size[0]) / 2, float(size[1]) / 2)
        else:
            half_extents = (
                float(size[0]) / 2,
                float(size[1]) / 2,
                float(size[2]) / 2 if len(size) > 2 else float(size[0]) / 2
            )
        
        # 提取旋转
        rotation = obj.get("rotation")
        if rotation is not None:
            if view.dimension == 2:
                rot = (float(rotation),) if not isinstance(rotation, (tuple, list)) else (float(rotation[0]),)
            else:
                if isinstance(rotation, (tuple, list)) and len(rotation) >= 3:
                    rot = tuple(float(r) for r in rotation[:3])
                else:
                    rot = (0.0, 0.0, 0.0)
        else:
            rot = None
        
        # OBB 碰撞检测
        return view.intersects_obb(center, half_extents, rot)
    
    def _to_object_data(self, obj: Dict[str, Any]) -> ObjectData:
        """
        将物体字典转换为 ObjectData。
        
        Args:
            obj: 物体字典
            
        Returns:
            ObjectData 实例
        """
        pos = obj.get("pos", (0, 0))
        obj_id = obj.get("id", obj.get("type", "unknown"))
        
        # 构建 ObjectData
        data = {
            "id": str(obj_id),
            "pos": tuple(pos)
        }
        
        # 复制其他字段
        for key in ["size", "transform", "z_index", "geometry", "model_path", "material", "scale_3d"]:
            if key in obj:
                data[key] = obj[key]
        
        return ObjectData(**data)
    
    def update_spatial_index(self) -> None:
        """
        更新空间索引。
        
        当环境状态发生变化时调用此方法，重建空间索引。
        如果没有设置空间索引，此方法不做任何操作。
        """
        if self.spatial_index is None:
            return
        
        from base.env.bounding_box import BoundingBox2D, BoundingBox3D
        
        self.spatial_index.clear()
        
        for obj in self._get_all_objects():
            obj_id = obj.get("id")
            pos = obj.get("pos")
            if obj_id is None or pos is None:
                continue
            
            size = obj.get("size", (1, 1, 1))
            
            # 创建包围盒
            if self.camera.dimension == 2:
                half_w = float(size[0]) / 2 if len(size) > 0 else 0.5
                half_h = float(size[1]) / 2 if len(size) > 1 else 0.5
                bounds = BoundingBox2D(
                    min_point=(pos[0] - half_w, pos[1] - half_h),
                    max_point=(pos[0] + half_w, pos[1] + half_h)
                )
            else:
                half_w = float(size[0]) / 2 if len(size) > 0 else 0.5
                half_h = float(size[1]) / 2 if len(size) > 1 else 0.5
                half_d = float(size[2]) / 2 if len(size) > 2 else 0.5
                z = float(pos[2]) if len(pos) > 2 else 0.0
                bounds = BoundingBox3D(
                    min_point=(pos[0] - half_w, pos[1] - half_h, z - half_d),
                    max_point=(pos[0] + half_w, pos[1] + half_h, z + half_d)
                )
            
            self.spatial_index.insert(str(obj_id), bounds)
    
    def on_observe(
        self,
        view: "ViewVolume",
        visible_objects: List[Union[ObjectData, Dict]]
    ) -> None:
        """
        观察时的副作用 hook（可选覆盖）。
        
        在获取可见物体之后调用，可用于实现：
        - 视野内物体的"被发现"标记
        - 触发观察相关的事件
        - 更新战争迷雾等
        
        Args:
            view: ViewVolume 实例
            visible_objects: get_visible_objects 返回的可见物体列表
        """
        pass
    
    def observe(self) -> SemanticView:
        """
        框架实现的观察流程，返回标准化的 SemanticView。
        
        流程：
        1. 获取当前相机视野 (ViewVolume)
        2. 调用 get_visible_objects(view) - 获取可见物体
        3. 调用 on_observe(view, objects) - 处理副作用
        4. 组装并返回 SemanticView
        
        Returns:
            SemanticView 实例，包含视野区域和可见物体列表
        """
        view = self.camera.view
        objects = self.get_visible_objects(view, self.spatial_index)
        self.on_observe(view, objects)
        
        # 构建 view_region 用于渲染器
        if hasattr(view, 'get_camera_params'):
            # 3D 相机
            view_region = view.get_camera_params()
        else:
            # 2D 相机 - 使用 bounds 和形状信息
            bounds = view.bounds
            view_region = {
                "x": bounds.min_point[0],
                "y": bounds.min_point[1],
                "width": bounds.size[0],
                "height": bounds.size[1],
                "vertices": view.vertices
            }
            
            # 添加形状元数据（用于渲染器精确遮罩）
            if hasattr(view, 'shape_info') and view.shape_info is not None:
                view_region.update(view.shape_info)
        
        return SemanticView(
            view_region=view_region,
            objects=objects,
            metadata={"step": self._t}
        )
    
    def step(self, action: Dict[str, Any]):
        """
        扩展的 step 逻辑，包含 Camera 更新。
        
        在状态转移后更新 Camera，然后调用父类的 step 逻辑。
        """
        # Reset last action result; transition can set it
        self._last_action_result = None
        s_next = self.transition(action)
        reward, events, rinfo = self.reward(action)
        self._t += 1
        
        # 更新 Camera（如果状态包含 agent 信息）
        if self._state and "agent" in self._state:
            self.camera.update(self._state["agent"])
        
        # 更新空间索引（如果物体可能移动）
        # 注意：如果物体频繁移动，可以考虑只更新变化的物体
        # self.update_spatial_index()
        
        # 获取语义观察和渲染输出
        raw_obs = self.observe_semantic()
        semantic_view = self.observe()
        agent_obs = self.render_skin(raw_obs)
        if_done = self.done()
        
        info = {
            "raw_obs": raw_obs,
            "semantic_view": semantic_view,
            "skinned": agent_obs,
            "events": events,
            "reward_info": rinfo,
            "last_action_result": self._last_action_result,
        }
        return s_next, reward, if_done, info
    
    def init_spatial_index(
        self,
        world_bounds: Optional["BoundingBox"] = None,
        max_objects: int = 10,
        max_depth: int = 8
    ) -> None:
        """
        初始化空间索引。
        
        根据相机维度自动选择 QuadTree (2D) 或 Octree (3D)。
        
        Args:
            world_bounds: 世界边界，如果为 None 则使用默认值
            max_objects: 节点分裂前的最大物体数
            max_depth: 最大深度
        """
        from base.env.spatial_index import create_spatial_index
        from base.env.bounding_box import BoundingBox2D, BoundingBox3D
        
        if world_bounds is None:
            # 使用默认边界
            if self.camera.dimension == 2:
                world_bounds = BoundingBox2D(
                    min_point=(-10000, -10000),
                    max_point=(10000, 10000)
                )
            else:
                world_bounds = BoundingBox3D(
                    min_point=(-1000, -1000, -1000),
                    max_point=(1000, 1000, 1000)
                )
        
        self.spatial_index = create_spatial_index(
            dimension=self.camera.dimension,
            bounds=world_bounds,
            max_objects=max_objects,
            max_depth=max_depth
        )
