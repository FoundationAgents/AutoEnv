from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import ShadowObservationPolicy
from env_generate import ShadowWorldGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class ShadowPuppetEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = ShadowObservationPolicy()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        if 'globals' in world_state and 'max_steps' in world_state['globals']:
            self.configs['termination']['max_steps'] = world_state['globals']['max_steps']
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = ShadowWorldGenerator(str(self.env_id), self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action.get('action', '')
        params = action.get('params', {})
        
        if action_name == 'MOVE_SHADOW':
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            # Be robust to string inputs like "-4" from upstream parsers
            dx = self._safe_int(dx, default=0)
            dy = self._safe_int(dy, default=0)
            self._move_shadow(dx, dy)
        elif action_name == 'CYCLE_SHAPE':
            shape = params.get('shape', 'square')
            self._cycle_shape(shape)
        elif action_name == 'TOGGLE_SHADOW':
            self._toggle_shadow()
        elif action_name == 'WIND_PULSE':
            self._wind_pulse()
        elif action_name == 'WAIT':
            pass
        
        self._apply_shadow_transformations()
        self._physics_step()
        
        return self._state

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Convert value to int, accepting numeric strings. Fallback to default on failure."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # Floor towards zero to preserve direction for small floats
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except (ValueError, TypeError):
                return default
        return default
    
    def _move_shadow(self, dx: int, dy: int):
        shadow_pos = self._state['shadow']['position']
        grid_size = self._state['globals']['grid_size']
        
        new_x = max(0, min(grid_size[0] - 1, shadow_pos[0] + dx))
        new_y = max(0, min(grid_size[1] - 1, shadow_pos[1] + dy))
        
        self._state['shadow']['position'] = [new_x, new_y]
    
    def _cycle_shape(self, shape: str):
        valid_shapes = ['square', 'circle', 'triangle', 'cross']
        if shape in valid_shapes:
            self._state['shadow']['shape'] = shape
    
    def _toggle_shadow(self):
        self._state['shadow']['active'] = not self._state['shadow']['active']
    
    def _wind_pulse(self):
        if not self._state['shadow']['active']:
            return
        
        shadow_pos = self._state['shadow']['position']
        
        for obj in self._state['objects']:
            if obj['property'] == 'Light':
                obj_pos = obj['position']
                dx = obj_pos[0] - shadow_pos[0]
                dy = obj_pos[1] - shadow_pos[1]
                
                if dx == 0 and dy == 0:
                    obj['velocity'] = [random.choice([-1, 1]), random.choice([-1, 1])]
                else:
                    if abs(dx) > abs(dy):
                        obj['velocity'] = [1 if dx > 0 else -1, 0]
                    else:
                        obj['velocity'] = [0, 1 if dy > 0 else -1]
    
    def _apply_shadow_transformations(self):
        if not self._state['shadow']['active']:
            return
        
        shadow_pos = self._state['shadow']['position']
        shadow_shape = self._state['shadow']['shape']
        
        property_mapping = {
            'square': 'Heavy',
            'circle': 'Light',
            'triangle': 'Bouncy',
            'cross': 'Sticky'
        }
        
        for obj in self._state['objects']:
            if obj['position'] == shadow_pos:
                obj['property'] = property_mapping[shadow_shape]
    
    def _physics_step(self):
        grid_size = self._state['globals']['grid_size']
        
        for obj in self._state['objects']:
            if obj['velocity'] != [0, 0]:
                new_x = obj['position'][0] + obj['velocity'][0]
                new_y = obj['position'][1] + obj['velocity'][1]
                
                new_x = max(0, min(grid_size[0] - 1, new_x))
                new_y = max(0, min(grid_size[1] - 1, new_y))
                
                obj['position'] = [new_x, new_y]
                
                if obj['property'] != 'Bouncy':
                    obj['velocity'] = [0, 0]
                elif new_x == 0 or new_x == grid_size[0] - 1:
                    obj['velocity'][0] = -obj['velocity'][0]
                elif new_y == 0 or new_y == grid_size[1] - 1:
                    obj['velocity'][1] = -obj['velocity'][1]
        
        self._resolve_collisions()
    
    def _resolve_collisions(self):
        positions = {}
        for i, obj in enumerate(self._state['objects']):
            pos = tuple(obj['position'])
            if pos not in positions:
                positions[pos] = []
            positions[pos].append(i)
        
        for pos, obj_indices in positions.items():
            if len(obj_indices) > 1:
                for i in obj_indices:
                    obj = self._state['objects'][i]
                    if obj['property'] == 'Sticky':
                        for j in obj_indices:
                            if i != j:
                                other_obj = self._state['objects'][j]
                                other_obj['velocity'] = [0, 0]
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        target_in_goal = self._check_target_in_goal()
        
        if target_in_goal:
            events.append("target_in_goal")
            return 1.0, events, reward_info
        
        return 0.0, events, reward_info
    
    def _check_target_in_goal(self) -> bool:
        goal_area = self._state['globals']['goal_area']
        
        for obj in self._state['objects']:
            if obj.get('is_target', False):
                obj_pos = obj['position']
                if (goal_area[0][0] <= obj_pos[0] <= goal_area[1][0] and
                    goal_area[0][1] <= obj_pos[1] <= goal_area[1][1]):
                    return True
        
        return False
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        grid_size = omega['grid_size']
        grid = [['.' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        
        goal_area = omega['goal_area']
        for x in range(goal_area[0][0], goal_area[1][0] + 1):
            for y in range(goal_area[0][1], goal_area[1][1] + 1):
                grid[x][y] = 'G'
        
        light_source = omega['light_source']
        grid[light_source[0]][light_source[1]] = 'L'
        
        if omega['shadow_active']:
            shadow_pos = omega['shadow_position']
            grid[shadow_pos[0]][shadow_pos[1]] = 'S'
        
        for obj in omega['objects']:
            pos = obj['position']
            if obj['is_target']:
                grid[pos[0]][pos[1]] = 'T'
            else:
                grid[pos[0]][pos[1]] = 'O'
        
        grid_display = '\n'.join([''.join(row) for row in grid])
        
        object_details = []
        for obj in omega['objects']:
            obj_type = "Target" if obj['is_target'] else "Object"
            object_details.append(f"{obj['id']} ({obj_type}): {obj['property']} at {obj['position']}")
        
        return f"""=== Shadow Puppet Reality Lab ===
Step {omega['t']}/{omega['max_steps']}

Grid (8x8):
{grid_display}

Legend: T=Target, O=Object, G=Goal, L=Light, S=Shadow
Shadow: {omega['shadow_shape']} at {omega['shadow_position']} ({'ON' if omega['shadow_active'] else 'OFF'})

Object Properties:
{chr(10).join(object_details)}

Actions: MOVE_SHADOW(dx,dy), CYCLE_SHAPE(shape), TOGGLE_SHADOW, WIND_PULSE, WAIT"""
    
    def done(self, state: Optional[Dict[str, Any]] = None) -> bool:
        if self._check_target_in_goal():
            return True
        
        if self._t >= self.configs["termination"]["max_steps"]:
            return True
        
        return False
