import os
import sys
import yaml
import random
import math
from typing import Dict, Any, Optional, Tuple, List

# Add the AutoEnv path to sys.path for imports
sys.path.append('../../../')
from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import ChemicalSensorPolicy
from env_generate import MazeChemicalGenerator

class MolecularTasteEnv(SkinEnv):
    def __init__(self, env_id: str):
        obs_policy = ChemicalSensorPolicy()
        super().__init__(env_id, obs_policy)
        self._goal_reached = False
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self._goal_reached = False
        
        if mode == "generate":
            world_id = self._generate_world(seed)
            
        if world_id is None:
            raise ValueError("world_id must be provided in load mode")
            
        world_state = self._load_world(world_id)
        self._state = world_state
        
        # Set random starting position ensuring valid path length
        valid_starts = self._find_valid_starts()
        if valid_starts:
            start_pos = random.choice(valid_starts)
            self._state["agent"]["pos"] = start_pos
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = MazeChemicalGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def _find_valid_starts(self) -> List[Tuple[int, int]]:
        valid_starts = []
        goal_pos = tuple(self._state["maze"]["goal_pos"])
        maze_size = self._state["maze"]["size"]
        walls = set(tuple(wall) for wall in self._state["maze"]["walls"])
        
        for x in range(maze_size[0]):
            for y in range(maze_size[1]):
                pos = (x, y)
                if pos not in walls and pos != goal_pos:
                    path_length = self._calculate_path_length(pos, goal_pos, walls, maze_size)
                    if 8 <= path_length <= 12:
                        valid_starts.append(pos)
        
        if not valid_starts:
            # Fallback to any non-wall, non-goal position
            for x in range(maze_size[0]):
                for y in range(maze_size[1]):
                    pos = (x, y)
                    if pos not in walls and pos != goal_pos:
                        valid_starts.append(pos)
        
        return valid_starts
    
    def _calculate_path_length(self, start, goal, walls, maze_size):
        # Simple BFS to calculate shortest path length
        from collections import deque
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == goal:
                return dist
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and 
                    (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        return float('inf')
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        action_name = action.get("action", "")
        
        current_pos = self._state["agent"]["pos"]
        new_pos = list(current_pos)
        
        if action_name == "MOVE_NORTH":
            new_pos[1] -= 1
        elif action_name == "MOVE_EAST":
            new_pos[0] += 1
        elif action_name == "MOVE_SOUTH":
            new_pos[1] += 1
        elif action_name == "MOVE_WEST":
            new_pos[0] -= 1
        # DO_NOTHING keeps current position
        
        # Check if move is valid
        if self._is_valid_position(new_pos):
            self._state["agent"]["pos"] = new_pos
        
        return self._state
    
    def _is_valid_position(self, pos):
        x, y = pos
        maze_size = self._state["maze"]["size"]
        walls = set(tuple(wall) for wall in self._state["maze"]["walls"])
        
        if not (0 <= x < maze_size[0] and 0 <= y < maze_size[1]):
            return False
        if tuple(pos) in walls:
            return False
        return True
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        current_pos = tuple(self._state["agent"]["pos"])
        goal_pos = tuple(self._state["maze"]["goal_pos"])
        
        if current_pos == goal_pos and not self._goal_reached:
            self._goal_reached = True
            return 1.0, ["goal_reached_first_time"], {"goal_reached": True}
        
        return 0.0, [], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        max_steps = self.configs["termination"]["max_steps"]
        
        template = f"""Step {self._t + 1}/{max_steps} - Chemical Sensor Reading

Flavor Vector (0.0-1.0):
Sweet: {omega['flavor_vector'][0]:.3f} | Sour: {omega['flavor_vector'][1]:.3f} | Salty: {omega['flavor_vector'][2]:.3f}
Bitter: {omega['flavor_vector'][3]:.3f} | Umami: {omega['flavor_vector'][4]:.3f}

Movement Options:
North: {omega['wall_bitmask'][0]} | East: {omega['wall_bitmask'][1]}
South: {omega['wall_bitmask'][2]} | West: {omega['wall_bitmask'][3]}

Actions: DO_NOTHING, MOVE_NORTH, MOVE_EAST, MOVE_SOUTH, MOVE_WEST"""
        
        return template
    
    def done(self, state=None) -> bool:
        max_steps = self.configs["termination"]["max_steps"]
        if self._state and "globals" in self._state and "max_steps" in self._state.get("globals", {}):
            max_steps = self._state["globals"]["max_steps"]
        
        return self._t >= max_steps or self._goal_reached
