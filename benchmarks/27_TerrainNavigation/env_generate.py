from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from collections import deque
import hashlib
import time

class IceLakeGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.state_template = config.get("state_template", {})
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_state = self._execute_pipeline(self.state_template.copy(), seed)
        world_id = self._generate_world_id(seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = self._init_from_template(base_state)
        world_state = self._generate_lake_layout(world_state)
        world_state = self._ensure_path_connectivity(world_state)
        world_state = self._finalize_positions(world_state)
        return world_state
    
    def _init_from_template(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        return base_state.copy()
    
    def _generate_lake_layout(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        p_hole = random.uniform(0.1, 0.2)
        grid_size = world_state["tiles"]["size"]
        start_pos = world_state["start_pos"]
        goal_pos = world_state["goal_pos"]
        
        layout = []
        for r in range(grid_size[0]):
            row = []
            for c in range(grid_size[1]):
                if [r, c] == start_pos or [r, c] == goal_pos:
                    row.append("ice")
                elif random.random() < p_hole:
                    row.append("water")
                else:
                    row.append("ice")
            layout.append(row)
        
        world_state["tiles"]["layout"] = layout
        return world_state
    
    def _ensure_path_connectivity(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        max_attempts = 10
        for attempt in range(max_attempts):
            if self._has_path(world_state):
                return world_state
            world_state = self._generate_lake_layout(world_state)
        return world_state
    
    def _has_path(self, world_state: Dict[str, Any]) -> bool:
        layout = world_state["tiles"]["layout"]
        start_pos = world_state["start_pos"]
        goal_pos = world_state["goal_pos"]
        grid_size = world_state["tiles"]["size"]
        
        queue = deque([tuple(start_pos)])
        visited = {tuple(start_pos)}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            r, c = queue.popleft()
            
            if [r, c] == goal_pos:
                return True
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_size[0] and 0 <= nc < grid_size[1] and 
                    (nr, nc) not in visited and layout[nr][nc] != "water"):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def _finalize_positions(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        world_state["agent"]["pos"] = world_state["start_pos"].copy()
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        else:
            timestamp = int(time.time() * 1000000)
            return f"world_{timestamp}"