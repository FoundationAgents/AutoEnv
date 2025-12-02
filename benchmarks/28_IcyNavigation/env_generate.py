from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import random
import yaml
import os
from collections import deque
from copy import deepcopy
import time

class ReverseLakeGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Get base state from config
        base_state = deepcopy(self.config['state_template'])
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world_id
        world_id = self._generate_world_id(seed)
        
        # Save world
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        max_attempts = 100
        
        for attempt in range(max_attempts):
            try:
                # Step 1: Initialize from template
                state = deepcopy(base_state)
                
                # Step 2: Place ice tiles (8-12 tiles)
                state = self._place_ice_tiles(state)
                
                # Step 3: Place goal flag on safe tile
                state = self._place_goal_flag(state)
                
                # Step 4: Place agent start on safe tile
                state = self._place_agent_start(state)
                
                # Step 5: Validate reachability
                if self._validate_reachability(state):
                    return state
                    
            except Exception as e:
                continue
        
        raise RuntimeError("Failed to generate valid world after maximum attempts")
    
    def _place_ice_tiles(self, state: Dict[str, Any]) -> Dict[str, Any]:
        grid_size = state['globals']['grid_size']
        min_ice = 8
        max_ice = 12
        
        num_ice = random.randint(min_ice, max_ice)
        ice_positions = set()
        
        # Generate random positions for ice tiles
        while len(ice_positions) < num_ice:
            x = random.randint(0, grid_size[0] - 1)
            y = random.randint(0, grid_size[1] - 1)
            ice_positions.add((x, y))
        
        # Store ice tiles
        state['objects']['ice_tiles'] = []
        for x, y in ice_positions:
            state['objects']['ice_tiles'].append({'pos': [x, y]})
        
        return state
    
    def _place_goal_flag(self, state: Dict[str, Any]) -> Dict[str, Any]:
        grid_size = state['globals']['grid_size']
        ice_positions = {tuple(ice['pos']) for ice in state['objects']['ice_tiles']}
        
        # Find safe positions (not ice)
        safe_positions = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if (x, y) not in ice_positions:
                    safe_positions.append([x, y])
        
        if not safe_positions:
            raise RuntimeError("No safe positions available for goal")
        
        goal_pos = random.choice(safe_positions)
        state['objects']['goal_flag']['pos'] = goal_pos
        state['objects']['goal_flag']['collected'] = False
        
        return state
    
    def _place_agent_start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        grid_size = state['globals']['grid_size']
        ice_positions = {tuple(ice['pos']) for ice in state['objects']['ice_tiles']}
        goal_pos = tuple(state['objects']['goal_flag']['pos'])
        
        # Find safe positions (not ice, not goal)
        safe_positions = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if (x, y) not in ice_positions and (x, y) != goal_pos:
                    safe_positions.append([x, y])
        
        if not safe_positions:
            raise RuntimeError("No safe positions available for agent start")
        
        start_pos = random.choice(safe_positions)
        state['agent']['pos'] = start_pos
        state['agent']['start_pos'] = start_pos
        
        return state
    
    def _validate_reachability(self, state: Dict[str, Any]) -> bool:
        # BFS to check if goal is reachable from start
        start_pos = tuple(state['agent']['pos'])
        goal_pos = tuple(state['objects']['goal_flag']['pos'])
        grid_size = state['globals']['grid_size']
        ice_positions = {tuple(ice['pos']) for ice in state['objects']['ice_tiles']}
        
        if start_pos == goal_pos:
            return True
        
        queue = deque([start_pos])
        visited = {start_pos}
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or nx >= grid_size[0] or ny < 0 or ny >= grid_size[1]:
                    continue
                
                # Skip if already visited
                if (nx, ny) in visited:
                    continue
                
                # Skip if ice tile
                if (nx, ny) in ice_positions:
                    continue
                
                # Check if reached goal
                if (nx, ny) == goal_pos:
                    return True
                
                visited.add((nx, ny))
                queue.append((nx, ny))
        
        return False
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"world_{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"