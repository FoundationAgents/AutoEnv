from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional, List, Tuple
import random
import yaml
import os
from collections import deque
from copy import deepcopy

class EMWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = deepcopy(self.config['state_template'])
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
            
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = deepcopy(base_state)
        grid_size = state['grid']['size']
        
        # Initialize walls list and em_field values
        state['walls'] = []
        state['em_field']['values'] = [[0 for _ in range(grid_size[0])] for _ in range(grid_size[1])]
        
        # Execute pipeline steps
        self._generate_walls(state, grid_size)
        self._place_vulnerability_node(state, grid_size)
        self._calculate_em_field(state, grid_size)
        self._spawn_agent(state, grid_size)
        
        return state
    
    def _generate_walls(self, state: Dict[str, Any], grid_size: List[int]):
        width, height = grid_size
        total_tiles = width * height
        target_walls = int(total_tiles * 0.2)
        
        walls = set()
        free_tiles = set((x, y) for x in range(width) for y in range(height))
        
        # Generate walls while ensuring connectivity
        attempts = 0
        max_attempts = target_walls * 3
        
        while len(walls) < target_walls and attempts < max_attempts:
            attempts += 1
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            
            if (x, y) not in walls:
                temp_walls = walls | {(x, y)}
                if self._check_connectivity(free_tiles - temp_walls, grid_size):
                    walls.add((x, y))
        
        state['walls'] = [list(wall) for wall in walls]
    
    def _check_connectivity(self, free_tiles: set, grid_size: List[int]) -> bool:
        if not free_tiles:
            return False
            
        start = next(iter(free_tiles))
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in free_tiles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return len(visited) == len(free_tiles)
    
    def _place_vulnerability_node(self, state: Dict[str, Any], grid_size: List[int]):
        width, height = grid_size
        walls = set(tuple(wall) if isinstance(wall, list) else wall for wall in state['walls'])
        free_tiles = [(x, y) for x in range(width) for y in range(height) if (x, y) not in walls]
        
        node_pos = random.choice(free_tiles)
        state['vulnerability_node']['pos'] = list(node_pos)
    
    def _calculate_em_field(self, state: Dict[str, Any], grid_size: List[int]):
        width, height = grid_size
        walls = set(tuple(wall) if isinstance(wall, list) else wall for wall in state['walls'])
        node_pos = tuple(state['vulnerability_node']['pos'])
        
        # Initialize field values
        field_values = [[0 for _ in range(width)] for _ in range(height)]
        
        # BFS to calculate field with Faraday shielding
        visited = set()
        queue = deque([(node_pos, 3)])  # (position, field_strength)
        
        while queue:
            (x, y), strength = queue.popleft()
            
            if (x, y) in visited or strength <= 0:
                continue
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if (x, y) in walls:
                continue
                
            visited.add((x, y))
            field_values[y][x] = strength
            
            # Propagate to adjacent tiles
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    queue.append(((nx, ny), strength - 1))
        
        state['em_field']['values'] = field_values
    
    def _spawn_agent(self, state: Dict[str, Any], grid_size: List[int]):
        width, height = grid_size
        walls = set(tuple(wall) if isinstance(wall, list) else wall for wall in state['walls'])
        free_tiles = [(x, y) for x in range(width) for y in range(height) if (x, y) not in walls]
        
        agent_pos = random.choice(free_tiles)
        facing_options = ['north', 'south', 'east', 'west']
        agent_facing = random.choice(facing_options)
        
        state['agent']['pos'] = list(agent_pos)
        state['agent']['facing'] = agent_facing
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        else:
            return f"world_{random.randint(1000, 9999)}"