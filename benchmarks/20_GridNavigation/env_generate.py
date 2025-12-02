from base.env.base_generator import WorldGenerator
import random
import yaml
import os
import time
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

class MazeGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        world_id = self._generate_world_id(seed)
        
        state_template = {
            "globals": {
                "width": 11,
                "height": 11,
                "max_steps": 40,
                "tile_types": ["Empty", "Wall", "Water", "Fire", "Treasure", "Agent"],
                "directions": ["N", "E", "S", "W"]
            },
            "agent": {
                "pos": [0, 0],
                "facing": "N",
                "steps_left": 40
            },
            "tiles": [[None for _ in range(11)] for _ in range(11)],
            "objects": []
        }
        
        world_state = self._execute_pipeline(deepcopy(state_template), seed)
        self._save_world(world_state, world_id)
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        
        width = base_state["globals"]["width"]
        height = base_state["globals"]["height"]
        
        # Initialize with Empty
        for y in range(height):
            for x in range(width):
                base_state["tiles"][y][x] = "Empty"
        
        # 1. Carve outer wall
        self._carve_outer_wall(base_state)
        
        # 2. Generate maze interior
        self._generate_maze_interior(base_state)
        
        # 3. Add water pools
        self._add_water_pools(base_state)
        
        # 4. Scatter fire pits
        self._scatter_fire_pits(base_state)
        
        # 5. Place treasure
        self._place_treasure(base_state)
        
        # 6. Place agent
        self._place_agent(base_state)
        
        return base_state
    
    def _carve_outer_wall(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        for y in range(height):
            for x in range(width):
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    tiles[y][x] = "Wall"
    
    def _generate_maze_interior(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        # Simple maze generation - add some walls
        for y in range(2, height - 2, 2):
            for x in range(2, width - 2, 2):
                if random.random() < 0.3:
                    tiles[y][x] = "Wall"
                    # Add connecting walls
                    if random.random() < 0.5 and x + 1 < width - 1:
                        tiles[y][x + 1] = "Wall"
                    if random.random() < 0.5 and y + 1 < height - 1:
                        tiles[y + 1][x] = "Wall"
    
    def _add_water_pools(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        empty_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tiles[y][x] == "Empty":
                    empty_cells.append((x, y))
        
        water_count = int(len(empty_cells) * 0.05)
        water_cells = random.sample(empty_cells, min(water_count, len(empty_cells)))
        
        for x, y in water_cells:
            tiles[y][x] = "Water"
    
    def _scatter_fire_pits(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        empty_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tiles[y][x] == "Empty":
                    empty_cells.append((x, y))
        
        fire_count = random.randint(5, 8)
        fire_cells = random.sample(empty_cells, min(fire_count, len(empty_cells)))
        
        for x, y in fire_cells:
            tiles[y][x] = "Fire"
    
    def _place_treasure(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        empty_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tiles[y][x] == "Empty":
                    empty_cells.append((x, y))
        
        if empty_cells:
            treasure_pos = random.choice(empty_cells)
            tiles[treasure_pos[1]][treasure_pos[0]] = "Treasure"
    
    def _place_agent(self, state: Dict[str, Any]):
        width = state["globals"]["width"]
        height = state["globals"]["height"]
        tiles = state["tiles"]
        
        empty_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if tiles[y][x] == "Empty":
                    empty_cells.append((x, y))
        
        if empty_cells:
            agent_pos = random.choice(empty_cells)
            state["agent"]["pos"] = list(agent_pos)
            state["agent"]["facing"] = random.choice(["N", "E", "S", "W"])
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        os.makedirs("./levels", exist_ok=True)
        file_path = f"./levels/{world_id}.yaml"
        with open(file_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        return world_id
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000) % 1000000
        if seed is not None:
            return f"world_s{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"