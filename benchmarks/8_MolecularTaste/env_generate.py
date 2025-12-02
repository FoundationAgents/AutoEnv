import os
import sys
import yaml
import random
import math
from typing import Dict, Any, Optional, List, Tuple

# Add the AutoEnv path to sys.path for imports
sys.path.append('../../../')
from base.env.base_generator import WorldGenerator

class MazeChemicalGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        
        # Load state template from config
        base_state = self.config["state_template"].copy()
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        world_state["level_info"]["world_id"] = world_id
        
        # Save world
        self._save_world(world_state, world_id)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = base_state.copy()
        
        # Generate maze layout
        state = self._generate_maze_layout(state)
        
        # Place goal position
        state = self._place_goal_position(state)
        
        # Generate chemical signatures
        state = self._generate_chemical_signatures(state)
        
        # Validate level
        state = self._validate_level(state)
        
        return state
    
    def _generate_maze_layout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        size_x, size_y = state["maze"]["size"]
        wall_density = 0.3
        walls = []
        
        # Generate random walls
        for x in range(size_x):
            for y in range(size_y):
                if random.random() < wall_density:
                    walls.append([x, y])
        
        state["maze"]["walls"] = walls
        return state
    
    def _place_goal_position(self, state: Dict[str, Any]) -> Dict[str, Any]:
        size_x, size_y = state["maze"]["size"]
        walls = set(tuple(wall) for wall in state["maze"]["walls"])
        
        # Find valid goal positions
        valid_positions = []
        for x in range(size_x):
            for y in range(size_y):
                if (x, y) not in walls:
                    valid_positions.append([x, y])
        
        if valid_positions:
            goal_pos = random.choice(valid_positions)
            state["maze"]["goal_pos"] = goal_pos
        
        return state
    
    def _generate_chemical_signatures(self, state: Dict[str, Any]) -> Dict[str, Any]:
        size_x, size_y = state["maze"]["size"]
        goal_pos = tuple(state["maze"]["goal_pos"])
        walls = set(tuple(wall) for wall in state["maze"]["walls"])
        chemical_map = {}
        
        # Calculate max distance for normalization
        max_distance = math.sqrt((size_x-1)**2 + (size_y-1)**2)
        
        for x in range(size_x):
            for y in range(size_y):
                if (x, y) not in walls:
                    # Calculate Manhattan distance to goal
                    distance = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
                    normalized_distance = distance / (size_x + size_y)  # Normalize by max possible Manhattan distance
                    
                    # Generate flavor signature based on distance
                    sweet = max(0.0, min(1.0, 0.9 - normalized_distance + random.uniform(-0.05, 0.05)))
                    umami = max(0.0, min(1.0, 0.8 - normalized_distance + random.uniform(-0.05, 0.05)))
                    bitter = max(0.0, min(1.0, 0.1 + normalized_distance + random.uniform(-0.05, 0.05)))
                    sour = random.uniform(0.1, 0.9)
                    salty = random.uniform(0.1, 0.9)
                    
                    signature = [sweet, sour, salty, bitter, umami]
                    chemical_map[f"{x},{y}"] = signature
        
        state["maze"]["chemical_map"] = chemical_map
        return state
    
    def _validate_level(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Basic validation - ensure goal is reachable
        size_x, size_y = state["maze"]["size"]
        goal_pos = tuple(state["maze"]["goal_pos"])
        walls = set(tuple(wall) for wall in state["maze"]["walls"])
        
        # Find at least one valid start position
        valid_starts = []
        for x in range(size_x):
            for y in range(size_y):
                if (x, y) not in walls and (x, y) != goal_pos:
                    path_length = self._calculate_shortest_path((x, y), goal_pos, walls, (size_x, size_y))
                    if 8 <= path_length <= 12:
                        valid_starts.append((x, y))
        
        # If no valid starts found, reduce wall density
        if not valid_starts:
            # Remove some walls randomly
            current_walls = state["maze"]["walls"]
            if len(current_walls) > size_x * size_y * 0.1:  # Keep at least 10% walls
                walls_to_remove = random.sample(range(len(current_walls)), 
                                               min(len(current_walls) // 3, len(current_walls) - 5))
                new_walls = [wall for i, wall in enumerate(current_walls) if i not in walls_to_remove]
                state["maze"]["walls"] = new_walls
        
        # Update optimal path length
        if valid_starts:
            optimal_length = self._calculate_shortest_path(valid_starts[0], goal_pos, 
                                                         set(tuple(wall) for wall in state["maze"]["walls"]), 
                                                         (size_x, size_y))
            state["level_info"]["optimal_path_length"] = optimal_length
        
        return state
    
    def _calculate_shortest_path(self, start, goal, walls, maze_size):
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
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        os.makedirs("./levels", exist_ok=True)
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'w') as f:
            yaml.dump(world_state, f)
        return world_id
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}_{random.randint(1000, 9999)}"
        else:
            return f"world_{random.randint(10000, 99999)}"
