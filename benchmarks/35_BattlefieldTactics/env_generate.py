from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import uuid
from copy import deepcopy

class SquadWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Load base state template
        base_state = deepcopy(self.config["state_template"])
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID
        world_id = self._generate_world_id(seed)
        
        # Save world
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        pipeline_config = self.config["generator"]["pipeline"]
        
        for step in pipeline_config:
            if step["name"] == "init_from_template":
                continue  # Already have base_state
                
            elif step["name"] == "randomize_squad_strength":
                args = step["args"]
                for squad in base_state["squads"]:
                    squad["strength"] = random.randint(args["min_strength"], args["max_strength"])
                    
            elif step["name"] == "generate_terrain":
                args = step["args"]
                self._generate_terrain(base_state, args["wall_density"], args["forest_density"])
                
            elif step["name"] == "place_enemy_camps":
                args = step["args"]
                self._place_enemy_camps(base_state, args)
                
            elif step["name"] == "initialize_fog_of_war":
                args = step["args"]
                self._initialize_fog_of_war(base_state, args["sensor_radius"])
        
        return base_state
    
    def _generate_terrain(self, state: Dict[str, Any], wall_density: float, forest_density: float):
        grid_size = state["globals"]["grid_size"]
        walls = []
        forests = []
        
        # Avoid spawn area (first 4x4 corner)
        spawn_area = [(x, y) for x in range(4) for y in range(4)]
        
        total_cells = grid_size[0] * grid_size[1] - len(spawn_area)
        num_walls = int(total_cells * wall_density)
        num_forests = int(total_cells * forest_density)
        
        # Generate walls
        for _ in range(num_walls):
            while True:
                x = random.randint(0, grid_size[0] - 1)
                y = random.randint(0, grid_size[1] - 1)
                if (x, y) not in spawn_area and [x, y] not in walls:
                    walls.append([x, y])
                    break
        
        # Generate forests
        for _ in range(num_forests):
            while True:
                x = random.randint(0, grid_size[0] - 1)
                y = random.randint(0, grid_size[1] - 1)
                if (x, y) not in spawn_area and [x, y] not in walls and [x, y] not in forests:
                    forests.append([x, y])
                    break
        
        state["terrain"]["walls"] = walls
        state["terrain"]["forests"] = forests
    
    def _place_enemy_camps(self, state: Dict[str, Any], args: Dict[str, Any]):
        grid_size = state["globals"]["grid_size"]
        walls = state["terrain"]["walls"]
        forests = state["terrain"]["forests"]
        min_dist = args["min_distance_from_spawn"]
        
        occupied_positions = set(tuple(pos) for pos in walls + forests)
        occupied_positions.update((x, y) for x in range(4) for y in range(4))  # Spawn area
        
        camps = []
        for i in range(args["num_camps"]):
            while True:
                x = random.randint(0, grid_size[0] - 1)
                y = random.randint(0, grid_size[1] - 1)
                
                # Check minimum distance from spawn
                if x < min_dist or y < min_dist:
                    continue
                    
                if (x, y) not in occupied_positions:
                    strength = random.randint(args["min_strength"], args["max_strength"])
                    camps.append({
                        "id": i,
                        "pos": [x, y],
                        "strength": strength,
                        "discovered": False
                    })
                    occupied_positions.add((x, y))
                    break
        
        state["enemy_camps"] = camps
    
    def _initialize_fog_of_war(self, state: Dict[str, Any], sensor_radius: int):
        grid_size = state["globals"]["grid_size"]
        explored = [[False for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        
        # Mark initial squad sensor areas as explored
        for squad in state["squads"]:
            pos = squad["pos"]
            for dx in range(-sensor_radius, sensor_radius + 1):
                for dy in range(-sensor_radius, sensor_radius + 1):
                    x, y = pos[0] + dx, pos[1] + dy
                    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                        explored[x][y] = True
        
        state["visibility_map"]["explored"] = explored
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_s{seed}"
        else:
            return f"world_{str(uuid.uuid4())[:8]}"