from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import numpy as np
import yaml
import os
from copy import deepcopy
import time

class DeceptiveGridGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        # Get base state from config
        base_state = deepcopy(self.config.get('state_template', {}))
        
        # Execute pipeline to generate world
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID
        world_id = self._generate_world_id(seed)
        
        # Save world
        self._save_world(world_state, world_id)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # Set up random number generator
        rng = np.random.RandomState(seed)
        
        # Get grid dimensions from base_state
        grid_size = base_state['globals']['size']
        rows, cols = grid_size
        
        # Initialize grid with default tiles
        grid = [[base_state['tiles']['default_type'] for _ in range(cols)] for _ in range(rows)]
        
        # Get pipeline parameters from config
        pipeline_config = self.config.get('pipeline', [])
        
        # Find configuration parameters with defaults
        trap_ratio = [0.1, 0.2]
        wall_ratio = [0.05, 0.15] 
        num_safe_tiles = [3, 6]
        
        for step in pipeline_config:
            if step['name'] == 'place_traps':
                trap_ratio = step['args']['trap_ratio']
            elif step['name'] == 'place_fake_walls':
                wall_ratio = step['args']['wall_ratio']
            elif step['name'] == 'place_safe_tiles':
                num_safe_tiles = step['args']['num_safe_tiles']
        
        # Place fake walls (⬛)
        wall_count = int(rng.uniform(wall_ratio[0], wall_ratio[1]) * rows * cols)
        wall_positions = set()
        
        for _ in range(wall_count):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                pos = (rng.randint(0, rows), rng.randint(0, cols))
                if pos not in wall_positions:
                    grid[pos[0]][pos[1]] = '⬛'
                    wall_positions.add(pos)
                    break
                attempts += 1
        
        # Place traps (💰)
        trap_count = int(rng.uniform(trap_ratio[0], trap_ratio[1]) * rows * cols)
        trap_positions = set()
        
        for _ in range(trap_count):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                pos = (rng.randint(0, rows), rng.randint(0, cols))
                if pos not in wall_positions and pos not in trap_positions:
                    grid[pos[0]][pos[1]] = '💰'
                    trap_positions.add(pos)
                    break
                attempts += 1
        
        # Place safe tiles (☠)
        safe_count = rng.randint(num_safe_tiles[0], num_safe_tiles[1] + 1)
        safe_positions = []
        
        for _ in range(safe_count):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                pos = (rng.randint(0, rows), rng.randint(0, cols))
                if pos not in wall_positions and pos not in trap_positions and pos not in safe_positions:
                    grid[pos[0]][pos[1]] = '☠'
                    safe_positions.append(pos)
                    break
                attempts += 1
        
        # Ensure we have at least one safe tile
        if not safe_positions:
            # Force place one safe tile
            while True:
                pos = (rng.randint(0, rows), rng.randint(0, cols))
                if pos not in wall_positions and pos not in trap_positions:
                    grid[pos[0]][pos[1]] = '☠'
                    safe_positions.append(pos)
                    break
        
        # Choose one safe tile as the goal
        goal_pos = safe_positions[rng.randint(0, len(safe_positions))]
        
        # Choose agent starting position
        attempts = 0
        while attempts < 100:
            agent_pos = [rng.randint(0, rows), rng.randint(0, cols)]
            pos_tuple = tuple(agent_pos)
            if (pos_tuple not in wall_positions and 
                pos_tuple not in trap_positions and 
                pos_tuple not in safe_positions and
                grid[agent_pos[0]][agent_pos[1]] == base_state['tiles']['default_type']):
                break
            attempts += 1
        
        # If we couldn't find a good starting position, use a safe default
        if attempts >= 100:
            # Find any empty space
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == base_state['tiles']['default_type']:
                        agent_pos = [i, j]
                        break
                else:
                    continue
                break
        
        # Update state
        base_state['agent']['pos'] = agent_pos
        base_state['tiles']['data'] = grid
        base_state['special']['goal_pos'] = list(goal_pos)
        
        return base_state
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        # Ensure levels directory exists
        levels_dir = "./levels/"
        os.makedirs(levels_dir, exist_ok=True)
        
        # Save to YAML file
        file_path = os.path.join(levels_dir, f"{world_id}.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"world_seed_{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"
