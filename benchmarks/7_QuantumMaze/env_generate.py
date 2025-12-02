from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from datetime import datetime

class QuantumMazeGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Get base state from config
        base_state = self.config["state_template"]
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID
        world_id = self._generate_world_id(seed)
        
        # Save world
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        self._save_world(world_state, save_path)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # Step 1: Initialize from template
        world_state = {
            "globals": base_state["globals"].copy(),
            "agent": base_state["agent"].copy(),
            "maze": {
                "size": base_state["maze"]["size"].copy(),
                "quantum_walls": {},
                "collapsed_walls": {},
                "wall_probabilities": {}
            }
        }
        
        # Step 2: Generate quantum probabilities
        generator_config = self.config["generator"]
        min_prob = 0.2
        max_prob = 0.5
        
        # Find args in pipeline
        for step in generator_config["pipeline"]:
            if step["name"] == "generate_quantum_probabilities":
                min_prob = step["args"].get("min_prob", 0.2)
                max_prob = step["args"].get("max_prob", 0.5)
                break
        
        grid_size = world_state["globals"]["grid_size"]
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                cell_key = f"{x},{y}"
                prob = random.uniform(min_prob, max_prob)
                world_state["maze"]["wall_probabilities"][cell_key] = prob
        
        # Step 3: Setup boundaries
        # Ensure start and exit positions are accessible
        start_pos = world_state["globals"]["start_pos"]
        exit_pos = world_state["globals"]["exit_pos"]
        
        # Force start and exit to have low wall probability
        start_key = f"{start_pos[0]},{start_pos[1]}"
        exit_key = f"{exit_pos[0]},{exit_pos[1]}"
        world_state["maze"]["wall_probabilities"][start_key] = 0.0
        world_state["maze"]["wall_probabilities"][exit_key] = 0.0
        
        return world_state
    
    def _save_world(self, world_state: Dict[str, Any], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if seed is not None:
            return f"quantum_maze_seed_{seed}_{timestamp}"
        else:
            return f"quantum_maze_{timestamp}"