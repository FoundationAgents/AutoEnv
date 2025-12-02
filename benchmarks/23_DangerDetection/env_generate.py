from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy

class TreasureWorldGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is None:
            seed = random.randint(0, 999999)
        
        # Base state from state_template
        base_state = {
            "globals": {
                "grid_size": [8, 8],
                "max_steps": 30
            },
            "agent": {
                "pos": [0, 0]
            },
            "tiles": {
                "size": [8, 8],
                "default_type": "unrevealed",
                "icons": {
                    "bomb_count": 1,
                    "flower_count": 10,
                    "empty_count": 53
                }
            },
            "grid": {
                "revealed": {},
                "icons": {}
            }
        }
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID and save
        world_id = f"world_{seed}"
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to file
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # Step 1: Deep copy base state
        world_state = deepcopy(base_state)
        
        # Step 2: Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Step 3: Generate all grid coordinates
        all_positions = []
        for x in range(8):
            for y in range(8):
                all_positions.append((x, y))
        
        # Step 4: Randomly shuffle positions
        random.shuffle(all_positions)
        
        # Step 5: Assign icons to positions
        # Convert tuples to strings for YAML compatibility
        icons_dict = {}
        
        # First position gets bomb
        bomb_pos = all_positions[0]
        icons_dict[f"{bomb_pos[0]},{bomb_pos[1]}"] = "bomb"
        
        # Next 10 positions get flowers
        for i in range(1, 11):
            flower_pos = all_positions[i]
            icons_dict[f"{flower_pos[0]},{flower_pos[1]}"] = "flower"
        
        # Remaining 53 positions get empty
        for i in range(11, 64):
            empty_pos = all_positions[i]
            icons_dict[f"{empty_pos[0]},{empty_pos[1]}"] = "empty"
        
        world_state["grid"]["icons"] = icons_dict
        
        # Step 6: Initialize all tiles as unrevealed
        world_state["grid"]["revealed"] = {}
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is None:
            seed = random.randint(0, 999999)
        return f"world_{seed}"
