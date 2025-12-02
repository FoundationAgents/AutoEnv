import sys
import os
sys.path.append('../../../')

from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import random
import yaml
import os

class InvertedBoxEscapeGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Generate world ID first
        world_id = self._generate_world_id(seed)
        
        # Create base state and execute pipeline
        base_state = {
            'globals': {'max_steps': 40},
            'agent': {'pos': [1, 1]},
            'grid': {'size': [8, 8], 'layout': []},
            'objects': {
                'crates': [],
                'storage_tiles': [],
                'exit_pos': [0, 0],
                'covered_tiles': []
            }
        }
        
        world_state = self._execute_pipeline(base_state, seed)
        
        # Save world
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """Execute the generation pipeline to create a solvable level"""
        if seed is not None:
            random.seed(seed)
        
        return self._create_simple_solvable_level()
    
    def _create_simple_solvable_level(self) -> Dict[str, Any]:
        """Create a simple, guaranteed solvable level with required 3-5 crates"""
        
        # Fixed 8x8 grid for more space
        H, W = 8, 8
        
        # Simple layout: just perimeter walls
        layout = []
        for row in range(H):
            layout_row = []
            for col in range(W):
                if row == 0 or row == H-1 or col == 0 or col == W-1:
                    layout_row.append('E')  # Wall
                else:
                    layout_row.append('A')  # Empty floor
            layout.append(layout_row)
        
        # Agent starts in top-left corner
        agent_pos = [1, 1]
        
        # Exit in bottom-right corner
        exit_pos = [H-2, W-2]  # [6, 6]
        
        # Create a simple line of crates that can be pushed to storage
        # Pattern: crates in a line, storage tiles in a line next to them
        
        num_crates = random.choice([3, 4, 5])
        
        # Place crates in middle area, storage tiles below them
        crates = []
        storage_tiles = []
        
        start_col = 2
        for i in range(num_crates):
            crate_pos = [3, start_col + i]  # Row 3, columns 2,3,4...
            storage_pos = [4, start_col + i]  # Row 4, columns 2,3,4...
            
            crates.append(crate_pos)
            storage_tiles.append(storage_pos)
        
        state = {
            'globals': {'max_steps': 40},
            'agent': {'pos': agent_pos},
            'grid': {'size': [H, W], 'layout': layout},
            'objects': {
                'crates': crates,
                'storage_tiles': storage_tiles,
                'exit_pos': exit_pos,
                'covered_tiles': []
            }
        }
        
        return state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        else:
            return f"world_{random.randint(10000, 99999)}"
