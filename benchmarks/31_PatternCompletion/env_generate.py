from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional

class PixelArtGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.art_library = self._load_art_library()
    
    def _load_art_library(self):
        # Create a simple library of 50 10x10 pixel art templates
        library = []
        random.seed(42)  # Fixed seed for consistent library
        
        # Generate 50 different pixel art patterns
        for i in range(50):
            art = []
            for y in range(10):
                row = []
                for x in range(10):
                    # Create simple geometric patterns with semantic consistency
                    if i < 10:  # Squares and rectangles
                        color = 1 if 2 <= x <= 7 and 2 <= y <= 7 else 0
                    elif i < 20:  # Circles
                        center_x, center_y = 4.5, 4.5
                        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                        color = 2 if dist <= 3 else 0
                    elif i < 30:  # Crosses
                        color = 3 if x == 4 or y == 4 else 0
                    elif i < 40:  # Diagonals
                        color = 4 if x == y or x == 9 - y else 0
                    else:  # Random patterns with limited colors
                        color = (x + y + i) % 6
                    row.append(color)
                art.append(row)
            library.append(art)
        
        return library
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Load base state from config
        base_state = self.config.get("state_template", {})
        
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
        # Step 1: Initialize from template
        world_state = self._init_from_template(base_state)
        
        # Step 2: Select ground truth
        world_state = self._select_ground_truth(world_state)
        
        # Step 3: Apply masking
        world_state = self._apply_masking(world_state)
        
        # Step 4: Initialize canvas
        world_state = self._initialize_canvas(world_state)
        
        return world_state
    
    def _init_from_template(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        world_state = {
            "globals": {self.config.get("termination", {}).get("max_steps", 50)},
            "agent": {"cursor_pos": [0, 0]},
            "canvas": {
                "size": [10, 10],
                "pixels": [[0 for _ in range(10)] for _ in range(10)],
                "ground_truth": [[0 for _ in range(10)] for _ in range(10)],
                "masked_positions": [],
                "mask_count": 0
            },
            "palette": {
                "size": 16,
                "colors": list(range(16))
            },
            "episode": {"correct_restorations": 0}
        }
        return world_state
    
    def _select_ground_truth(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        # Select random art from library
        art_id = random.randint(0, len(self.art_library) - 1)
        ground_truth = self.art_library[art_id]
        world_state["canvas"]["ground_truth"] = ground_truth
        return world_state
    
    def _apply_masking(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        # Randomly mask 20-30 pixels
        mask_count = random.randint(15, 20)
        all_positions = [(x, y) for x in range(10) for y in range(10)]
        masked_positions_tuples = random.sample(all_positions, mask_count)
        masked_positions = [[x, y] for x, y in masked_positions_tuples]
        
        world_state["canvas"]["masked_positions"] = masked_positions
        world_state["canvas"]["mask_count"] = mask_count
        return world_state
    
    def _initialize_canvas(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = world_state["canvas"]["ground_truth"]
        masked_positions = world_state["canvas"]["masked_positions"]
        
        # Copy ground truth to canvas, but mask specified positions
        canvas = [[ground_truth[y][x] for x in range(10)] for y in range(10)]
        
        # Replace masked positions with placeholder (will be handled in observation)
        for pos in masked_positions:
            x, y = pos
            canvas[y][x] = -1  # Special value for masked
        
        world_state["canvas"]["pixels"] = canvas
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_seed_{seed}"
        else:
            return f"world_random_{random.randint(1000, 9999)}"