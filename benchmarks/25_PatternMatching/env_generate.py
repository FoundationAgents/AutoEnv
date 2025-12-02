from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import time

class MemoryGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        world_id = self._generate_world_id(seed)
        
        base_state = {
            'globals': {
                'max_steps': 40,
                'total_pairs': 8,
                'grid_size': [4, 4]
            },
            'game': {
                'cards': [],
                'card_states': [],
                'revealed_positions': [],
                'cleared_pairs': 0,
                'current_revealed_symbol': -1,
                'explored_positions': []
            },
            'agent': {
                'steps_remaining': 40
            }
        }
        
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f)
            
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = base_state.copy()
        
        symbols = []
        for i in range(8):
            symbols.extend([i, i])
        
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(symbols)
        else:
            random.shuffle(symbols)
        
        world_state['game']['cards'] = symbols
        world_state['game']['card_states'] = [0] * 16
        world_state['game']['revealed_positions'] = []
        world_state['game']['cleared_pairs'] = 0
        world_state['game']['current_revealed_symbol'] = -1
        world_state['game']['explored_positions'] = []
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"memory_world_seed_{seed}_{timestamp}"
        else:
            return f"memory_world_{timestamp}"