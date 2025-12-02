from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy
import time

class MemoryWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        state_template = self.config.get("state_template", {})
        base_state = deepcopy(state_template)
        
        complete_world = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        self._save_world(complete_world, save_path)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # For a 4x4 grid with 16 positions, we need 8 unique symbols
        # Each symbol appears exactly twice to form 8 pairs
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        # Create pairing relationships - in this mismatched memory game,
        # visually different symbols can form pairs
        shuffled_symbols = symbols.copy()
        random.shuffle(shuffled_symbols)
        
        symbol_pairs = {}
        for i in range(0, len(shuffled_symbols), 2):
            sym1, sym2 = shuffled_symbols[i], shuffled_symbols[i + 1]
            symbol_pairs[sym1] = sym2
            symbol_pairs[sym2] = sym1
        
        # Create board with each symbol appearing exactly twice
        all_cards = []
        for symbol in symbols:
            all_cards.extend([symbol, symbol])  # Each symbol appears twice
        
        # Shuffle the positions
        random.shuffle(all_cards)
        
        # Arrange into 4x4 grid
        grid_size = base_state.get("globals", {}).get("grid_size", 4)
        cards_2d = []
        card_states_2d = []
        
        idx = 0
        for i in range(grid_size):
            row_cards = []
            row_states = []
            for j in range(grid_size):
                row_cards.append(all_cards[idx])
                row_states.append(0)  # All cards start face-down
                idx += 1
            cards_2d.append(row_cards)
            card_states_2d.append(row_states)
        
        # Update state
        base_state["board"]["cards"] = cards_2d
        base_state["board"]["card_states"] = card_states_2d
        base_state["game"]["symbol_pairs"] = symbol_pairs
        base_state["game"]["discovered_pairs"] = 0
        base_state["game"]["seen_symbols"] = []
        base_state["game"]["step_count"] = 0
        base_state["game"]["cumulative_reward"] = 0.0
        
        return base_state
    
    def _save_world(self, world_state: Dict[str, Any], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000000)
        if seed is not None:
            return f"memory_world_{seed}_{timestamp}"
        else:
            return f"memory_world_{timestamp}"
