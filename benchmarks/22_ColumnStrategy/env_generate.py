from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import yaml
import os
import random
from datetime import datetime

class ConnectFourGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = {
            'globals': {
                'max_steps': 40,
                'board_height': 6,
                'board_width': 7
            },
            'agent': {
                'player_id': 1,
                'wins': 0
            },
            'opponent': {
                'player_id': 2,
                'last_move': -1,
                'policy': 'heuristic_depth1'
            },
            'board': {
                'grid': [],
                'filled_columns': []
            },
            'game': {
                'current_player': 1,
                'winner': 0,
                'game_over': False,
                'moves_made': 0
            }
        }
        
        world_state = self._execute_pipeline(base_state, seed)
        world_id = self._generate_world_id(seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = base_state.copy()
        
        # setup_empty_board
        height = 6
        width = 7
        world_state['board']['grid'] = [[0 for _ in range(width)] for _ in range(height)]
        world_state['board']['filled_columns'] = []
        
        # initialize_game_state  
        world_state['game']['current_player'] = 1
        world_state['game']['game_over'] = False
        world_state['game']['winner'] = 0
        world_state['game']['moves_made'] = 0
        
        # setup_opponent_heuristic
        world_state['opponent']['policy'] = 'win_block_random'
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"_seed{seed}" if seed is not None else ""
        return f"connect_four_{timestamp}{seed_str}"