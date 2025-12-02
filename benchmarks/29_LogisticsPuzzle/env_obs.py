from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
import numpy as np

class FullGridObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        grid_size = env_state['grid']['size']
        H, W = grid_size[0], grid_size[1]
        
        # Initialize grid with safe floor 'A'
        grid = [['A' for _ in range(W)] for _ in range(H)]
        
        # Place walls
        layout = env_state['grid']['layout']
        for row in range(H):
            for col in range(W):
                if row < len(layout) and col < len(layout[row]):
                    if layout[row][col] == 'E':
                        grid[row][col] = 'E'
        
        # Place crates (lethal)
        for crate_pos in env_state['objects']['crates']:
            row, col = crate_pos[0], crate_pos[1]
            if 0 <= row < H and 0 <= col < W:
                grid[row][col] = 'B'
        
        # Place storage tiles (dangerous if uncovered)
        covered_tiles = set(tuple(pos) for pos in env_state['objects']['covered_tiles'])
        for storage_pos in env_state['objects']['storage_tiles']:
            row, col = storage_pos[0], storage_pos[1]
            if 0 <= row < H and 0 <= col < W:
                if tuple(storage_pos) not in covered_tiles:
                    grid[row][col] = 'C'
                # If covered, it remains 'A' (safe floor)
        
        # Place exit
        exit_pos = env_state['objects']['exit_pos']
        exit_row, exit_col = exit_pos[0], exit_pos[1]
        if 0 <= exit_row < H and 0 <= exit_col < W:
            grid[exit_row][exit_col] = 'D'
        
        # Place agent (overwrites everything else at agent position)
        agent_pos = env_state['agent']['pos']
        agent_row, agent_col = agent_pos[0], agent_pos[1]
        if 0 <= agent_row < H and 0 <= agent_col < W:
            grid[agent_row][agent_col] = 'P'
        
        # Calculate coverage progress
        total_storage = len(env_state['objects']['storage_tiles'])
        covered_count = len(env_state['objects']['covered_tiles'])
        
        max_steps = env_state['globals']['max_steps']
        
        return {
            'grid': grid,
            'grid_size': [H, W],
            'agent_pos': agent_pos,
            'step_count': t + 1,  # Fix: Show step number starting from 1
            'max_steps': max_steps,
            'steps_remaining': max_steps - t,
            'covered_count': covered_count,
            'total_storage': total_storage,
            'all_storage_covered': covered_count == total_storage
        }