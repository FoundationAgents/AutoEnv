#!/usr/bin/env python3

import yaml
import json
import os
from datetime import datetime
from collections import deque
import numpy as np

def load_level(level_file):
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def find_blank_position(grid):
    """Find the position of the blank space (0)."""
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                return (i, j)
    return None

def get_valid_moves(blank_pos):
    """Get valid moves from current blank position."""
    moves = []
    row, col = blank_pos
    
    if row > 0:  # Can slide UP
        moves.append(('SLIDE_UP', (row - 1, col)))
    if row < 2:  # Can slide DOWN
        moves.append(('SLIDE_DOWN', (row + 1, col)))
    if col > 0:  # Can slide LEFT
        moves.append(('SLIDE_LEFT', (row, col - 1)))
    if col < 2:  # Can slide RIGHT
        moves.append(('SLIDE_RIGHT', (row, col + 1)))
    
    return moves

def apply_move(grid, blank_pos, new_pos):
    """Apply a move and return new grid state."""
    new_grid = [row[:] for row in grid]  # Deep copy
    # Swap blank with target position
    new_grid[blank_pos[0]][blank_pos[1]], new_grid[new_pos[0]][new_pos[1]] = \
        new_grid[new_pos[0]][new_pos[1]], new_grid[blank_pos[0]][blank_pos[1]]
    return new_grid

def grid_to_tuple(grid):
    """Convert grid to tuple for hashing."""
    return tuple(tuple(row) for row in grid)

def can_reach_chaos_pattern(start_grid, chaos_pattern, max_steps=30):
    """Use BFS to check if chaos pattern is reachable within max_steps."""
    start_tuple = grid_to_tuple(start_grid)
    target_tuple = grid_to_tuple(chaos_pattern)
    
    if start_tuple == target_tuple:
        return True, 0
    
    # BFS
    queue = deque([(start_grid, 0)])  # (grid, steps)
    visited = {start_tuple}
    
    while queue:
        current_grid, steps = queue.popleft()
        
        if steps >= max_steps:
            continue
            
        blank_pos = find_blank_position(current_grid)
        valid_moves = get_valid_moves(blank_pos)
        
        for move_name, new_pos in valid_moves:
            new_grid = apply_move(current_grid, blank_pos, new_pos)
            new_tuple = grid_to_tuple(new_grid)
            
            if new_tuple == target_tuple:
                return True, steps + 1
            
            if new_tuple not in visited and steps + 1 < max_steps:
                visited.add(new_tuple)
                queue.append((new_grid, steps + 1))
    
    return False, -1

def calculate_max_reward_for_level(level_file):
    """Calculate maximum possible reward for a single level."""
    level_data = load_level(level_file)
    
    start_grid = level_data['board']['grid']
    chaos_pattern = level_data['targets']['chaos_pattern']
    forbidden_pattern = level_data['targets']['forbidden_pattern']
    max_steps = level_data['globals']['max_steps']
    
    # Check if we can reach chaos pattern
    can_reach, min_steps = can_reach_chaos_pattern(start_grid, chaos_pattern, max_steps)
    
    result = {
        'max_reward': 1.0 if can_reach else 0.0,
        'calculation_method': 'bfs_optimal_path_analysis',
        'reachable': can_reach,
        'min_steps_to_chaos': min_steps if can_reach else None,
        'notes': f'Chaos pattern {"reachable" if can_reach else "not reachable"} within {max_steps} steps'
    }
    
    if can_reach:
        result['notes'] += f' (minimum {min_steps} steps required)'
    
    return result

def main():
    """Main function to calculate max rewards for all levels."""
    levels_dir = './levels'
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        'environment_id': '20250919_121700_env_104_spatial_tile_arrangement',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reward_structure': {
            'chaos_pattern_reached': 1.0,
            'forbidden_pattern_reached': 0.0,
            'step_taken': 0.0
        },
        'levels': {},
        'summary': {}
    }
    
    max_rewards = []
    
    print("Calculating maximum rewards for all levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        level_path = os.path.join(levels_dir, level_file)
        
        try:
            level_result = calculate_max_reward_for_level(level_path)
            results['levels'][level_file] = level_result
            max_rewards.append(level_result['max_reward'])
            print(f"  Max reward: {level_result['max_reward']}")
            
        except Exception as e:
            print(f"  Error processing {level_file}: {e}")
            results['levels'][level_file] = {
                'max_reward': 0.0,
                'calculation_method': 'error',
                'error': str(e),
                'notes': 'Error occurred during calculation'
            }
            max_rewards.append(0.0)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_levels': len(max_rewards),
        'average_max_reward': sum(max_rewards) / len(max_rewards) if max_rewards else 0,
        'min_max_reward': min(max_rewards) if max_rewards else 0,
        'max_max_reward': max(max_rewards) if max_rewards else 0,
        'solvable_levels': sum(1 for r in max_rewards if r > 0),
        'unsolvable_levels': sum(1 for r in max_rewards if r == 0)
    }
    
    # Save results to JSON file
    output_file = 'level_max_rewards.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {results['summary']['total_levels']}")
    print(f"  Solvable levels: {results['summary']['solvable_levels']}")
    print(f"  Unsolvable levels: {results['summary']['unsolvable_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']:.2f}")

if __name__ == '__main__':
    main()