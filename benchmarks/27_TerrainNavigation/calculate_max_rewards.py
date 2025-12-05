import yaml
import os
from datetime import datetime
from collections import deque
import json

def load_level(filename):
    """Load a level YAML file"""
    with open(os.path.join('levels', filename), 'r') as f:
        return yaml.safe_load(f)

def find_path_exists(layout, start_pos, goal_pos):
    """Check if a valid path exists from start to goal avoiding water tiles"""
    rows, cols = len(layout), len(layout[0])
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    
    # If start position is water, no path possible
    if layout[start_r][start_c] == 'water':
        return False
    
    # If goal position is water, no path possible  
    if layout[goal_r][goal_c] == 'water':
        return False
    
    # BFS to find path
    queue = deque([(start_r, start_c)])
    visited = set()
    visited.add((start_r, start_c))
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # E, W, S, N
    
    while queue:
        r, c = queue.popleft()
        
        if r == goal_r and c == goal_c:
            return True
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                (nr, nc) not in visited and 
                layout[nr][nc] != 'water'):
                
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    return False

def calculate_max_reward_for_level(filename):
    """Calculate maximum possible reward for a single level"""
    level_data = load_level(filename)
    
    layout = level_data['tiles']['layout']
    start_pos = level_data['start_pos']
    goal_pos = level_data['goal_pos']
    
    # Check if path exists from start to goal
    path_exists = find_path_exists(layout, start_pos, goal_pos)
    
    # In this environment, max reward is 1.0 if goal is reachable, 0.0 otherwise
    max_reward = 1.0 if path_exists else 0.0
    
    return {
        'max_reward': max_reward,
        'calculation_method': 'optimal_path_analysis', 
        'notes': f'Path exists: {path_exists}. Reward = 1.0 if goal reachable, 0.0 otherwise.'
    }

def main():
    # Get all level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    results = {}
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} level files...")
    
    for filename in level_files:
        print(f"Processing {filename}...")
        result = calculate_max_reward_for_level(filename)
        results[filename] = result
        max_rewards.append(result['max_reward'])
        print(f"  Max reward: {result['max_reward']}")
    
    # Create summary statistics
    summary = {
        'total_levels': len(level_files),
        'average_max_reward': sum(max_rewards) / len(max_rewards) if max_rewards else 0,
        'min_max_reward': min(max_rewards) if max_rewards else 0,
        'max_max_reward': max(max_rewards) if max_rewards else 0
    }
    
    # Create final JSON output
    output_data = {
        'environment_id': '20250918_204705_env_83_icy_terrain_navigation',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': results,
        'summary': summary
    }
    
    # Write to JSON file
    with open('level_max_rewards.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSUMMARY:")
    print(f"Total levels analyzed: {summary['total_levels']}")
    print(f"Average max reward: {summary['average_max_reward']:.2f}")
    print(f"Min max reward: {summary['min_max_reward']:.2f}")
    print(f"Max max reward: {summary['max_max_reward']:.2f}")
    print(f"\nResults saved to level_max_rewards.json")

if __name__ == "__main__":
    main()