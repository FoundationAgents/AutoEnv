import yaml
import os
from datetime import datetime
import json

def load_level(level_file):
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def find_treasure_position(tiles):
    """Find the position of the treasure in the grid."""
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile == "Treasure":
                return (x, y)
    return None

def find_agent_position(level_data):
    """Get agent starting position."""
    return tuple(level_data['agent']['pos'])

def is_reachable(start_pos, target_pos, tiles):
    """Check if target position is reachable from start position using BFS."""
    if start_pos == target_pos:
        return True
    
    from collections import deque
    
    queue = deque([start_pos])
    visited = set([start_pos])
    width = len(tiles[0])
    height = len(tiles)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < width and 0 <= ny < height:
                # Check if position is visitable (not Wall or Water)
                if tiles[ny][nx] not in ["Wall", "Water"] and (nx, ny) not in visited:
                    if (nx, ny) == target_pos:
                        return True
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return False

def calculate_max_reward(level_file):
    """Calculate maximum possible reward for a level."""
    level_data = load_level(level_file)
    
    tiles = level_data['tiles']
    agent_pos = find_agent_position(level_data)
    treasure_pos = find_treasure_position(tiles)
    
    if treasure_pos is None:
        return {
            "max_reward": 0.0,
            "calculation_method": "no_treasure_found",
            "notes": "No treasure found in level"
        }
    
    # Check if treasure is reachable from agent position
    if is_reachable(agent_pos, treasure_pos, tiles):
        return {
            "max_reward": 1.0,
            "calculation_method": "pathfinding_analysis",
            "notes": "Treasure is reachable from agent starting position"
        }
    else:
        return {
            "max_reward": 0.0,
            "calculation_method": "pathfinding_analysis", 
            "notes": "Treasure is not reachable from agent starting position"
        }

def main():
    """Main function to process all levels."""
    levels_dir = "./levels/"
    results = {
        "environment_id": "20250918_183647_env_91_grid_world_navigation",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort to ensure consistent ordering
    
    max_rewards = []
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        reward_info = calculate_max_reward(level_path)
        results["levels"][level_file] = reward_info
        max_rewards.append(reward_info["max_reward"])
        print(f"Processed {level_file}: max_reward = {reward_info['max_reward']}")
    
    # Calculate summary statistics
    results["summary"] = {
        "total_levels": len(max_rewards),
        "average_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0,
        "min_max_reward": min(max_rewards) if max_rewards else 0,
        "max_max_reward": max(max_rewards) if max_rewards else 0
    }
    
    # Save results to JSON file
    with open("level_max_rewards.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary:")
    print(f"Total levels processed: {results['summary']['total_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"Min max reward: {results['summary']['min_max_reward']}")
    print(f"Max max reward: {results['summary']['max_max_reward']}")
    print(f"\nResults saved to level_max_rewards.json")

if __name__ == "__main__":
    main()