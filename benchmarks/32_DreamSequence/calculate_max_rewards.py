import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
import sys

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def find_shortest_path(start_room: int, key_room: int, portal_room: int, 
                      connections: Dict[int, Dict[str, int]], 
                      room_types: Dict[int, str]) -> Tuple[bool, int]:
    """
    Find if there's a path from start -> key -> portal and calculate minimum steps.
    Returns (is_possible, min_steps)
    """
    from collections import deque
    
    def bfs(start: int, target: int) -> Tuple[bool, int]:
        """BFS to find shortest path between two rooms."""
        if start == target:
            return True, 0
            
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current_room, steps = queue.popleft()
            
            if current_room not in connections:
                continue
                
            room_connections = connections[current_room]
            room_type = room_types.get(current_room, 'Normal')
            
            # For each door in current room
            for door_color, next_room in room_connections.items():
                if next_room in visited:
                    continue
                    
                visited.add(next_room)
                new_steps = steps + 1
                
                # Add extra step for Time-Slow rooms (need to WAIT first)
                if room_type == 'Time-Slow' and steps > 0:  # Not the first move
                    new_steps += 1
                
                if next_room == target:
                    return True, new_steps
                    
                queue.append((next_room, new_steps))
        
        return False, float('inf')
    
    # Path from start to key
    can_reach_key, steps_to_key = bfs(start_room, key_room)
    if not can_reach_key:
        return False, float('inf')
    
    # Path from key to portal
    can_reach_portal, steps_to_portal = bfs(key_room, portal_room)
    if not can_reach_portal:
        return False, float('inf')
    
    # Add 1 step for picking up the key
    total_steps = steps_to_key + 1 + steps_to_portal
    
    return True, total_steps

def calculate_max_reward(level_data: Dict[str, Any]) -> Tuple[float, str, str]:
    """
    Calculate maximum possible reward for a level.
    Returns (max_reward, calculation_method, notes)
    """
    try:
        # Extract level information
        start_room = level_data['agent']['current_room']
        key_location = level_data['world']['key_location']
        portal_room = level_data['world']['portal_room']
        connections = level_data['world']['connections']
        rooms = level_data['world']['rooms']
        max_steps = level_data['globals']['max_steps']
        
        # Convert room types to dict for easier access
        room_types = {int(room_id): room_data['type'] for room_id, room_data in rooms.items()}
        
        # Convert connections keys to int
        int_connections = {}
        for room_id, room_connections in connections.items():
            int_connections[int(room_id)] = room_connections
        
        # Find if level is completable and minimum steps required
        is_possible, min_steps = find_shortest_path(
            start_room, key_location, portal_room, int_connections, room_types
        )
        
        if not is_possible:
            return 0.0, "path_analysis", "No valid path from start to key to portal"
        
        if min_steps > max_steps:
            return 0.0, "path_analysis", f"Minimum required steps ({min_steps}) exceeds limit ({max_steps})"
        
        # If completable within step limit, max reward is 1.0
        return 1.0, "optimal_path_analysis", f"Completable in {min_steps} steps (limit: {max_steps})"
        
    except Exception as e:
        return 0.0, "error", f"Error during calculation: {str(e)}"

def main():
    """Main function to calculate max rewards for all levels."""
    levels_dir = "levels"
    results = {
        "environment_id": "20250919_185220_env_10_NEW_dream_sequence_n",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        print(f"Processing {level_file}...")
        
        try:
            level_data = load_level(level_path)
            max_reward, method, notes = calculate_max_reward(level_data)
            
            results["levels"][level_file] = {
                "max_reward": max_reward,
                "calculation_method": method,
                "notes": notes
            }
            
            max_rewards.append(max_reward)
            print(f"  - Max reward: {max_reward}")
            
        except Exception as e:
            print(f"  - Error: {str(e)}")
            results["levels"][level_file] = {
                "max_reward": 0.0,
                "calculation_method": "error",
                "notes": f"Failed to process level: {str(e)}"
            }
            max_rewards.append(0.0)
    
    # Calculate summary statistics
    results["summary"] = {
        "total_levels": len(max_rewards),
        "average_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
        "min_max_reward": min(max_rewards) if max_rewards else 0.0,
        "max_max_reward": max(max_rewards) if max_rewards else 0.0,
        "completable_levels": sum(1 for r in max_rewards if r > 0),
        "reward_distribution": {
            "0.0": max_rewards.count(0.0),
            "1.0": max_rewards.count(1.0)
        }
    }
    
    # Save results
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  - Total levels: {results['summary']['total_levels']}")
    print(f"  - Completable levels: {results['summary']['completable_levels']}")
    print(f"  - Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"  - Reward distribution: {results['summary']['reward_distribution']}")

if __name__ == "__main__":
    main()