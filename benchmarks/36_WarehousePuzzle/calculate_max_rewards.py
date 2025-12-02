import yaml
import os
from collections import deque
from datetime import datetime
import json

def load_level(level_file):
    """Load a level YAML file and return the parsed data."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def is_solvable(level_data, max_steps=40):
    """
    Check if a level is solvable within max_steps using BFS pathfinding.
    Returns True if solvable (max reward = 1.0), False otherwise (max reward = 0.0).
    """
    # Extract initial state
    agent_pos = tuple(level_data['agent']['pos'])
    boxes = [tuple(box['pos']) for box in level_data['objects']['boxes']]
    docks = set(tuple(dock['pos']) for dock in level_data['objects']['docks'])
    grid = level_data['tiles']['grid']
    
    # Create initial state: (agent_pos, frozenset(box_positions))
    initial_state = (agent_pos, frozenset(boxes))
    
    # BFS to find solution
    queue = deque([(initial_state, 0)])  # (state, steps)
    visited = {initial_state}
    
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # North, South, East, West
    
    while queue:
        (agent_pos, box_positions), steps = queue.popleft()
        
        # Check if all boxes are on docks (winning condition)
        if box_positions.issubset(docks):
            return True
        
        # Check if we've exceeded max steps
        if steps >= max_steps:
            continue
            
        # Try each direction
        for dx, dy in directions:
            new_agent_x, new_agent_y = agent_pos[0] + dx, agent_pos[1] + dy
            
            # Check bounds
            if not (0 <= new_agent_x < 10 and 0 <= new_agent_y < 10):
                continue
                
            # Check if moving into wall
            if grid[new_agent_x][new_agent_y] == 'wall':
                continue
            
            new_agent_pos = (new_agent_x, new_agent_y)
            new_box_positions = set(box_positions)
            
            # Check if there's a box at the new position
            if new_agent_pos in box_positions:
                # Try to push the box
                box_new_x, box_new_y = new_agent_x + dx, new_agent_y + dy
                box_new_pos = (box_new_x, box_new_y)
                
                # Check if box can be pushed (bounds, walls, other boxes)
                if (0 <= box_new_x < 10 and 0 <= box_new_y < 10 and 
                    grid[box_new_x][box_new_y] != 'wall' and 
                    box_new_pos not in box_positions):
                    
                    # Update box positions
                    new_box_positions.remove(new_agent_pos)
                    new_box_positions.add(box_new_pos)
                else:
                    # Can't push box, skip this move
                    continue
            
            # Create new state
            new_state = (new_agent_pos, frozenset(new_box_positions))
            
            # Skip if already visited
            if new_state in visited:
                continue
                
            visited.add(new_state)
            queue.append((new_state, steps + 1))
    
    return False

def calculate_max_rewards():
    """Calculate maximum rewards for all levels."""
    levels_dir = "./levels/"
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": "20250918_202145_env_81_warehouse_logistics_puzzle",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    total_levels = 0
    solvable_count = 0
    max_rewards = []
    
    for level_file in level_files:
        print(f"Analyzing {level_file}...")
        level_path = os.path.join(levels_dir, level_file)
        level_data = load_level(level_path)
        
        # Check if level is solvable
        solvable = is_solvable(level_data)
        max_reward = 1.0 if solvable else 0.0
        
        results["levels"][level_file] = {
            "max_reward": max_reward,
            "calculation_method": "BFS_pathfinding_analysis",
            "notes": f"Binary reward system: 1.0 if solvable within 40 steps, 0.0 otherwise. Level is {'solvable' if solvable else 'not solvable'}.",
            "total_boxes": level_data['level_info']['total_boxes'],
            "solvable": solvable
        }
        
        total_levels += 1
        max_rewards.append(max_reward)
        if solvable:
            solvable_count += 1
        
        print(f"  Max reward: {max_reward} ({'solvable' if solvable else 'not solvable'})")
    
    # Calculate summary statistics
    results["summary"] = {
        "total_levels": total_levels,
        "solvable_levels": solvable_count,
        "unsolvable_levels": total_levels - solvable_count,
        "average_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
        "min_max_reward": min(max_rewards) if max_rewards else 0.0,
        "max_max_reward": max(max_rewards) if max_rewards else 0.0
    }
    
    return results

if __name__ == "__main__":
    results = calculate_max_rewards()
    
    # Save to JSON file
    with open("level_max_rewards.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted analysis of {results['summary']['total_levels']} levels")
    print(f"Solvable levels: {results['summary']['solvable_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")
    print("Results saved to level_max_rewards.json")