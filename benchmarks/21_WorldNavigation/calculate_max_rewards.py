import yaml
import os
from datetime import datetime
import json

def load_level(level_file):
    """Load a level YAML file"""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward_for_level(level_data, level_filename):
    """
    Calculate the maximum possible reward for a given level.
    
    Based on the environment analysis:
    - Only reward is +1.0 for reaching the goal
    - Agent fails (0 reward) if they step on trap (💰) or run out of steps
    - Maximum theoretical reward is 1.0 if goal is reachable
    """
    
    # Extract key information
    agent_start = level_data['agent']['pos']
    goal_pos = level_data['special']['goal_pos']
    max_steps = level_data['agent']['steps_remaining']
    grid_size = level_data['globals']['size']
    tiles = level_data['tiles']['data']
    
    # Calculate Manhattan distance to goal
    manhattan_distance = abs(agent_start[0] - goal_pos[0]) + abs(agent_start[1] - goal_pos[1])
    
    # Check if goal is theoretically reachable within step limit
    is_reachable = manhattan_distance <= max_steps
    
    # Maximum reward analysis
    if is_reachable:
        max_reward = 1.0
        calculation_method = "goal_reachable_analysis"
        notes = f"Goal at {goal_pos} is reachable from start {agent_start} in {manhattan_distance} steps (≤{max_steps} limit)"
    else:
        max_reward = 0.0
        calculation_method = "goal_unreachable_analysis"
        notes = f"Goal at {goal_pos} unreachable from start {agent_start}: needs {manhattan_distance} steps but limit is {max_steps}"
    
    return {
        "max_reward": max_reward,
        "calculation_method": calculation_method,
        "notes": notes,
        "agent_start": agent_start,
        "goal_position": goal_pos,
        "manhattan_distance": manhattan_distance,
        "step_limit": max_steps,
        "grid_size": grid_size
    }

def main():
    # Environment information
    env_id = "20250918_152710_env_92_grid_world_navigation"
    levels_dir = "./levels/"
    
    # Get all level files
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    # Results storage
    results = {
        "environment_id": env_id,
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    # Process each level
    max_rewards = []
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        
        level_path = os.path.join(levels_dir, level_file)
        level_data = load_level(level_path)
        
        # Calculate maximum reward
        level_result = calculate_max_reward_for_level(level_data, level_file)
        results["levels"][level_file] = level_result
        
        max_rewards.append(level_result["max_reward"])
        
        print(f"  Max reward: {level_result['max_reward']}")
        print(f"  Method: {level_result['calculation_method']}")
        print(f"  Notes: {level_result['notes']}")
        print()
    
    # Calculate summary statistics
    if max_rewards:
        results["summary"] = {
            "total_levels": len(max_rewards),
            "average_max_reward": sum(max_rewards) / len(max_rewards),
            "min_max_reward": min(max_rewards),
            "max_max_reward": max(max_rewards),
            "reachable_levels": sum(1 for r in max_rewards if r > 0),
            "unreachable_levels": sum(1 for r in max_rewards if r == 0)
        }
    
    # Save results to JSON
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {results['summary']['total_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"  Reachable goals: {results['summary']['reachable_levels']}")
    print(f"  Unreachable goals: {results['summary']['unreachable_levels']}")

if __name__ == "__main__":
    main()