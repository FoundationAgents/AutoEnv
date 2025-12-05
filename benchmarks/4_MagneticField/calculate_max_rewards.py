#!/usr/bin/env python3

import yaml
import json
from datetime import datetime
import os

def calculate_max_reward_for_level(level_file):
    """
    Calculate maximum possible reward for a given level.
    For this environment, max reward is always 1.0 (binary success/failure).
    """
    with open(level_file, 'r') as f:
        level_data = yaml.safe_load(f)
    
    # Extract the encoded message (correct answer)
    encoded_message = level_data['grid']['encoded_message']
    max_steps = level_data['globals']['max_steps']
    
    # Maximum reward is 1.0 for correct submission, 0.0 for incorrect
    # Since this is asking for theoretical maximum, we assume optimal play
    max_reward = 1.0
    
    calculation_notes = f"Optimal agent submits correct answer '{encoded_message}' within {max_steps} steps"
    
    return {
        "max_reward": max_reward,
        "calculation_method": "binary_reward_analysis", 
        "encoded_message": encoded_message,
        "max_steps": max_steps,
        "notes": calculation_notes
    }

def main():
    # Get all level files
    levels_dir = "./levels/"
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": "20250905_002350_env_magnetic_field_p",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_structure": "Binary: 1.0 for correct answer, 0.0 for incorrect",
        "levels": {},
        "summary": {}
    }
    
    total_levels = 0
    total_max_reward = 0.0
    min_reward = float('inf')
    max_reward = float('-inf')
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        level_result = calculate_max_reward_for_level(level_path)
        
        results["levels"][level_file] = level_result
        
        reward = level_result["max_reward"]
        total_max_reward += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        total_levels += 1
    
    # Calculate summary statistics
    avg_reward = total_max_reward / total_levels if total_levels > 0 else 0.0
    
    results["summary"] = {
        "total_levels": total_levels,
        "average_max_reward": avg_reward,
        "min_max_reward": min_reward,
        "max_max_reward": max_reward,
        "total_possible_reward": total_max_reward
    }
    
    # Write results to JSON file
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Maximum reward calculation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total levels analyzed: {total_levels}")
    print(f"Maximum possible reward per level: {max_reward}")
    print(f"Total maximum possible reward across all levels: {total_max_reward}")

if __name__ == "__main__":
    main()