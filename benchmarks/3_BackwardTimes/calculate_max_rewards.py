#!/usr/bin/env python3

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file and return the parsed data."""
    with open(f"./levels/{level_file}", 'r') as f:
        return yaml.safe_load(f)

def analyze_level_max_reward(level_data: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    """
    Analyze a level and calculate its theoretical maximum reward.
    
    Based on the environment analysis:
    - The environment has a binary reward structure
    - Maximum reward = 1.0 for correct full identification
    - Reward = 0.0 for incorrect identification or timeout
    
    The theoretical maximum is always 1.0 if the level is solvable.
    """
    
    # Extract ground truth information
    ground_truth = level_data.get('ground_truth', {})
    root_cause_time = ground_truth.get('root_cause_time')
    root_cause_perpetrator = ground_truth.get('root_cause_perpetrator')
    root_cause_action = ground_truth.get('root_cause_action')
    essential_clues = ground_truth.get('essential_clues', [])
    
    # Validate that the level has all required information for solution
    has_solution = all([
        root_cause_time is not None,
        root_cause_perpetrator is not None and root_cause_perpetrator != "",
        root_cause_action is not None and root_cause_action != ""
    ])
    
    # Check if there are essential clues to support the investigation
    has_essential_clues = len([c for c in essential_clues if c.get('relevance') == 'essential']) > 0
    
    # Calculate theoretical maximum
    if has_solution and has_essential_clues:
        max_reward = 1.0
        calculation_method = "binary_success_reward"
        notes = f"Correct identification yields 1.0 reward. Root cause: {root_cause_perpetrator} performed {root_cause_action} at time {root_cause_time}"
    else:
        max_reward = 0.0
        calculation_method = "unsolvable_level"
        notes = "Level appears to be missing required solution components"
    
    return {
        "max_reward": max_reward,
        "calculation_method": calculation_method,
        "notes": notes,
        "ground_truth_summary": {
            "root_cause_time": root_cause_time,
            "root_cause_perpetrator": root_cause_perpetrator,
            "root_cause_action": root_cause_action,
            "essential_clues_count": len([c for c in essential_clues if c.get('relevance') == 'essential']),
            "decoy_clues_count": len([c for c in essential_clues if c.get('relevance') == 'decoy'])
        }
    }

def main():
    """Main function to calculate max rewards for all levels."""
    
    # Get list of level files
    levels_dir = "./levels/"
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(level_files)} level files")
    
    # Calculate max rewards for each level
    results = {
        "environment_id": "20250904_170900_env_backwards_time_i",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_structure_analysis": {
            "reward_type": "binary",
            "max_possible_per_level": 1.0,
            "success_condition": "Correct identification of root cause (time, perpetrator, action)",
            "failure_conditions": ["Incorrect identification", "Timeout (time index <= 0)", "Max steps exceeded"]
        },
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    for level_file in level_files:
        print(f"Analyzing {level_file}...")
        try:
            level_data = load_level(level_file)
            level_analysis = analyze_level_max_reward(level_data, level_file)
            results["levels"][level_file] = level_analysis
            max_rewards.append(level_analysis["max_reward"])
            print(f"  Max reward: {level_analysis['max_reward']}")
        except Exception as e:
            print(f"  Error analyzing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": 0.0,
                "calculation_method": "error",
                "notes": f"Error during analysis: {str(e)}"
            }
            max_rewards.append(0.0)
    
    # Calculate summary statistics
    if max_rewards:
        results["summary"] = {
            "total_levels": len(max_rewards),
            "average_max_reward": sum(max_rewards) / len(max_rewards),
            "min_max_reward": min(max_rewards),
            "max_max_reward": max(max_rewards),
            "solvable_levels": len([r for r in max_rewards if r > 0])
        }
    
    # Save results to JSON file
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary: {len(max_rewards)} levels analyzed")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"Solvable levels: {results['summary']['solvable_levels']}/{len(max_rewards)}")

if __name__ == "__main__":
    main()