#!/usr/bin/env python3

import yaml
import json
from datetime import datetime
import os
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file"""
    with open(f"levels/{level_file}", 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def analyze_level_max_reward(level_data: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    """
    Analyze a level and calculate its theoretical maximum reward.
    
    Based on the environment code analysis:
    - Reward is 1.0 if ALL storage tiles are covered AND agent reaches exit
    - Reward is 0.0 otherwise
    - This is a binary reward system
    
    Maximum theoretical reward = 1.0 (successful completion)
    """
    
    # Extract level information
    num_crates = len(level_data['objects']['crates'])
    num_storage_tiles = len(level_data['objects']['storage_tiles'])
    agent_start = level_data['agent']['pos']
    exit_pos = level_data['objects']['exit_pos']
    max_steps = level_data['globals']['max_steps']
    grid_size = level_data['grid']['size']
    
    # Analysis
    analysis = {
        "max_reward": 1.0,
        "calculation_method": "binary_reward_analysis",
        "notes": "Maximum reward is 1.0 for successful puzzle completion (all storage tiles covered + agent at exit)",
        "level_stats": {
            "crates": num_crates,
            "storage_tiles": num_storage_tiles,
            "balanced_puzzle": num_crates == num_storage_tiles,
            "max_steps": max_steps,
            "grid_size": grid_size,
            "agent_start": agent_start,
            "exit_position": exit_pos
        },
        "solvability": "assumed_solvable" if num_crates == num_storage_tiles else "potential_issue"
    }
    
    # Check for potential issues
    if num_crates != num_storage_tiles:
        analysis["notes"] += f" WARNING: Unbalanced puzzle ({num_crates} crates vs {num_storage_tiles} storage tiles)"
        analysis["max_reward"] = 0.0  # Impossible to solve if unbalanced
    
    return analysis

def main():
    """Main function to analyze all levels and generate JSON report"""
    
    # Get list of level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    print(f"Found {len(level_files)} level files")
    
    results = {
        "environment_id": "20250918_171219_env_82_warehouse_logistics_puzzle",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_system_analysis": {
            "type": "binary_completion_reward",
            "success_reward": 1.0,
            "failure_reward": 0.0,
            "description": "Agent gets 1.0 reward only when ALL storage tiles are covered by crates AND agent reaches exit position"
        },
        "levels": {},
        "summary": {
            "total_levels": len(level_files),
            "solvable_levels": 0,
            "average_max_reward": 0.0,
            "min_max_reward": 0.0,
            "max_max_reward": 0.0
        }
    }
    
    max_rewards = []
    solvable_count = 0
    
    # Analyze each level
    for level_file in level_files:
        print(f"Analyzing {level_file}...")
        
        try:
            level_data = load_level(level_file)
            analysis = analyze_level_max_reward(level_data, level_file)
            
            results["levels"][level_file] = analysis
            max_rewards.append(analysis["max_reward"])
            
            if analysis["max_reward"] > 0:
                solvable_count += 1
            
        except Exception as e:
            print(f"Error analyzing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": 0.0,
                "calculation_method": "error",
                "notes": f"Analysis failed: {str(e)}"
            }
            max_rewards.append(0.0)
    
    # Calculate summary statistics
    if max_rewards:
        results["summary"]["solvable_levels"] = solvable_count
        results["summary"]["average_max_reward"] = sum(max_rewards) / len(max_rewards)
        results["summary"]["min_max_reward"] = min(max_rewards)
        results["summary"]["max_max_reward"] = max(max_rewards)
    
    # Save results
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total levels: {len(level_files)}")
    print(f"Solvable levels: {solvable_count}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")

if __name__ == "__main__":
    main()