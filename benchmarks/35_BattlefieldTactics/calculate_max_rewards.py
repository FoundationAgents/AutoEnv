#!/usr/bin/env python3

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def calculate_max_reward_for_level(level_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the theoretical maximum reward for a given level.
    
    Based on the analysis:
    - Each enemy camp eliminated = 0.5 reward
    - Total enemy camps per level = 2
    - Maximum possible reward = 2 × 0.5 = 1.0
    """
    
    total_camps = level_data["globals"]["total_enemy_camps"]
    reward_per_camp = 0.5  # From config.yaml and env_main.py analysis
    
    # Calculate total squad strength
    total_squad_strength = sum(squad["strength"] for squad in level_data["squads"] if squad["active"])
    
    # Get enemy camp strengths
    enemy_camps = level_data["enemy_camps"]
    enemy_strengths = [camp["strength"] for camp in enemy_camps]
    
    # Check if it's theoretically possible to eliminate all camps
    # For each camp, we need combined adjacent squad strength > camp strength
    max_possible_reward = total_camps * reward_per_camp
    
    # Analyze feasibility
    can_eliminate_all = True
    elimination_notes = []
    
    for camp in enemy_camps:
        if total_squad_strength <= camp["strength"]:
            # If total squad strength is not greater than camp strength,
            # it's impossible to eliminate this camp
            can_eliminate_all = False
            elimination_notes.append(f"Camp {camp['id']} (strength {camp['strength']}) cannot be eliminated with total squad strength {total_squad_strength}")
    
    if can_eliminate_all:
        theoretical_max = max_possible_reward
        notes = f"All {total_camps} camps can be theoretically eliminated. Total squad strength: {total_squad_strength}, Enemy strengths: {enemy_strengths}"
    else:
        # Calculate how many camps can potentially be eliminated
        eliminable_camps = 0
        for camp in enemy_camps:
            if total_squad_strength > camp["strength"]:
                eliminable_camps += 1
        
        theoretical_max = eliminable_camps * reward_per_camp
        notes = f"Only {eliminable_camps}/{total_camps} camps can be eliminated. " + "; ".join(elimination_notes)
    
    return {
        "max_reward": theoretical_max,
        "calculation_method": "analytical_optimal_strategy",
        "notes": notes,
        "level_details": {
            "total_enemy_camps": total_camps,
            "enemy_camp_strengths": enemy_strengths,
            "total_squad_strength": total_squad_strength,
            "squad_strengths": [squad["strength"] for squad in level_data["squads"]],
            "reward_per_camp": reward_per_camp
        }
    }

def main():
    levels_dir = "./levels/"
    results = {
        "environment_id": "20250919_144322_env_105_battlefield_tactics",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_structure": {
            "reward_per_camp_eliminated": 0.5,
            "total_camps_per_level": 2,
            "theoretical_maximum_per_level": 1.0,
            "no_time_bonuses": True,
            "no_penalties": True
        },
        "levels": {},
        "summary": {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort to ensure consistent ordering
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        level_path = os.path.join(levels_dir, level_file)
        
        try:
            level_data = load_level(level_path)
            reward_analysis = calculate_max_reward_for_level(level_data)
            
            results["levels"][level_file] = reward_analysis
            max_rewards.append(reward_analysis["max_reward"])
            
            print(f"  Max reward: {reward_analysis['max_reward']}")
            
        except Exception as e:
            print(f"Error processing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": None,
                "calculation_method": "error",
                "notes": f"Error during analysis: {str(e)}"
            }
    
    # Calculate summary statistics
    if max_rewards:
        results["summary"] = {
            "total_levels": len(max_rewards),
            "average_max_reward": sum(max_rewards) / len(max_rewards),
            "min_max_reward": min(max_rewards),
            "max_max_reward": max(max_rewards),
            "reward_distribution": {
                str(reward): max_rewards.count(reward) for reward in set(max_rewards)
            }
        }
    
    # Save results to JSON file
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels analyzed: {results['summary']['total_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"  Min max reward: {results['summary']['min_max_reward']}")
    print(f"  Max max reward: {results['summary']['max_max_reward']}")

if __name__ == "__main__":
    main()