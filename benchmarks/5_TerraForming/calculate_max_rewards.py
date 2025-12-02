#!/usr/bin/env python3

import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import copy

class RewardCalculator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract reward constants from config
        self.rewards = self.config['reward']
        
    def calculate_max_reward(self, level_file: str) -> Dict[str, Any]:
        """Calculate theoretical maximum reward for a level"""
        with open(f"levels/{level_file}", 'r') as f:
            initial_state = yaml.safe_load(f)
        
        max_steps = initial_state.get('globals', {}).get('max_steps', 40)
        
        # Calculate theoretical maximum by considering all possible positive rewards
        # and minimizing penalties
        
        # 1. Mission completion reward (guaranteed if achievable)
        completion_reward = 20.0
        
        # 2. Maximum stability rewards (0.1 per step for max_steps)
        # Assume we can maintain instability < 50% for most steps
        max_stability_rewards = 0.1 * (max_steps - 5)  # Conservative estimate
        
        # 3. Maximum atmospheric progress rewards
        # From 0% to 25% O2 (optimal range 15-25%), so 10 points in optimal range
        max_atmospheric_rewards = 10.0 * 0.05  # 0.5
        
        # 4. Maximum hydrological progress rewards  
        # From current water% to 70% (optimal range 30-70%), so up to 40 points in optimal range
        initial_water = initial_state['hydrosphere']['surface_water_pct']
        water_potential = max(0, 70.0 - max(30.0, initial_water))
        max_hydro_rewards = water_potential * 0.1
        
        # 5. Maximum habitability increase rewards
        # From 0% to 100% habitability = 100 points * 0.2 = 20.0
        max_habitability_rewards = 100.0 * 0.2
        
        # 6. Calculate potential penalties (minimize these)
        # Assume optimal play avoids catastrophic failure (-40.0) and minimizes instability penalties
        min_penalties = 0.0  # Optimal play should avoid penalties
        
        # Total theoretical maximum
        theoretical_max = (completion_reward + max_stability_rewards + 
                          max_atmospheric_rewards + max_hydro_rewards + 
                          max_habitability_rewards + min_penalties)
        
        # However, many of these rewards overlap or are mutually exclusive
        # Let's calculate a more realistic maximum:
        
        # Realistic scenario: Complete mission successfully with good efficiency
        realistic_max = (
            completion_reward +  # 20.0 for mission success
            max_stability_rewards * 0.7 +  # 70% of max stability rewards
            max_atmospheric_rewards +  # 0.5 for atmospheric progress  
            max_hydro_rewards * 0.8 +  # 80% of max hydro rewards
            10.0  # Reasonable habitability progress rewards (50 points * 0.2)
        )
        
        return {
            "max_reward": round(realistic_max, 2),
            "theoretical_ceiling": round(theoretical_max, 2),
            "calculation_method": "analytical_optimal_path",
            "components": {
                "mission_completion": completion_reward,
                "stability_rewards_est": round(max_stability_rewards * 0.7, 2),
                "atmospheric_progress_est": max_atmospheric_rewards,
                "hydrological_progress_est": round(max_hydro_rewards * 0.8, 2),
                "habitability_progress_est": 10.0
            },
            "notes": f"Based on {max_steps} max steps, assumes optimal play avoiding penalties",
            "initial_conditions": {
                "energy_reserves": initial_state['infrastructure']['energy_reserves'],
                "initial_water_pct": initial_state['hydrosphere']['surface_water_pct'],
                "initial_instability": initial_state['global_metrics']['instability_index'],
                "max_steps": max_steps
            }
        }

def main():
    calculator = RewardCalculator()
    
    # Get all level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": "20250905_101109_env_terraforming_pro",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        try:
            level_data = calculator.calculate_max_reward(level_file)
            results["levels"][level_file] = level_data
            max_rewards.append(level_data["max_reward"])
            print(f"  Max reward: {level_data['max_reward']}")
        except Exception as e:
            print(f"  Error processing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": None,
                "error": str(e),
                "calculation_method": "failed"
            }
    
    # Calculate summary statistics
    valid_rewards = [r for r in max_rewards if r is not None]
    if valid_rewards:
        results["summary"] = {
            "total_levels": len(level_files),
            "successful_calculations": len(valid_rewards),
            "average_max_reward": round(sum(valid_rewards) / len(valid_rewards), 2),
            "min_max_reward": round(min(valid_rewards), 2),
            "max_max_reward": round(max(valid_rewards), 2)
        }
    
    # Save results
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to level_max_rewards.json")
    print(f"Summary:")
    if valid_rewards:
        print(f"  Total levels: {len(level_files)}")
        print(f"  Average max reward: {results['summary']['average_max_reward']}")
        print(f"  Range: {results['summary']['min_max_reward']} - {results['summary']['max_max_reward']}")

if __name__ == "__main__":
    main()