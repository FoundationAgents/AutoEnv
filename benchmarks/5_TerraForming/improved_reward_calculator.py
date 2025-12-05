#!/usr/bin/env python3

import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import copy

class ImprovedRewardCalculator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract reward constants from config
        self.rewards = self.config['reward']
        
    def simulate_optimal_strategy(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an optimal strategy to calculate realistic maximum reward"""
        
        max_steps = initial_state.get('globals', {}).get('max_steps', 40)
        
        # Key constraints and parameters
        initial_energy = initial_state['infrastructure']['energy_reserves']
        initial_water = initial_state['hydrosphere']['surface_water_pct']
        initial_ice = initial_state['hydrosphere']['subsurface_ice_pct']
        initial_microbes = initial_state['biosphere_seeds']['dormant_microbes']
        initial_flora = initial_state['biosphere_seeds']['dormant_flora']
        
        # Estimate optimal action sequence efficiency
        # Consider energy constraints and action costs
        
        # Energy costs per action (from env_main.py analysis):
        # DEPLOY_ATMOSPHERIC_PROCESSOR: 100 energy
        # RELEASE_WATER_CATALYSTS: 80 energy  
        # SEED_MICROBIAL_LIFE: 60 energy
        # STABILIZE_TECTONICS: 50-120 energy (variable)
        # CONSTRUCT_UPGRADE_STATION: 150-200 energy
        # DIVERT_ENERGY_TO_SHIELDS: 50-200 energy
        
        # Calculate realistic progression potential
        total_energy_budget = initial_energy
        
        # Estimate number of major actions possible
        avg_action_cost = 85  # Average energy cost per meaningful action
        possible_actions = total_energy_budget // avg_action_cost
        
        # Mission completion reward (achievable if energy budget allows)
        completion_reward = 20.0 if possible_actions >= 15 else 0.0  # Estimate 15 actions needed
        
        # Stability rewards - depends on managing instability
        # More energy = better stability management
        stability_efficiency = min(1.0, total_energy_budget / 1500.0)  # Normalize by reasonable budget
        max_stability_steps = int((max_steps - 5) * stability_efficiency)
        stability_rewards = max_stability_steps * 0.1
        
        # Atmospheric progress rewards
        # Need multiple atmospheric processor deployments
        atmospheric_actions = min(8, possible_actions // 3)  # Estimate 1/3 actions for atmosphere
        atmospheric_progress = atmospheric_actions * 3.0  # ~3% O2 per action
        atmospheric_rewards = min(10.0, atmospheric_progress) * 0.05  # Cap at optimal range
        
        # Hydrological progress rewards
        # Water conversion potential based on ice reserves and energy
        water_actions = min(6, possible_actions // 4)  # Estimate 1/4 actions for water
        potential_water_gain = min(initial_ice * 0.8, water_actions * 12.0)
        
        # Only count rewards for water in optimal range (30-70%)
        target_water = initial_water + potential_water_gain
        if target_water > 30.0:
            rewarded_water = min(70.0, target_water) - max(30.0, initial_water)
            rewarded_water = max(0.0, rewarded_water)
        else:
            rewarded_water = 0.0
            
        hydro_rewards = rewarded_water * 0.1
        
        # Habitability increase rewards
        # This is the big one - depends on successfully improving all systems
        # Estimate based on realistic habitat improvement potential
        estimated_habitat_gain = 0.0
        
        if completion_reward > 0:  # If mission completable
            estimated_habitat_gain = 100.0  # Full habitability achieved
        else:
            # Partial progress based on available actions
            estimated_habitat_gain = min(80.0, possible_actions * 4.0)
            
        habitability_rewards = estimated_habitat_gain * 0.2
        
        # Calculate penalties (minimize in optimal play)
        # Assume good management avoids most penalties
        estimated_penalties = -2.0 if possible_actions < 10 else 0.0  # Energy shortage penalties
        
        total_reward = (completion_reward + stability_rewards + atmospheric_rewards + 
                       hydro_rewards + habitability_rewards + estimated_penalties)
        
        return {
            "max_reward": round(total_reward, 2),
            "calculation_method": "simulation_based_optimal",
            "energy_analysis": {
                "initial_energy": initial_energy,
                "possible_major_actions": possible_actions,
                "estimated_completable": completion_reward > 0
            },
            "reward_components": {
                "mission_completion": completion_reward,
                "stability_rewards": round(stability_rewards, 2),
                "atmospheric_progress": round(atmospheric_rewards, 2),
                "hydrological_progress": round(hydro_rewards, 2),
                "habitability_increase": round(habitability_rewards, 2),
                "penalties": estimated_penalties
            },
            "progression_analysis": {
                "estimated_habitat_gain_pct": estimated_habitat_gain,
                "atmospheric_actions_est": atmospheric_actions,
                "water_actions_est": water_actions,
                "potential_water_gain": round(potential_water_gain, 1),
                "rewarded_water_range": round(rewarded_water, 1)
            },
            "notes": f"Based on {max_steps} steps, {possible_actions} major actions possible with {initial_energy} energy"
        }

def main():
    calculator = ImprovedRewardCalculator()
    
    # Get all level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": "20250905_101109_env_terraforming_pro",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "calculation_method": "improved_simulation_based",
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels with improved method...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        try:
            with open(f"levels/{level_file}", 'r') as f:
                initial_state = yaml.safe_load(f)
            
            level_data = calculator.simulate_optimal_strategy(initial_state)
            level_data["initial_conditions"] = {
                "energy_reserves": initial_state['infrastructure']['energy_reserves'],
                "initial_water_pct": initial_state['hydrosphere']['surface_water_pct'], 
                "initial_ice_pct": initial_state['hydrosphere']['subsurface_ice_pct'],
                "initial_instability": initial_state['global_metrics']['instability_index'],
                "max_steps": initial_state.get('globals', {}).get('max_steps', 40)
            }
            
            results["levels"][level_file] = level_data
            max_rewards.append(level_data["max_reward"])
            print(f"  Max reward: {level_data['max_reward']} (actions possible: {level_data['energy_analysis']['possible_major_actions']})")
            
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
            "max_max_reward": round(max(valid_rewards), 2),
            "reward_range": round(max(valid_rewards) - min(valid_rewards), 2)
        }
    
    # Save results
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nImproved results saved to level_max_rewards.json")
    print(f"Summary:")
    if valid_rewards:
        print(f"  Total levels: {len(level_files)}")
        print(f"  Average max reward: {results['summary']['average_max_reward']}")
        print(f"  Range: {results['summary']['min_max_reward']} - {results['summary']['max_max_reward']}")
        print(f"  Variation: {results['summary']['reward_range']}")

if __name__ == "__main__":
    main()