#!/usr/bin/env python3

import yaml
import json
from datetime import datetime
import os
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward(level_data: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    """Calculate maximum theoretical reward for a level."""
    
    # Extract key parameters
    target_population = level_data['globals']['target_population']
    initial_population = level_data['colony']['population']
    max_steps = level_data['globals']['max_steps']
    
    # Available resources for discovery
    resources = level_data['resources']
    resource_mappings = level_data['_hidden']['resource_mappings']
    
    # Available buildings for discovery
    buildings = level_data['buildings']
    building_mappings = level_data['_hidden']['building_mappings']
    
    # Explorable resources that can be unlocked
    explorable_resources = level_data['_hidden'].get('explorable_resources', {})
    
    # Calculate maximum rewards
    max_reward = 0.0
    reward_breakdown = {}
    
    # 1. Resource Discovery Rewards (0.1 each)
    # All initial resources can potentially be discovered
    discoverable_resources = list(resource_mappings.keys())
    # Add explorable resources that can be unlocked
    discoverable_resources.extend(list(explorable_resources.keys()))
    
    resource_discovery_reward = len(discoverable_resources) * 0.1
    max_reward += resource_discovery_reward
    reward_breakdown['resource_discoveries'] = {
        'count': len(discoverable_resources),
        'reward_per_discovery': 0.1,
        'total': resource_discovery_reward,
        'resources': discoverable_resources
    }
    
    # 2. Building Discovery Rewards (0.1 each)
    # All buildings can potentially be discovered if population > 15
    building_types = set()
    for building in buildings:
        building_types.add(building['type'])
    # Add building types from mappings that might be buildable
    for building_type in building_mappings.keys():
        building_types.add(building_type)
    
    building_discovery_reward = len(building_types) * 0.1
    max_reward += building_discovery_reward
    reward_breakdown['building_discoveries'] = {
        'count': len(building_types),
        'reward_per_discovery': 0.1,
        'total': building_discovery_reward,
        'buildings': list(building_types)
    }
    
    # 3. Goal Achievement Reward (0.7)
    # Can the population reach the target? Need to analyze if it's feasible
    population_needed = target_population - initial_population
    
    # Calculate potential population boost from resources
    # Based on _handle_allocate_resource in env_main.py:
    # - nutrition_boost: +1 population per unit
    # - health_boost: +2 happiness per unit
    # - happiness_boost: +3 happiness per unit
    # - efficiency_boost: +0.1 building efficiency per unit
    
    potential_population_boost = 0
    total_nutrition_resources = 0
    
    for resource, amount in resources.items():
        true_effect = resource_mappings.get(resource)
        if true_effect == 'nutrition_boost':
            total_nutrition_resources += amount
    
    # Add explorable nutrition resources
    for resource, info in explorable_resources.items():
        true_effect = resource_mappings.get(resource)
        if true_effect == 'nutrition_boost':
            total_nutrition_resources += info['initial_amount']
    
    potential_population_boost = total_nutrition_resources
    
    # Check if goal is achievable
    goal_achievable = (initial_population + potential_population_boost >= target_population)
    
    if goal_achievable:
        goal_reward = 0.7
        max_reward += goal_reward
        reward_breakdown['goal_achievement'] = {
            'reward': goal_reward,
            'achievable': True,
            'target_population': target_population,
            'initial_population': initial_population,
            'potential_boost': potential_population_boost
        }
    else:
        reward_breakdown['goal_achievement'] = {
            'reward': 0.0,
            'achievable': False,
            'target_population': target_population,
            'initial_population': initial_population,
            'potential_boost': potential_population_boost
        }
    
    return {
        'max_reward': round(max_reward, 2),
        'calculation_method': 'optimal_resource_allocation_analysis',
        'notes': f'Assumes perfect resource discovery and optimal allocation within {max_steps} steps',
        'breakdown': reward_breakdown,
        'feasibility': {
            'population_gap': population_needed,
            'available_nutrition_resources': total_nutrition_resources,
            'goal_achievable': goal_achievable
        }
    }

def main():
    """Calculate maximum rewards for all levels."""
    
    env_id = "20250919_185613_env_107_AlienDec"
    levels_dir = "./levels/"
    
    # Get all level files
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": env_id,
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {
            "total_levels": len(level_files),
            "average_max_reward": 0.0,
            "min_max_reward": float('inf'),
            "max_max_reward": 0.0
        }
    }
    
    total_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        
        try:
            level_path = os.path.join(levels_dir, level_file)
            level_data = load_level(level_path)
            
            reward_analysis = calculate_max_reward(level_data, level_file)
            results["levels"][level_file] = reward_analysis
            
            max_reward = reward_analysis['max_reward']
            total_rewards.append(max_reward)
            
            print(f"  Max reward: {max_reward}")
            
        except Exception as e:
            print(f"Error processing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": 0.0,
                "error": str(e),
                "calculation_method": "error",
                "notes": f"Failed to calculate due to: {str(e)}"
            }
    
    # Calculate summary statistics
    if total_rewards:
        results["summary"]["average_max_reward"] = round(sum(total_rewards) / len(total_rewards), 2)
        results["summary"]["min_max_reward"] = min(total_rewards)
        results["summary"]["max_max_reward"] = max(total_rewards)
    
    # Save results
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {results['summary']['total_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']}")
    print(f"  Min max reward: {results['summary']['min_max_reward']}")
    print(f"  Max max reward: {results['summary']['max_max_reward']}")

if __name__ == "__main__":
    main()