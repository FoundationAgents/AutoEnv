#!/usr/bin/env python3

"""
Recalculate maximum theoretical rewards for underground city environment
based on the updated reward system.
"""

import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any

def load_level_initial_state(level_file: str) -> Dict[str, Any]:
    """Load initial state from a level file"""
    with open(f"./levels/{level_file}", 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward_for_level(initial_state: Dict[str, Any]) -> Dict[str, float]:
    """Calculate maximum theoretical reward for a level with new reward system"""
    
    # Initial values
    init_integrity = initial_state['metrics']['structural_integrity']
    init_air = initial_state['metrics']['breathable_air_index']
    init_districts = initial_state['agent']['districts_built']
    init_power = initial_state['agent']['power_storage']
    init_research = sum(initial_state['research'].values())
    
    max_rewards = {}
    
    # Base survival bonus (40 steps * 0.2)
    max_rewards['base_survival'] = 40 * 0.2
    
    # Structural improvement bonus (exponential scaling)
    # Assume perfect improvement to 100%
    max_integrity_improvement = 100 - init_integrity
    max_rewards['structural_improvement'] = (max_integrity_improvement * 2.0 + 
                                           (max_integrity_improvement ** 1.5) * 0.1)
    
    # Air improvement bonus
    # Assume improvement to 100% with excellence bonus
    max_air_improvement = 100 - init_air
    max_rewards['air_improvement'] = max_air_improvement * 2.0  # Excellence level
    
    # District completion bonus (scaled by complexity)
    # Building 6 districts: 1*(5+2) + 1*(5+4) + 1*(5+6) + 1*(5+8) + 1*(5+10) + 1*(5+12)
    district_rewards = []
    for i in range(1, 7):  # Building districts 1-6
        district_reward = 1 * (5.0 + i * 2.0)
        district_rewards.append(district_reward)
    max_rewards['district_completion'] = sum(district_rewards)
    
    # Research breakthrough bonus (with synergy)
    # Assume all 3 research completed: 8 + 5 + 10 = 23 per breakthrough
    max_rewards['research_breakthrough'] = 3 * 23.0  # Full mastery bonus
    
    # Power milestone bonus
    max_rewards['power_milestone'] = 15.0
    
    # Strategic excellence bonus (assume 5 conditions met)
    max_rewards['strategic_excellence'] = 5 * 3.0
    
    # Mission progress bonus (from 50% to 100% completion)
    max_rewards['mission_progress'] = 0.5 * 20.0  # 50% progress range
    
    # Mission completion bonus
    max_rewards['mission_completion'] = 100.0
    
    return max_rewards

def main():
    """Recalculate maximum rewards for all levels"""
    
    level_files = [f for f in os.listdir('./levels') if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        "environment_id": "20250907_173627_env_underground_city",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_system_version": "enhanced_strategic_v2",
        "levels": {},
        "summary": {}
    }
    
    total_max_rewards = []
    
    for level_file in level_files:
        print(f"Calculating max reward for {level_file}...")
        
        try:
            initial_state = load_level_initial_state(level_file)
            max_rewards = calculate_max_reward_for_level(initial_state)
            
            total_max = sum(max_rewards.values())
            total_max_rewards.append(total_max)
            
            results["levels"][level_file] = {
                "max_reward": total_max,
                "calculation_method": "enhanced_strategic_analysis_v2",
                "initial_state": {
                    "integrity": initial_state['metrics']['structural_integrity'],
                    "air": initial_state['metrics']['breathable_air_index'],
                    "districts": initial_state['agent']['districts_built'],
                    "power": initial_state['agent']['power_storage'],
                    "research": sum(initial_state['research'].values())
                },
                "reward_breakdown": max_rewards,
                "assumptions": [
                    "Perfect strategic play with no penalties",
                    "All possible improvements achieved",
                    "Strategic excellence bonus achieved",
                    "All research unlocked with synergy",
                    "Mission completed successfully",
                    "Exponential improvement scaling utilized"
                ]
            }
            
            print(f"  Max reward: {total_max:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary statistics
    if total_max_rewards:
        results["summary"] = {
            "total_levels": len(level_files),
            "successfully_analyzed": len(total_max_rewards),
            "average_max_reward": sum(total_max_rewards) / len(total_max_rewards),
            "min_max_reward": min(total_max_rewards),
            "max_max_reward": max(total_max_rewards),
            "total_max_reward": sum(total_max_rewards)
        }
    
    # Save updated results
    with open('./level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Recalculated max rewards for {len(total_max_rewards)} levels")
    print(f"Total max reward across all levels: {sum(total_max_rewards):.2f}")
    print(f"Average max reward per level: {sum(total_max_rewards)/len(total_max_rewards):.2f}")

if __name__ == "__main__":
    main()