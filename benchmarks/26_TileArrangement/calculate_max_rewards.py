#!/usr/bin/env python3
import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any

def calculate_max_reward_for_level(level_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate maximum possible reward for a given level.
    
    Reward sources:
    1. Pair discovery: 1.0 per pair found
    2. Symbol exploration: 0.05 per unique symbol seen for first time
    """
    
    # Extract level information
    total_pairs = level_data["globals"]["total_pairs"]
    symbol_pairs = level_data["game"]["symbol_pairs"]
    cards = level_data["board"]["cards"]
    
    # Calculate unique symbols on the board
    unique_symbols = set()
    for row in cards:
        for symbol in row:
            unique_symbols.add(symbol)
    
    # Maximum reward calculation
    pair_reward = total_pairs * 1.0  # 1.0 reward per pair found
    exploration_reward = len(unique_symbols) * 0.05  # 0.05 per unique symbol
    
    max_reward = pair_reward + exploration_reward
    
    return {
        "max_reward": max_reward,
        "breakdown": {
            "pair_rewards": pair_reward,
            "exploration_rewards": exploration_reward,
            "total_pairs": total_pairs,
            "unique_symbols": len(unique_symbols),
            "symbol_list": sorted(list(unique_symbols))
        },
        "calculation_method": "optimal_path_analysis",
        "notes": "Assumes perfect execution: all pairs found + all symbols explored"
    }

def main():
    # Environment info
    env_id = "20250919_122826_env_102_pattern_memory_matching"
    levels_dir = "./levels"
    
    # Get all level files
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    results = {
        "environment_id": env_id,
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    print(f"Calculating maximum rewards for {len(level_files)} levels...")
    
    # Process each level
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        
        try:
            # Load level data
            with open(level_path, 'r') as f:
                level_data = yaml.safe_load(f)
            
            # Calculate max reward
            level_result = calculate_max_reward_for_level(level_data)
            max_reward = level_result["max_reward"]
            max_rewards.append(max_reward)
            
            # Store result
            results["levels"][level_file] = level_result
            
            print(f"✓ {level_file}: Max reward = {max_reward}")
            
        except Exception as e:
            print(f"✗ Error processing {level_file}: {str(e)}")
            results["levels"][level_file] = {
                "max_reward": None,
                "error": str(e),
                "calculation_method": "failed",
                "notes": "Could not calculate due to error"
            }
    
    # Calculate summary statistics
    if max_rewards:
        results["summary"] = {
            "total_levels": len(level_files),
            "successful_calculations": len(max_rewards),
            "average_max_reward": round(sum(max_rewards) / len(max_rewards), 2),
            "min_max_reward": min(max_rewards),
            "max_max_reward": max(max_rewards),
            "total_max_reward_all_levels": sum(max_rewards)
        }
    else:
        results["summary"] = {
            "total_levels": len(level_files),
            "successful_calculations": 0,
            "error": "No levels could be processed successfully"
        }
    
    # Save results to JSON
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary: {len(max_rewards)}/{len(level_files)} levels processed successfully")
    if max_rewards:
        print(f"Average max reward: {results['summary']['average_max_reward']}")
        print(f"Range: {results['summary']['min_max_reward']} - {results['summary']['max_max_reward']}")

if __name__ == "__main__":
    main()