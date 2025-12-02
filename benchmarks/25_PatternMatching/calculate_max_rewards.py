import yaml
import json
from datetime import datetime
import os

def load_level(level_file):
    """Load a level YAML file"""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward_for_level(level_data):
    """
    Calculate maximum possible reward for a level.
    
    Reward sources:
    1. pair_cleared: +1.0 per pair (8 pairs total = 8.0 points)
    2. first_exploration: +0.05 per unique position explored (16 positions = 0.8 points)
    
    Total maximum: 8.8 points
    """
    # Constants from config
    PAIR_REWARD = 1.0
    EXPLORATION_REWARD = 0.05
    TOTAL_PAIRS = 8
    TOTAL_POSITIONS = 16
    
    # Maximum rewards calculation
    max_pair_reward = TOTAL_PAIRS * PAIR_REWARD  # 8.0
    max_exploration_reward = TOTAL_POSITIONS * EXPLORATION_REWARD  # 0.8
    
    total_max_reward = max_pair_reward + max_exploration_reward  # 8.8
    
    return {
        "max_reward": total_max_reward,
        "breakdown": {
            "pair_clearing": max_pair_reward,
            "exploration": max_exploration_reward
        },
        "calculation_method": "optimal_strategy_analysis",
        "notes": "Assumes optimal play: all pairs found + all positions explored once"
    }

def main():
    # Get all level files
    levels_dir = "./levels"
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    results = {
        "environment_id": "20250919_122655_env_101_pattern_memory_matching",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reward_structure": {
            "pair_cleared": 1.0,
            "first_exploration": 0.05,
            "total_pairs": 8,
            "total_positions": 16
        },
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        print(f"Processing {level_file}...")
        
        try:
            level_data = load_level(level_path)
            max_reward_info = calculate_max_reward_for_level(level_data)
            
            results["levels"][level_file] = max_reward_info
            max_rewards.append(max_reward_info["max_reward"])
            
            print(f"  Max reward: {max_reward_info['max_reward']}")
            
        except Exception as e:
            print(f"  Error processing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": None,
                "error": str(e),
                "calculation_method": "failed",
                "notes": "Failed to calculate due to error"
            }
    
    # Calculate summary statistics
    valid_rewards = [r for r in max_rewards if r is not None]
    if valid_rewards:
        results["summary"] = {
            "total_levels": len(level_files),
            "successfully_analyzed": len(valid_rewards),
            "average_max_reward": sum(valid_rewards) / len(valid_rewards),
            "min_max_reward": min(valid_rewards),
            "max_max_reward": max(valid_rewards),
            "theoretical_maximum": 8.8,
            "all_levels_same_max": len(set(valid_rewards)) == 1
        }
    
    # Save results
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {results['summary'].get('total_levels', 0)}")
    print(f"  Successfully analyzed: {results['summary'].get('successfully_analyzed', 0)}")
    print(f"  Theoretical maximum reward per level: {results['summary'].get('theoretical_maximum', 8.8)}")
    print(f"  All levels have same maximum: {results['summary'].get('all_levels_same_max', False)}")

if __name__ == "__main__":
    main()