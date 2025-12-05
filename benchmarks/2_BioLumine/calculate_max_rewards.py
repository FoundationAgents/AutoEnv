import yaml
import json
from datetime import datetime
import os
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward_for_level(level_data: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    """
    Calculate the maximum theoretical reward for a given level.
    
    In this environment:
    - Each successful handshake gives 1.0 reward
    - Failed handshake gives 0.0 reward and ends the session
    - Maximum 3 handshakes needed to complete the game
    - Maximum theoretical reward is 3.0 (3 successful handshakes)
    
    Since the protocols have deterministic rules, an optimal agent can 
    always achieve perfect communication if it knows the protocol.
    """
    
    # The maximum reward is always 3.0 in this environment
    # This assumes optimal play where the agent correctly decodes
    # the protocol and responds perfectly to all 3 handshake attempts
    max_reward = 3.0
    
    # Get some information about the level for documentation
    protocol = level_data['session']['active_protocol']
    protocol_params = level_data['session']['protocol_params']
    initial_pattern_length = len(level_data['session']['current_incoming_pattern'])
    max_steps = level_data['globals']['max_steps']
    
    return {
        "max_reward": max_reward,
        "calculation_method": "optimal_protocol_decoding",
        "notes": f"Assumes perfect protocol decoding for {protocol} with params {protocol_params}. Initial pattern length: {initial_pattern_length}. Max steps: {max_steps}",
        "protocol": protocol,
        "protocol_params": protocol_params,
        "initial_pattern_length": initial_pattern_length,
        "max_steps": max_steps
    }

def main():
    """Calculate maximum rewards for all levels and generate JSON output."""
    
    levels_dir = "./levels"
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    results = {
        "environment_id": "20250904_170900_env_bioluminescent_s",
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {}
    }
    
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} level files...")
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        print(f"Processing {level_file}...")
        
        try:
            level_data = load_level(level_path)
            level_analysis = calculate_max_reward_for_level(level_data, level_file)
            
            results["levels"][level_file] = level_analysis
            max_rewards.append(level_analysis["max_reward"])
            
        except Exception as e:
            print(f"Error processing {level_file}: {e}")
            results["levels"][level_file] = {
                "max_reward": None,
                "calculation_method": "error",
                "notes": f"Error during analysis: {str(e)}"
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
            "theoretical_explanation": "All levels have the same maximum reward of 3.0 as the environment awards 1.0 for each successful handshake, with exactly 3 handshakes required to complete a session."
        }
    else:
        results["summary"] = {
            "total_levels": len(level_files),
            "successfully_analyzed": 0,
            "error": "No levels could be analyzed successfully"
        }
    
    # Save results to JSON file
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Successfully analyzed {len(valid_rewards)} out of {len(level_files)} levels")
    if valid_rewards:
        print(f"Maximum theoretical reward for all levels: {max(valid_rewards)}")

if __name__ == "__main__":
    main()