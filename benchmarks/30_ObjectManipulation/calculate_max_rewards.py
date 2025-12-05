import yaml
import os
import json
from datetime import datetime

def analyze_level_max_reward(level_file):
    """
    Calculate the theoretical maximum reward for a given level.
    
    Reward structure:
    - pickup_target_object: +0.3 (when picking up object needed for chore)
    - complete_chore: +0.7 (for each of 3 chores)
    - complete_final_chore: +1.0 (bonus when all chores completed)
    
    Maximum possible: 3*0.7 + 1.0 + pickup_bonuses = 3.1 + pickup_bonuses
    """
    
    with open(level_file, 'r') as f:
        level_data = yaml.safe_load(f)
    
    # Base rewards for completing all 3 chores
    chore_completion_reward = 3 * 0.7  # 2.1
    final_chore_bonus = 1.0
    
    # Calculate pickup bonuses based on chore requirements
    instructions = level_data['chores']['instructions']
    objects = level_data['objects']
    
    pickup_bonuses = 0.0
    pickup_details = []
    
    for instruction in instructions:
        instruction_lower = instruction.lower()
        
        # Check if this chore requires picking up an object
        if 'move' in instruction_lower or 'put' in instruction_lower:
            # Find if there's a matching object that can be picked up
            for obj in objects:
                if obj['color'] in instruction_lower and obj['type'] in instruction_lower:
                    pickup_bonuses += 0.3
                    pickup_details.append(f"Pickup {obj['color']} {obj['type']} for: {instruction}")
                    break
    
    total_max_reward = chore_completion_reward + final_chore_bonus + pickup_bonuses
    
    return {
        'max_reward': total_max_reward,
        'breakdown': {
            'chore_completions': chore_completion_reward,
            'final_bonus': final_chore_bonus,
            'pickup_bonuses': pickup_bonuses,
            'pickup_count': len(pickup_details)
        },
        'pickup_details': pickup_details,
        'chores': instructions,
        'calculation_method': 'analytical_reward_structure',
        'notes': 'Assumes optimal execution: all chores completed with all applicable pickup bonuses'
    }

def main():
    levels_dir = './levels'
    results = {
        'environment_id': '20250918_222610_env_85_household_object_manipulation',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    max_rewards = []
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        level_result = analyze_level_max_reward(level_path)
        results['levels'][level_file] = level_result
        max_rewards.append(level_result['max_reward'])
        
        print(f"Processed {level_file}: Max reward = {level_result['max_reward']}")
    
    # Calculate summary statistics
    results['summary'] = {
        'total_levels': len(level_files),
        'average_max_reward': sum(max_rewards) / len(max_rewards),
        'min_max_reward': min(max_rewards),
        'max_max_reward': max(max_rewards)
    }
    
    # Save to JSON file
    output_file = 'level_max_rewards.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {results['summary']['total_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"  Min max reward: {results['summary']['min_max_reward']}")
    print(f"  Max max reward: {results['summary']['max_max_reward']}")

if __name__ == '__main__':
    main()