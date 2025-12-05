import yaml
import json
from datetime import datetime
import os
from typing import Dict, Any, List, Tuple

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file"""
    with open(f"levels/{level_file}", 'r') as f:
        return yaml.safe_load(f)

def analyze_reward_structure():
    """Analyze the reward structure from config"""
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    reward_config = config.get('reward', {})
    events = reward_config.get('events', [])
    completion_bonus = reward_config.get('completion_bonus', 0.0)
    
    return events, completion_bonus

def calculate_max_reward_for_level(level_file: str) -> Dict[str, Any]:
    """Calculate maximum theoretical reward for a single level"""
    
    # Load level data
    level_data = load_level(level_file)
    
    # Get reward structure
    events, completion_bonus = analyze_reward_structure()
    
    # Find target object
    target_obj = None
    for obj in level_data['objects']:
        if obj.get('is_target', False):
            target_obj = obj
            break
    
    if not target_obj:
        return {
            'max_reward': 0.0,
            'calculation_method': 'no_target_found',
            'notes': 'No target object found in level'
        }
    
    # Calculate maximum possible reward
    max_reward = 0.0
    notes = []
    
    # Target in goal reward
    target_in_goal_reward = 0.0
    for event in events:
        if event.get('trigger') == 'target_in_goal':
            target_in_goal_reward = event.get('value', 0.0)
            break
    
    # The maximum reward is achieved when target reaches goal
    # This gives both the event reward and terminates successfully
    max_reward = target_in_goal_reward
    notes.append(f"Target in goal reward: {target_in_goal_reward}")
    
    # Check if level is theoretically solvable
    target_pos = target_obj['position']
    goal_area = level_data['globals']['goal_area']
    max_steps = level_data['globals']['max_steps']
    
    # Calculate minimum Manhattan distance to goal
    min_dist_to_goal = float('inf')
    for x in range(goal_area[0][0], goal_area[1][0] + 1):
        for y in range(goal_area[0][1], goal_area[1][1] + 1):
            dist = abs(target_pos[0] - x) + abs(target_pos[1] - y)
            min_dist_to_goal = min(min_dist_to_goal, dist)
    
    solvable = min_dist_to_goal <= max_steps
    if not solvable:
        notes.append(f"Level might not be solvable: min distance to goal {min_dist_to_goal} > max steps {max_steps}")
    
    return {
        'max_reward': max_reward,
        'calculation_method': 'optimal_completion',
        'notes': '; '.join(notes),
        'target_position': target_pos,
        'goal_area': goal_area,
        'min_distance_to_goal': min_dist_to_goal,
        'max_steps': max_steps,
        'theoretically_solvable': solvable
    }

def main():
    """Main function to calculate max rewards for all levels"""
    
    # Get list of all level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    print(f"Found {len(level_files)} level files")
    
    # Calculate max rewards for each level
    results = {}
    max_rewards = []
    
    for level_file in level_files:
        print(f"Analyzing {level_file}...")
        try:
            result = calculate_max_reward_for_level(level_file)
            results[level_file] = result
            max_rewards.append(result['max_reward'])
            print(f"  Max reward: {result['max_reward']}")
        except Exception as e:
            print(f"  Error analyzing {level_file}: {e}")
            results[level_file] = {
                'max_reward': 0.0,
                'calculation_method': 'error',
                'notes': f'Error during analysis: {str(e)}'
            }
            max_rewards.append(0.0)
    
    # Create summary statistics
    summary = {
        'total_levels': len(level_files),
        'average_max_reward': sum(max_rewards) / len(max_rewards) if max_rewards else 0,
        'min_max_reward': min(max_rewards) if max_rewards else 0,
        'max_max_reward': max(max_rewards) if max_rewards else 0
    }
    
    # Create final JSON output
    output = {
        'environment_id': '20250919_185300_env_16_shadow_puppet_re',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'reward_structure': {
            'target_in_goal_reward': 1.0,
            'completion_bonus': 1.0,
            'max_steps': 40
        },
        'levels': results,
        'summary': summary
    }
    
    # Write to JSON file
    with open('level_max_rewards.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults written to level_max_rewards.json")
    print(f"Summary:")
    print(f"  Total levels: {summary['total_levels']}")
    print(f"  Average max reward: {summary['average_max_reward']:.2f}")
    print(f"  Min max reward: {summary['min_max_reward']:.2f}")
    print(f"  Max max reward: {summary['max_max_reward']:.2f}")

if __name__ == "__main__":
    main()