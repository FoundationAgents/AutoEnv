import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any

def analyze_level(level_file: str) -> Dict[str, Any]:
    """Analyze a level file and calculate maximum possible reward."""
    with open(level_file, 'r') as f:
        level_data = yaml.safe_load(f)
    
    # Get grid icons
    icons = level_data.get('grid', {}).get('icons', {})
    
    # Count bombs (treasures)
    bomb_count = sum(1 for icon in icons.values() if icon == 'bomb')
    
    # Count flowers (dangers) 
    flower_count = sum(1 for icon in icons.values() if icon == 'flower')
    
    # Find bomb positions
    bomb_positions = [pos for pos, icon in icons.items() if icon == 'bomb']
    
    # Maximum reward is 1.0 per bomb found
    # In this environment, each level has exactly 1 bomb
    max_reward = float(bomb_count * 1.0)
    
    analysis = {
        'max_reward': max_reward,
        'calculation_method': 'optimal_treasure_collection',
        'notes': f'Level has {bomb_count} bomb(s) at {bomb_positions}, {flower_count} flowers. Max reward assumes finding all bombs while avoiding flowers.',
        'bomb_count': bomb_count,
        'flower_count': flower_count,
        'bomb_positions': bomb_positions
    }
    
    return analysis

def main():
    """Calculate maximum rewards for all levels."""
    levels_dir = './levels/'
    
    if not os.path.exists(levels_dir):
        print(f"Error: Levels directory '{levels_dir}' not found")
        return
    
    # Get all YAML level files
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    if not level_files:
        print(f"Error: No YAML level files found in '{levels_dir}'")
        return
    
    print(f"Found {len(level_files)} level files")
    
    # Analyze each level
    results = {}
    max_rewards = []
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        print(f"Analyzing {level_file}...")
        
        try:
            analysis = analyze_level(level_path)
            results[level_file] = analysis
            max_rewards.append(analysis['max_reward'])
            print(f"  Max reward: {analysis['max_reward']}")
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
        'average_max_reward': sum(max_rewards) / len(max_rewards) if max_rewards else 0.0,
        'min_max_reward': min(max_rewards) if max_rewards else 0.0,
        'max_max_reward': max(max_rewards) if max_rewards else 0.0
    }
    
    # Create final JSON output
    output = {
        'environment_id': '20250919_121700_env_100_hidden_danger_detection',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': results,
        'summary': summary
    }
    
    # Save to JSON file
    output_file = 'level_max_rewards.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total levels: {summary['total_levels']}")
    print(f"  Average max reward: {summary['average_max_reward']:.2f}")
    print(f"  Min max reward: {summary['min_max_reward']:.2f}")
    print(f"  Max max reward: {summary['max_max_reward']:.2f}")

if __name__ == '__main__':
    main()