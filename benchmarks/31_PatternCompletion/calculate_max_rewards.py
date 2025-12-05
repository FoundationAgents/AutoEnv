import yaml
import os
from datetime import datetime
import json

def load_level(level_file):
    """Load a level YAML file and return the state."""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward(level_file):
    """Calculate the maximum possible reward for a given level."""
    level_state = load_level(level_file)
    
    # Get masked positions from the level
    masked_positions = level_state['canvas']['masked_positions']
    ground_truth = level_state['canvas']['ground_truth']
    
    # Maximum reward is achieved by correctly filling all masked positions
    # Each correct restoration gives 1.0 reward (from config.yaml)
    max_reward = len(masked_positions) * 1.0
    
    # Verify the level structure
    mask_count = level_state['canvas']['mask_count']
    if len(masked_positions) != mask_count:
        print(f"Warning: mask_count mismatch in {level_file}")
    
    return {
        'max_reward': max_reward,
        'masked_positions_count': len(masked_positions),
        'mask_count': mask_count,
        'calculation_method': 'optimal_completion',
        'notes': f'Assumes perfect restoration of all {len(masked_positions)} masked pixels'
    }

def main():
    """Calculate maximum rewards for all levels."""
    levels_dir = './levels'
    results = {
        'environment_id': '20250919_152054_env_106_visual_pattern_completion',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()  # Sort for consistent ordering
    
    max_rewards = []
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        try:
            level_result = calculate_max_reward(level_path)
            results['levels'][level_file] = level_result
            max_rewards.append(level_result['max_reward'])
            print(f"✓ {level_file}: Max reward = {level_result['max_reward']}")
        except Exception as e:
            print(f"✗ Error processing {level_file}: {e}")
            results['levels'][level_file] = {
                'max_reward': 0.0,
                'error': str(e),
                'calculation_method': 'failed',
                'notes': 'Failed to calculate due to error'
            }
    
    # Calculate summary statistics
    if max_rewards:
        results['summary'] = {
            'total_levels': len(max_rewards),
            'average_max_reward': sum(max_rewards) / len(max_rewards),
            'min_max_reward': min(max_rewards),
            'max_max_reward': max(max_rewards)
        }
    else:
        results['summary'] = {
            'total_levels': 0,
            'average_max_reward': 0.0,
            'min_max_reward': 0.0,
            'max_max_reward': 0.0
        }
    
    # Save results to JSON file
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Summary:")
    print(f"Total levels processed: {results['summary']['total_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.1f}")
    print(f"Min max reward: {results['summary']['min_max_reward']:.1f}")
    print(f"Max max reward: {results['summary']['max_max_reward']:.1f}")
    print(f"\n✅ Results saved to level_max_rewards.json")

if __name__ == "__main__":
    main()