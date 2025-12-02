import yaml
import os
from itertools import combinations
from typing import List, Dict, Any
import json
from datetime import datetime

def calculate_mechanical_advantage(gear_chain: List[int]) -> float:
    """Calculate mechanical advantage for a gear chain."""
    if len(gear_chain) == 0:
        return 1.0
        
    ma = 1.0
    for i in range(0, len(gear_chain) - 1, 2):
        if i + 1 < len(gear_chain):
            ma *= gear_chain[i] / gear_chain[i + 1]
    
    return ma

def is_within_tolerance(current_ma: float, target_ma: float, tolerance: float) -> bool:
    """Check if current MA is within tolerance of target MA."""
    error_ratio = abs(current_ma - target_ma) / target_ma
    return error_ratio <= tolerance

def find_optimal_solution(available_gears: List[int], target_ma: float, tolerance: float, max_chain_length: int = 10) -> tuple:
    """Find if there's a solution that achieves the target MA within tolerance."""
    best_ma = 1.0
    best_chain = []
    best_error = float('inf')
    found_solution = False
    
    # Try all possible chain lengths from 2 to max_chain_length (must be even for valid gear pairs)
    for chain_length in range(2, min(max_chain_length + 1, len(available_gears) + 1), 2):
        # Try all combinations of gears for this chain length
        for gear_combination in combinations(available_gears, chain_length):
            # Try all permutations of this combination
            from itertools import permutations
            for gear_chain in permutations(gear_combination):
                ma = calculate_mechanical_advantage(list(gear_chain))
                error_ratio = abs(ma - target_ma) / target_ma
                
                if is_within_tolerance(ma, target_ma, tolerance):
                    return True, list(gear_chain), ma, error_ratio
                
                # Keep track of the best solution even if not within tolerance
                if error_ratio < best_error:
                    best_error = error_ratio
                    best_ma = ma
                    best_chain = list(gear_chain)
    
    return False, best_chain, best_ma, best_error

def analyze_level(level_file: str) -> Dict[str, Any]:
    """Analyze a single level file and determine maximum achievable reward."""
    with open(level_file, 'r') as f:
        level_data = yaml.safe_load(f)
    
    available_gears = level_data['gear_system']['available_gears']
    target_ma = level_data['gear_system']['target_ma']
    tolerance = level_data['globals']['tolerance']
    
    # Find optimal solution
    solvable, best_chain, best_ma, best_error = find_optimal_solution(
        available_gears, target_ma, tolerance
    )
    
    max_reward = 1.0 if solvable else 0.0
    
    return {
        'max_reward': max_reward,
        'solvable': solvable,
        'target_ma': target_ma,
        'best_achievable_ma': best_ma,
        'best_error_ratio': best_error,
        'optimal_chain': best_chain,
        'available_gears': available_gears,
        'tolerance': tolerance,
        'calculation_method': 'exhaustive_search'
    }

def main():
    """Main function to analyze all levels and generate the JSON report."""
    levels_dir = './levels'
    results = {
        'environment_id': '20250918_153416_env_77_mechanical_gear_optimization',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {}
    }
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    total_max_reward = 0.0
    solvable_count = 0
    max_rewards = []
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        level_path = os.path.join(levels_dir, level_file)
        print(f"Analyzing {level_file}...")
        
        try:
            analysis = analyze_level(level_path)
            results['levels'][level_file] = {
                'max_reward': analysis['max_reward'],
                'calculation_method': analysis['calculation_method'],
                'notes': f"Target MA: {analysis['target_ma']:.4f}, Best achievable: {analysis['best_achievable_ma']:.4f}, Error: {analysis['best_error_ratio']:.1%}"
            }
            
            if analysis['solvable']:
                results['levels'][level_file]['optimal_solution'] = {
                    'gear_chain': analysis['optimal_chain'],
                    'achieved_ma': analysis['best_achievable_ma']
                }
                solvable_count += 1
            
            total_max_reward += analysis['max_reward']
            max_rewards.append(analysis['max_reward'])
            
            print(f"  - Max reward: {analysis['max_reward']}")
            print(f"  - Solvable: {analysis['solvable']}")
            print(f"  - Target MA: {analysis['target_ma']:.4f}")
            print(f"  - Best achievable MA: {analysis['best_achievable_ma']:.4f}")
            
        except Exception as e:
            print(f"Error analyzing {level_file}: {e}")
            results['levels'][level_file] = {
                'max_reward': 0.0,
                'calculation_method': 'error',
                'notes': f"Error during analysis: {str(e)}"
            }
            max_rewards.append(0.0)
    
    # Calculate summary statistics
    results['summary'] = {
        'total_levels': len(level_files),
        'solvable_levels': solvable_count,
        'unsolvable_levels': len(level_files) - solvable_count,
        'average_max_reward': total_max_reward / len(level_files) if level_files else 0.0,
        'min_max_reward': min(max_rewards) if max_rewards else 0.0,
        'max_max_reward': max(max_rewards) if max_rewards else 0.0,
        'total_possible_reward': total_max_reward
    }
    
    # Write results to JSON file
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary:")
    print(f"Total levels: {results['summary']['total_levels']}")
    print(f"Solvable levels: {results['summary']['solvable_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"Total possible reward: {results['summary']['total_possible_reward']:.1f}")
    print(f"\nResults saved to level_max_rewards.json")

if __name__ == '__main__':
    main()