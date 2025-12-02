#!/usr/bin/env python3

import yaml
import json
import os
from datetime import datetime
import numpy as np
from typing import Dict, Any, List

def load_level(level_file: str) -> Dict[str, Any]:
    """Load a level YAML file"""
    with open(level_file, 'r') as f:
        return yaml.safe_load(f)

def load_config() -> Dict[str, Any]:
    """Load the environment configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_exchange_rate_at_step(drift_params: List[float], step: int) -> float:
    """Calculate exchange rate at a given step using drift parameters"""
    amplitude, frequency, phase = drift_params
    base_rate = 1.0
    rate = base_rate + amplitude * np.sin(frequency * step + phase)
    return max(0.1, rate)  # Prevent negative rates

def find_optimal_trading_sequence(level_data: Dict[str, Any], max_steps: int = 40) -> Dict[str, Any]:
    """
    Analyze optimal trading sequence for maximum profit.
    This is a simplified heuristic approach focusing on:
    1. Converting all inventory to credits via optimal trades
    2. Currency arbitrage using exchange rate fluctuations
    3. Minimizing penalties and maximizing bonuses
    """
    
    # Initial conditions
    inventory = level_data['agent']['inventory'].copy()
    ledgers = level_data['agent']['ledgers'].copy()
    embargo_risks = level_data['market']['embargo_risks'].copy()
    drift_params = level_data['market']['exchange_matrices']['drift_params']
    
    # Calculate total inventory value (10 credits per item as per env_main.py)
    total_inventory_value = sum(qty for qty in inventory.values() if qty > 0) * 10
    
    # Initial ledger total
    initial_total = sum(ledgers.values())
    
    # Find best exchange rates over time for arbitrage opportunities
    max_rates = {}
    min_rates = {}
    
    for rate_pair, params in drift_params.items():
        rates_over_time = []
        for step in range(max_steps):
            rate = calculate_exchange_rate_at_step(params, step)
            rates_over_time.append(rate)
        
        max_rates[rate_pair] = max(rates_over_time)
        min_rates[rate_pair] = min(rates_over_time)
    
    # Theoretical maximum profit calculation
    # 1. Convert all inventory to the dimension with best sell rates
    max_profit_from_inventory = total_inventory_value
    
    # 2. Currency arbitrage potential
    # Find the best arbitrage cycle (e.g., mass -> entropy -> historical -> mass)
    arbitrage_multiplier = 1.0
    dimensions = ['mass', 'entropy', 'historical']
    
    # Simple 3-way arbitrage calculation
    for dim1 in dimensions:
        for dim2 in dimensions:
            for dim3 in dimensions:
                if dim1 != dim2 != dim3 != dim1:
                    rate1 = max_rates.get(f"{dim1}_{dim2}", 1.0)
                    rate2 = max_rates.get(f"{dim2}_{dim3}", 1.0)
                    rate3 = max_rates.get(f"{dim3}_{dim1}", 1.0)
                    cycle_multiplier = rate1 * rate2 * rate3
                    arbitrage_multiplier = max(arbitrage_multiplier, cycle_multiplier)
    
    # Apply arbitrage to initial capital
    max_arbitrage_profit = initial_total * (arbitrage_multiplier - 1.0)
    
    # Total theoretical maximum profit
    max_total_profit = max_profit_from_inventory + max_arbitrage_profit
    
    return {
        'max_profit': max_total_profit,
        'inventory_value': total_inventory_value,
        'initial_capital': initial_total,
        'arbitrage_multiplier': arbitrage_multiplier,
        'arbitrage_profit': max_arbitrage_profit
    }

def calculate_maximum_reward(level_file: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the theoretical maximum reward for a level"""
    
    level_data = load_level(level_file)
    max_steps = config['termination']['max_steps']
    
    # Get optimal trading analysis
    trading_analysis = find_optimal_trading_sequence(level_data, max_steps)
    max_profit = trading_analysis['max_profit']
    
    total_reward = 0.0
    reward_breakdown = {}
    
    # 1. Profit rewards (1.0 per credit gained)
    profit_reward = max_profit * 1.0
    total_reward += profit_reward
    reward_breakdown['profit_rewards'] = profit_reward
    
    # 2. Goal achievement bonus (50.0 if profit >= 100.0)
    if max_profit >= 100.0:
        goal_reward = 50.0
        total_reward += goal_reward
        reward_breakdown['goal_rewards'] = goal_reward
    else:
        reward_breakdown['goal_rewards'] = 0.0
    
    # 3. Stability bonus (0.2 per step if embargo risks < 80%)
    # Assume optimal play keeps embargo risks low for most steps
    # Conservative estimate: 80% of steps get stability bonus
    stable_steps = int(max_steps * 0.8)
    stability_reward = stable_steps * 0.2
    total_reward += stability_reward
    reward_breakdown['stability_rewards'] = stability_reward
    
    # 4. Fairness bonus (5.0 per step if all ledgers >= 0)
    # Assume optimal play maintains positive balances for most steps
    # Conservative estimate: 90% of steps get fairness bonus
    fair_steps = int(max_steps * 0.9)
    fairness_reward = fair_steps * 5.0
    total_reward += fairness_reward
    reward_breakdown['fairness_rewards'] = fairness_reward
    
    # 5. Research rewards (3.0 per successful research)
    # Assume doing research every 5 steps for rate improvements
    research_actions = max_steps // 5
    research_reward = research_actions * 3.0
    total_reward += research_reward
    reward_breakdown['research_rewards'] = research_reward
    
    return {
        'max_reward': round(total_reward, 2),
        'reward_breakdown': reward_breakdown,
        'trading_analysis': trading_analysis,
        'calculation_method': 'optimal_path_analysis_with_heuristics',
        'notes': 'Conservative estimate assuming optimal play with 80% stability and 90% fairness bonus achievement'
    }

def main():
    """Main function to calculate maximum rewards for all levels"""
    
    # Load configuration
    config = load_config()
    
    # Get all level files
    levels_dir = './levels/'
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    level_files.sort()
    
    # Calculate maximum rewards for each level
    results = {
        'environment_id': '20250904_235235_env_interdimensional',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {
            'total_levels': len(level_files),
            'average_max_reward': 0.0,
            'min_max_reward': float('inf'),
            'max_max_reward': 0.0
        }
    }
    
    max_rewards = []
    
    for level_file in level_files:
        print(f"Calculating maximum reward for {level_file}...")
        
        level_path = os.path.join(levels_dir, level_file)
        level_result = calculate_maximum_reward(level_path, config)
        
        results['levels'][level_file] = level_result
        max_rewards.append(level_result['max_reward'])
        
        print(f"  Max reward: {level_result['max_reward']}")
    
    # Calculate summary statistics
    results['summary']['average_max_reward'] = round(np.mean(max_rewards), 2)
    results['summary']['min_max_reward'] = round(min(max_rewards), 2)
    results['summary']['max_max_reward'] = round(max(max_rewards), 2)
    
    # Save results to JSON file
    output_file = 'level_max_rewards.json'
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