import yaml
import os
from datetime import datetime
import json
from env_main import PressureValveEnv, HydraulicSimulator

def calculate_max_reward_for_level(level_file):
    """Calculate the theoretical maximum reward for a given level."""
    # Load the level
    with open(f'levels/{level_file}', 'r') as f:
        level_data = yaml.safe_load(f)
    
    # Extract target pressures
    target_pressures = level_data['hydraulics']['target_pressures']
    
    # The reward system is binary: 1.0 if all sensors match targets exactly, 0.0 otherwise
    # Since the environment generates solvable levels, the maximum reward is always 1.0
    # if there exists at least one valve configuration that can achieve the targets
    
    # We need to check if this level is theoretically solvable
    # Since levels are generated as "solvable" according to the generator config,
    # the maximum reward for each level should be 1.0
    
    # However, let's verify by testing if we can find a solution
    simulator = HydraulicSimulator()
    pump_speed = level_data['hydraulics']['pump_speed']
    reservoir_pressure = level_data['hydraulics']['reservoir_pressure']
    
    # Try all possible valve combinations (2^9 = 512 combinations)
    best_reward = 0.0
    solvable = False
    
    for valve_config in range(512):  # 2^9 combinations
        # Convert integer to binary valve states
        valve_states = []
        for i in range(9):
            valve_states.append(bool((valve_config >> i) & 1))
        
        # Calculate resulting pressures
        pipe_pressures, sensor_readings = simulator.calculate_steady_state(
            valve_states, pump_speed, reservoir_pressure
        )
        
        # Check if all sensors match targets
        all_match = True
        for i in range(4):
            if abs(sensor_readings[i] - target_pressures[i]) > 1e-6:
                all_match = False
                break
        
        if all_match:
            best_reward = 1.0
            solvable = True
            break
    
    return {
        'max_reward': best_reward,
        'solvable': solvable,
        'target_pressures': target_pressures
    }

def main():
    # Get all level files
    level_files = [f for f in os.listdir('levels') if f.endswith('.yaml')]
    level_files.sort()
    
    results = {
        'environment_id': '20250918_153416_env_75_hydraulic_system_engineering',
        'calculation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'levels': {},
        'summary': {
            'total_levels': len(level_files),
            'calculation_method': 'exhaustive_valve_configuration_search',
            'reward_structure': 'binary_reward_1.0_for_exact_match_all_sensors'
        }
    }
    
    max_rewards = []
    solvable_count = 0
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        result = calculate_max_reward_for_level(level_file)
        
        results['levels'][level_file] = {
            'max_reward': result['max_reward'],
            'calculation_method': 'exhaustive_search_512_valve_combinations',
            'solvable': result['solvable'],
            'notes': f"Target pressures: {result['target_pressures']}"
        }
        
        max_rewards.append(result['max_reward'])
        if result['solvable']:
            solvable_count += 1
    
    # Calculate summary statistics
    results['summary'].update({
        'solvable_levels': solvable_count,
        'unsolvable_levels': len(level_files) - solvable_count,
        'average_max_reward': sum(max_rewards) / len(max_rewards) if max_rewards else 0,
        'min_max_reward': min(max_rewards) if max_rewards else 0,
        'max_max_reward': max(max_rewards) if max_rewards else 0
    })
    
    # Save results
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to level_max_rewards.json")
    print(f"Summary: {solvable_count}/{len(level_files)} levels are solvable")
    print(f"Average max reward: {results['summary']['average_max_reward']:.3f}")

if __name__ == "__main__":
    main()