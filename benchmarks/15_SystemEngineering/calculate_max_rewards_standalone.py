import yaml
import os
from datetime import datetime
import json

class HydraulicSimulator:
    """Standalone hydraulic simulator based on the logic from env_main.py"""
    
    def calculate_steady_state(self, valve_states, pump_speed, reservoir_pressure):
        base_pressure = reservoir_pressure
        pipe_pressures = []
        
        # Calculate pressure for each pipe section based on valve states and network topology
        for i in range(12):
            pressure_modifier = 1.0
            
            # Apply valve influence based on network topology
            for j, valve_open in enumerate(valve_states):
                if valve_open:
                    # Open valve increases pressure in downstream pipes
                    if (i + j) % 3 == 0:
                        pressure_modifier += 0.1
                else:
                    # Closed valve creates backpressure
                    if (i + j) % 4 == 0:
                        pressure_modifier += 0.05
            
            # Add pump influence
            pump_influence = (pump_speed / 1500.0) * (1.0 + 0.1 * (i % 3))
            
            pipe_pressure = base_pressure * pressure_modifier * pump_influence
            pipe_pressures.append(pipe_pressure)
        
        # Extract sensor readings from specific pipe locations
        sensor_readings = [
            pipe_pressures[2],   # Sensor 1 at pipe 2
            pipe_pressures[5],   # Sensor 2 at pipe 5  
            pipe_pressures[8],   # Sensor 3 at pipe 8
            pipe_pressures[11]   # Sensor 4 at pipe 11
        ]
        
        return pipe_pressures, sensor_readings

def calculate_max_reward_for_level(level_file):
    """Calculate the theoretical maximum reward for a given level."""
    # Load the level
    with open(f'levels/{level_file}', 'r') as f:
        level_data = yaml.safe_load(f)
    
    # Extract target pressures
    target_pressures = level_data['hydraulics']['target_pressures']
    
    # Initialize simulator
    simulator = HydraulicSimulator()
    pump_speed = level_data['hydraulics']['pump_speed']
    reservoir_pressure = level_data['hydraulics']['reservoir_pressure']
    
    # Try all possible valve combinations (2^9 = 512 combinations)
    best_reward = 0.0
    solvable = False
    solution_found = None
    
    for valve_config in range(512):  # 2^9 combinations
        # Convert integer to binary valve states
        valve_states = []
        for i in range(9):
            valve_states.append(bool((valve_config >> i) & 1))
        
        # Calculate resulting pressures
        pipe_pressures, sensor_readings = simulator.calculate_steady_state(
            valve_states, pump_speed, reservoir_pressure
        )
        
        # Check if all sensors match targets (with same tolerance as in env_main.py)
        all_match = True
        for i in range(4):
            if abs(sensor_readings[i] - target_pressures[i]) > 1e-6:
                all_match = False
                break
        
        if all_match:
            best_reward = 1.0
            solvable = True
            solution_found = valve_states
            break
    
    return {
        'max_reward': best_reward,
        'solvable': solvable,
        'target_pressures': target_pressures,
        'solution': solution_found
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
            'reward_structure': 'binary_reward_1.0_for_exact_match_all_sensors',
            'tolerance': '1e-6'
        }
    }
    
    max_rewards = []
    solvable_count = 0
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for level_file in level_files:
        print(f"Processing {level_file}...")
        result = calculate_max_reward_for_level(level_file)
        
        notes = f"Target pressures: {result['target_pressures']}"
        if result['solvable']:
            notes += f" | Solution found: valve_states={result['solution']}"
        else:
            notes += " | No exact solution found with exhaustive search"
        
        results['levels'][level_file] = {
            'max_reward': result['max_reward'],
            'calculation_method': 'exhaustive_search_512_valve_combinations',
            'solvable': result['solvable'],
            'notes': notes
        }
        
        max_rewards.append(result['max_reward'])
        if result['solvable']:
            solvable_count += 1
            print(f"  -> SOLVABLE (max_reward = 1.0)")
        else:
            print(f"  -> NOT SOLVABLE (max_reward = 0.0)")
    
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
    print(f"Min max reward: {results['summary']['min_max_reward']:.3f}")
    print(f"Max max reward: {results['summary']['max_max_reward']:.3f}")

if __name__ == "__main__":
    main()