#!/usr/bin/env python3

import sys
import os
sys.path.append('.')
sys.path.append('../../../')  # Add root directory to find autoenv

from env_main import PressureValveEnv
import random

def generate_levels(num_levels=10, start_seed=1000):
    """Generate multiple levels for the Pressure Valve Engineering environment."""
    
    # Create environment instance
    env = PressureValveEnv(env_id=1)
    
    print(f"Generating {num_levels} levels for Pressure Valve Engineering Environment...")
    print("=" * 60)
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"\nGenerating Level {i+1}/{num_levels} (seed: {seed})")
        
        try:
            # Reset environment in generate mode with specific seed
            state = env.reset(mode="generate", seed=seed)
            
            # Get initial observation to verify the level works
            initial_obs = env.observe_semantic()
            
            # Find the generated world file
            level_files = [f for f in os.listdir('./levels/') if f.endswith('.yaml')]
            if level_files:
                level_files.sort(key=lambda f: os.path.getmtime(f'./levels/{f}'), reverse=True)
                world_id = level_files[0].replace('.yaml', '')
                generated_worlds.append(world_id)
            
            print(f"✓ Successfully generated level with seed {seed}")
            print(f"  - Initial valve states: {''.join(['1' if v else '0' for v in initial_obs['valve_states']])}")
            print(f"  - Target pressures: {[f'{p:.1f}' for p in initial_obs['target_pressures']]}")
            print(f"  - Initial sensor readings: {[f'{p:.1f}' for p in initial_obs['sensor_readings']]}")
            
        except Exception as e:
            print(f"✗ Failed to generate level with seed {seed}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Level generation complete! Generated {len(generated_worlds)} levels.")
    print("\nTo test a level, you can use:")
    print("```python")
    print("env = PressureValveEnv(1)")
    print("env.reset(mode='load', world_id='WORLD_ID')")
    print("obs = env.observe_semantic()")
    print("print(env.render_skin(obs))")
    print("```")
    
    return generated_worlds

def test_environment():
    """Test the environment with a simple interaction."""
    print("\nTesting environment functionality...")
    print("-" * 40)
    
    env = PressureValveEnv(env_id=1)
    
    # Generate and load a test level
    env.reset(mode="generate", seed=12345)
    
    print("Initial State:")
    obs = env.observe_semantic()
    print(env.render_skin(obs))
    
    print("\nTesting actions...")
    
    # Test NO_OP action
    action = {"action": "NO_OP", "params": {}}
    new_state = env.transition(action)
    reward, events, info = env.reward(action)
    done = env.done()
    print(f"\nAfter NO_OP: reward={reward}, done={done}")
    
    # Test TOGGLE_VALVE action  
    action = {"action": "TOGGLE_VALVE", "params": {"valve_id": 0}}
    new_state = env.transition(action)
    reward, events, info = env.reward(action)
    done = env.done()
    print(f"After TOGGLE_VALVE(0): reward={reward}, done={done}")
    print(f"Action result: {env._last_action_result}")
    
    obs = env.observe_semantic()
    print(f"New valve states: {''.join(['1' if v else '0' for v in obs['valve_states']])}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_environment()
    else:
        num_levels = 5
        if len(sys.argv) > 1:
            try:
                num_levels = int(sys.argv[1])
            except ValueError:
                print("Invalid number of levels specified. Using default: 5")
        
        generate_levels(num_levels)
