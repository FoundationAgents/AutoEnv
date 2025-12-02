#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

from env_main import AtmosphereEnv

def generate_levels():
    """Generate multiple atmosphere regulation levels for training and testing."""
    
    # Create levels directory
    os.makedirs('./levels', exist_ok=True)
    
    # Initialize environment
    env = AtmosphereEnv(env_id=1)
    
    # Generate training levels with different seeds
    print("Generating training levels...")
    training_seeds = [42, 123, 456, 789, 1011, 1314, 1617, 1920, 2223, 2526]
    
    for i, seed in enumerate(training_seeds):
        world_id = f"training_{i+1:02d}"
        try:
            env.reset(mode="generate", seed=seed)
            # Rename generated file to match training convention
            generated_id = f"world_{seed}"
            old_path = f"./levels/{generated_id}.yaml"
            new_path = f"./levels/{world_id}.yaml"
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            print(f"Generated training level: {world_id}")
        except Exception as e:
            print(f"Error generating training level {world_id}: {e}")
    
    # Generate test levels
    print("\nGenerating test levels...")
    test_seeds = [9999, 8888, 7777, 6666, 5555]
    
    for i, seed in enumerate(test_seeds):
        world_id = f"test_{i+1:02d}"
        try:
            env.reset(mode="generate", seed=seed)
            # Rename generated file to match test convention
            generated_id = f"world_{seed}"
            old_path = f"./levels/{generated_id}.yaml"
            new_path = f"./levels/{world_id}.yaml"
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            print(f"Generated test level: {world_id}")
        except Exception as e:
            print(f"Error generating test level {world_id}: {e}")
    
    print(f"\nLevel generation complete! Generated {len(training_seeds)} training levels and {len(test_seeds)} test levels.")
    print("Levels saved in ./levels/ directory")

def test_environment():
    """Test the environment with a sample level."""
    print("\nTesting environment...")
    
    env = AtmosphereEnv(env_id=1)
    
    # Test loading a generated level
    try:
        state = env.reset(mode="load", world_id="training_01")
        obs = env.observe_semantic()
        skin = env.render_skin(obs)
        
        print("Environment loaded successfully!")
        print("Initial observation:")
        print(skin)
        
        # Test a few actions
        actions = [
            {"action": "inject_cold_ions", "params": {}},
            {"action": "release_dry_fog", "params": {}},
            {"action": "vent_heavy_vapor", "params": {}}
        ]
        
        for i, action in enumerate(actions):
            if env.done():
                break
                
            print(f"\n--- Step {i+1}: {action['action']} ---")
            state, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            print(f"Events: {info['events']}")
            print(info['skinned'])
            
            if done:
                print("Episode terminated!")
                break
                
    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_levels()
    test_environment()