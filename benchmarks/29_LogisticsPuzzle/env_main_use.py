#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_main import InvertedBoxEscapeEnv
import random

def generate_levels(num_levels=10, start_seed=1000):
    """Generate multiple levels for the Inverted Box Escape environment."""
    
    # Create environment instance
    env = InvertedBoxEscapeEnv(env_id=1)
    
    print(f"Generating {num_levels} levels for Inverted Box Escape environment...")
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        try:
            # Generate new world
            world_id = env._generate_world(seed=seed)
            generated_worlds.append(world_id)
            print(f"  ✓ Generated world: {world_id}")
            
            # Test the world by loading it
            env.reset(mode="load", world_id=world_id)
            print(f"  ✓ Successfully validated world: {world_id}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate level {i+1}: {str(e)}")
            continue
    
    print(f"\n=== Generation Summary ===")
    print(f"Successfully generated {len(generated_worlds)} out of {num_levels} levels")
    print(f"Generated worlds: {generated_worlds}")
    print(f"Levels saved to: ./levels/")
    
    return generated_worlds

def test_environment():
    """Test the generated environment with a sample episode."""
    
    print("\n=== Testing Environment ===")
    
    # Create environment
    env = InvertedBoxEscapeEnv(env_id=1)
    
    # Generate and load a test world
    world_id = env._generate_world(seed=42)
    print(f"Generated test world: {world_id}")
    
    # Reset environment
    obs = env.reset(mode="load", world_id=world_id)
    print("Initial observation:")
    rendered = env.render_skin(obs)
    print(rendered)
    
    # Try a few random actions
    actions = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST", "WAIT"]
    
    for step in range(5):
        action = random.choice(actions)
        action_dict = {"action": action, "params": {}}
        
        print(f"\n--- Step {step + 1}: {action} ---")
        
        state, reward, done, info = env.step(action_dict)
        rendered_obs = info['skinned']
        print(rendered_obs)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        
        if done:
            print("Episode terminated!")
            break
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        try:
            num_levels = int(sys.argv[1])
        except ValueError:
            print("Usage: python env_main_code_use.py [num_levels]")
            print("num_levels should be an integer")
            sys.exit(1)
    else:
        num_levels = 10
    
    # Generate levels
    generated_worlds = generate_levels(num_levels)
    
    # Test environment if any worlds were generated
    if generated_worlds:
        test_environment()
    else:
        print("No worlds were generated successfully. Please check the configuration.")