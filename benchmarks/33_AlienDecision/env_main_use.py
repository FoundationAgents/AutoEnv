#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_main import AlienColonyEnv
import random

def generate_levels(num_levels=5, start_seed=1000):
    """Generate multiple levels for the AlienColony environment"""
    
    print("Generating AlienColony levels...")
    print("=" * 50)
    
    # Create environment instance
    env = AlienColonyEnv(env_id=107)
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"\nGenerating Level {i+1}/{num_levels} (seed: {seed})")
        
        try:
            # Generate new world
            obs = env.reset(mode="generate", seed=seed)
            world_id = f"world_{seed}"  # Simplified world ID for testing
            
            print(f"✓ Generated world: {world_id}")
            print("Initial state preview:")
            print("-" * 30)
            print(obs[:300] + "..." if len(obs) > 300 else obs)
            
            generated_worlds.append(world_id)
            
        except Exception as e:
            print(f"✗ Failed to generate level {i+1}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print(f"Generation complete! Created {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    # Test loading a generated level
    if generated_worlds:
        print(f"\nTesting level loading with {generated_worlds[0]}...")
        try:
            obs = env.reset(mode="load", world_id=generated_worlds[0])
            print("✓ Level loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load level: {e}")

def test_environment():
    """Test the environment with sample actions"""
    
    print("\nTesting environment functionality...")
    print("=" * 50)
    
    env = AlienColonyEnv(env_id=107)
    
    # Generate and load a test level
    obs = env.reset(mode="generate", seed=42)
    print("Initial observation:")
    print(obs[:400] + "..." if len(obs) > 400 else obs)
    
    # Test some actions
    test_actions = [
        {"action": "gather_resource", "params": {"resource_type": "toxic_waste", "amount": 3}},
        {"action": "allocate_resource", "params": {"resource_type": "toxic_waste", "allocation_amount": 2, "target_system": "colony"}},
        {"action": "explore_area", "params": {"direction": "north", "investment_level": 2}},
        {"action": "build_structure", "params": {"building_type": "waste_processor", "x": 1, "y": 1}},
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1}: {action}")
        try:
            state, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Result: {info.get('last_action_result', 'No result')}")
            
            if done:
                print("Episode finished!")
                break
                
        except Exception as e:
            print(f"✗ Action failed: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_environment()
        elif sys.argv[1] == "generate":
            num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            start_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
            generate_levels(num_levels, start_seed)
        else:
            print("Usage:")
            print("  python env_main_code_use.py generate [num_levels] [start_seed]")
            print("  python env_main_code_use.py test")
    else:
        # Default: generate 3 levels
        generate_levels(3, 1000)