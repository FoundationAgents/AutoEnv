#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_main import SquadReconEnv
import random

def generate_levels(num_levels=5, start_seed=1000):
    """Generate multiple levels for the Squad Reconnaissance environment."""
    
    # Create environment instance
    env = SquadReconEnv(env_id=1)
    
    print(f"Generating {num_levels} levels for Squad Reconnaissance and Elimination...")
    print("=" * 60)
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        try:
            # Generate new world
            world_id = env._generate_world(seed=seed)
            generated_worlds.append(world_id)
            
            # Test the generated world by resetting and taking a step
            state = env.reset(mode="load", world_id=world_id)
            
            print(f"  ✓ Successfully generated world: {world_id}")
            print(f"  - Grid size: {state['globals']['grid_size']}")
            print(f"  - Enemy camps: {len(state['enemy_camps'])}")
            print(f"  - Walls: {len(state['terrain']['walls'])}")
            print(f"  - Forests: {len(state['terrain']['forests'])}")
            print(f"  - Squad strengths: {[s['strength'] for s in state['squads']]}")
            print()
            
        except Exception as e:
            print(f"  ✗ Failed to generate level {i+1}: {e}")
            continue
    
    print("=" * 60)
    print(f"Level generation complete!")
    print(f"Successfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    print(f"\nLevel files saved in: ./levels/")
    print("You can now load these levels using env.reset(mode='load', world_id='WORLD_ID')")
    
    return generated_worlds

def test_generated_level():
    """Test a generated level by running a few steps."""
    
    print("\nTesting generated level...")
    print("=" * 40)
    
    env = SquadReconEnv(env_id=1)
    
    # Generate and load a test world
    world_id = env._generate_world(seed=42)
    state = env.reset(mode="load", world_id=world_id)
    
    print("Initial observation:")
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print(rendered)
    print()
    
    # Test a few actions
    test_actions = [
        {"action": "MOVE_NORTH", "params": {"squad_id": 0}},
        {"action": "MOVE_EAST", "params": {"squad_id": 1}},
        {"action": "HOLD_POSITION", "params": {"squad_id": 2}}
    ]
    
    for i, action in enumerate(test_actions, 1):
        print(f"Step {i}: {action}")
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        if done:
            print("Episode ended!")
            break
        print()

if __name__ == "__main__":
    # Generate multiple levels
    generated_worlds = generate_levels(num_levels=3, start_seed=1000)
    
    # Test one level
    if generated_worlds:
        test_generated_level()