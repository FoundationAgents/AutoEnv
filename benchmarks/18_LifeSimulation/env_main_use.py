#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

from env_main import ValleyFarmEnv

def generate_levels():
    """Generate sample levels for the Valley Farm environment"""
    env = ValleyFarmEnv(env_id=1)
    
    print("Generating Valley Farm levels...")
    
    # Generate 3 levels with different seeds
    seeds = [42, 123, 999]
    
    for i, seed in enumerate(seeds):
        print(f"\nGenerating level {i+1} with seed {seed}...")
        
        # Reset with generation mode
        obs = env.reset(mode="generate", seed=seed)
        
        print(f"Level {i+1} generated successfully!")
        print("Initial observation:")
        print(env.render_skin(obs))
        
        # Test a few basic actions
        print("\nTesting basic actions:")
        
        # Try moving
        state, reward, done, info = env.step({"action": "MoveNorth", "params": {}})
        print(f"Move North - Reward: {reward}, Done: {done}")
        
        # Try waiting
        state, reward, done, info = env.step({"action": "Wait", "params": {}})
        print(f"Wait - Reward: {reward}, Done: {done}")
        
        print(f"Level {i+1} testing complete!")
    
    print(f"\nAll levels generated successfully!")
    print("Level files saved in ./levels/ directory")

if __name__ == "__main__":
    generate_levels()