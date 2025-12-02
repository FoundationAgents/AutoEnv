#!/usr/bin/env python3

import os
import sys
import argparse
from env_main import ElectromagneticEnv

def main():
    parser = argparse.ArgumentParser(description='Generate levels for Electromagnetic Anomaly Detection Environment')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed_start', type=int, default=1000, help='Starting seed for level generation')
    parser.add_argument('--env_id', type=int, default=1, help='Environment ID')
    
    args = parser.parse_args()
    
    # Create levels directory if it doesn't exist
    os.makedirs('./levels', exist_ok=True)
    
    # Initialize environment
    env = ElectromagneticEnv(args.env_id)
    
    print(f"Generating {args.num_levels} levels...")
    
    generated_levels = []
    for i in range(args.num_levels):
        seed = args.seed_start + i
        print(f"Generating level {i+1}/{args.num_levels} with seed {seed}...")
        
        try:
            # Generate new world
            world_id = env._generate_world(seed)
            generated_levels.append(world_id)
            print(f"  ✓ Generated: {world_id}")
            
            # Test loading the generated world
            obs = env.reset(mode="load", world_id=world_id)
            print(f"  ✓ Verified loadable")
            
        except Exception as e:
            print(f"  ✗ Error generating level with seed {seed}: {e}")
            continue
    
    print(f"\nSuccessfully generated {len(generated_levels)} levels:")
    for level in generated_levels:
        print(f"  - {level}.yaml")
    
    print(f"\nLevels saved to: ./levels/")
    print("You can now load these levels using env.reset(mode='load', world_id='world_id')")

if __name__ == "__main__":
    main()