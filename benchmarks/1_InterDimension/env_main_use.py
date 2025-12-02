#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

from env_main import InterdimensionalMarketEnv
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate levels for Interdimensional Market Trading Environment')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed_start', type=int, default=1000, help='Starting seed value')
    parser.add_argument('--test_run', action='store_true', help='Run a test episode after generation')
    
    args = parser.parse_args()
    
    # Create environment instance
    env = InterdimensionalMarketEnv(env_id=1)
    
    print(f"Generating {args.num_levels} levels for Interdimensional Market Trading Environment...")
    
    generated_worlds = []
    
    # Generate levels
    for i in range(args.num_levels):
        seed = args.seed_start + i
        print(f"Generating level {i+1}/{args.num_levels} with seed {seed}...")
        
        try:
            world_id = env._generate_world(seed=seed)
            generated_worlds.append(world_id)
            print(f"  Generated world: {world_id}")
        except Exception as e:
            print(f"  Error generating world with seed {seed}: {e}")
    
    print(f"\nSuccessfully generated {len(generated_worlds)} worlds:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    # Test run if requested
    if args.test_run and generated_worlds:
        print(f"\nRunning test episode with world: {generated_worlds[0]}")
        try:
            env.reset(mode="load", world_id=generated_worlds[0])
            obs = env.observe_semantic()
            rendered = env.render_skin(obs)
            print("\nInitial state:")
            print(rendered)
            
            # Test a simple action
            test_action = {"action": "RESEARCH", "params": {}}
            state, reward, done, info = env.step(test_action)
            print(f"\nAfter RESEARCH action:")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Action result: {info.get('last_action_result', 'No result')}")
            
        except Exception as e:
            print(f"Error during test run: {e}")

if __name__ == "__main__":
    main()