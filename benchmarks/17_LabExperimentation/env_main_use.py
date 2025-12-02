#!/usr/bin/env python3

from env_main import BizarroLabEnv
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Generate levels for BizarroLab environment")
    parser.add_argument("--num_levels", type=int, default=5, help="Number of levels to generate")
    parser.add_argument("--seed_start", type=int, default=1000, help="Starting seed value")
    
    args = parser.parse_args()
    
    env = BizarroLabEnv(env_id=1)
    
    print(f"Generating {args.num_levels} levels for BizarroLab environment...")
    
    for i in range(args.num_levels):
        seed = args.seed_start + i
        print(f"Generating level {i+1}/{args.num_levels} with seed {seed}...")
        
        try:
            world_id = env._generate_world(seed)
            print(f"  Generated world ID: {world_id}")
            
            obs = env.reset(mode="load", world_id=world_id)
            print(f"  Target compound: {obs['globals']['target_compound']}")
            print(f"  Success! Level saved as: ./levels/{world_id}.yaml")
        except Exception as e:
            print(f"  Error generating level {i+1}: {e}")
    
    print("Level generation complete!")

if __name__ == "__main__":
    main()