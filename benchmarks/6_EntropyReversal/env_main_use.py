#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_main import EntropyReversalEnv

def generate_levels():
    env = EntropyReversalEnv(env_id=1)
    
    print("Generating Entropy Reversal levels...")
    
    seeds = [42, 123, 456, 789, 999]
    
    for i, seed in enumerate(seeds):
        world_id = env._generate_world(seed)
        print(f"Generated level {i+1}: {world_id} (seed: {seed})")
    
    print(f"\nGenerated {len(seeds)} levels in ./levels/ directory")
    print("Levels are saved as YAML files and can be loaded using:")
    print("env.reset(mode='load', world_id='<world_id>')")

if __name__ == "__main__":
    generate_levels()