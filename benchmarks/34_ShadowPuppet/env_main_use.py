#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from env_main import ShadowPuppetEnv
import yaml

def generate_levels():
    env = ShadowPuppetEnv(env_id=1)
    
    print("Generating 5 levels for Shadow Puppet Environment...")
    
    for i in range(1, 6):
        seed = 1000 + i
        print(f"Generating level {i} with seed {seed}...")
        
        state = env.reset(mode="generate", seed=seed)
        world_id = f"level_{i}"
        
        level_path = f"./levels/{world_id}.yaml"
        with open(level_path, 'w') as f:
            yaml.dump(state, f, default_flow_style=False)
        
        print(f"Level {i} saved to {level_path}")
        
        obs = env.observe_semantic()
        display = env.render_skin(obs)
        print(f"\nLevel {i} preview:")
        print(display)
        print("-" * 50)
    
    print("All levels generated successfully!")

if __name__ == "__main__":
    generate_levels()