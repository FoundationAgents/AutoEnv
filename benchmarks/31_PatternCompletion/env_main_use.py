#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_main import MaskedPixelArtEnv
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate levels for Masked Pixel Art Completion Environment")
    parser.add_argument("--count", type=int, default=5, help="Number of levels to generate")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for generation")
    args = parser.parse_args()
    
    # Create environment instance
    env = MaskedPixelArtEnv(env_id=1)
    
    print(f"Generating {args.count} levels...")
    
    for i in range(args.count):
        seed = args.seed + i if args.seed is not None else None
        
        try:
            # Generate new world
            world_id = env._generate_world(seed=seed)
            print(f"Generated level {i+1}: {world_id}")
            
            # Test loading and basic functionality
            env.reset(mode="load", world_id=world_id)
            obs = env.observe_semantic()
            rendered = env.render_skin(obs)
            
            print(f"Level {world_id} test successful")
            print(f"Masked positions: {len(env._state['canvas']['masked_positions'])}")
            print("---")
            
        except Exception as e:
            print(f"Error generating level {i+1}: {e}")
            continue
    
    print(f"Level generation complete. Files saved to ./levels/")

if __name__ == "__main__":
    main()