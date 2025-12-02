#!/usr/bin/env python3

from env_main import MagneticFieldEnv
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate levels for Magnetic Field Environment')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed_start', type=int, default=1000, help='Starting seed value')
    
    args = parser.parse_args()
    
    env = MagneticFieldEnv("magnetic_field_decode")
    
    print(f"Generating {args.num_levels} levels...")
    
    for i in range(args.num_levels):
        seed = args.seed_start + i
        print(f"Generating level {i+1} with seed {seed}")
        
        try:
            world_id = env._generate_world(seed=seed)
            print(f"  Generated: {world_id}")
            
            state = env.reset(mode="load", world_id=world_id)
            print(f"  Message: {state['grid']['encoded_message']}")
            
            obs = env.observe_semantic()
            rendered = env.render_skin(obs)
            print(f"  Grid preview (first 3 rows):")
            for row_idx, row in enumerate(state['grid']['pattern'][:3]):
                print(f"    {' '.join(str(cell) for cell in row)}")
            print()
            
        except Exception as e:
            print(f"  Error generating level {i+1}: {e}")
    
    print("Level generation complete!")

if __name__ == "__main__":
    main()