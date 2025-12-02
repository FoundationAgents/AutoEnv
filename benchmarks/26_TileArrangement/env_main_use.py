from env_main import MismatchedMemoryEnv
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Generate levels for Mismatched Memory Game")
    parser.add_argument("--num_levels", type=int, default=5, help="Number of levels to generate")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for generation")
    parser.add_argument("--env_id", type=int, default=102, help="Environment ID")
    
    args = parser.parse_args()
    
    env = MismatchedMemoryEnv(args.env_id)
    
    base_seed = args.seed if args.seed is not None else random.randint(1, 1000000)
    
    print(f"Generating {args.num_levels} levels for Mismatched Memory Game...")
    print(f"Base seed: {base_seed}")
    print("=" * 50)
    
    generated_worlds = []
    
    for i in range(args.num_levels):
        level_seed = base_seed + i
        print(f"Generating level {i+1}/{args.num_levels} with seed {level_seed}...")
        
        try:
            world_id = env._generate_world(level_seed)
            generated_worlds.append(world_id)
            print(f"Successfully generated: {world_id}")
            
            obs = env.reset(mode="load", world_id=world_id)
            print("Level validation: PASSED")
            
        except Exception as e:
            print(f"Error generating level {i+1}: {e}")
        
        print("-" * 30)
    
    print("Generation complete!")
    print(f"Generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")

if __name__ == "__main__":
    main()