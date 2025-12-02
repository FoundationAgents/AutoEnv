from env_main import MemoryPairEnv
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_levels.py <num_levels> [seed_start]")
        return
    
    num_levels = int(sys.argv[1])
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    env = MemoryPairEnv(env_id=1)
    
    generated_worlds = []
    for i in range(num_levels):
        seed = seed_start + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        world_id = env._generate_world(seed)
        generated_worlds.append(world_id)
        print(f"Generated: {world_id}")
    
    print(f"\nSuccessfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")

if __name__ == "__main__":
    main()