import yaml
import argparse
from env_main import DeceptiveGridWorld

def generate_level(env_id="deceptive_grid_world_v1", seed=None, num_levels=1):
    """Generate one or more levels for the Deceptive Grid World environment."""
    
    # Create environment instance
    env = DeceptiveGridWorld(env_id)
    
    generated_worlds = []
    
    for i in range(num_levels):
        # Use provided seed or generate based on index
        current_seed = seed + i if seed is not None else None
        
        # Generate and reset environment
        env.reset(mode="generate", seed=current_seed)
        
        # Get the world ID from the last generated world
        world_id = env._generate_world(current_seed)
        generated_worlds.append(world_id)
        
        print(f"Generated level {i+1}/{num_levels}: {world_id}")
        
        # Optionally display the world
        obs = env.observe_semantic()
        rendered = env.render_skin(obs)
        print("\nGenerated world preview:")
        print(rendered)
        print("-" * 50)
    
    return generated_worlds

def main():
    parser = argparse.ArgumentParser(description="Generate levels for Deceptive Grid World")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--num-levels", type=int, default=1, help="Number of levels to generate")
    parser.add_argument("--env-id", type=str, default="deceptive_grid_world_v1", help="Environment ID")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_levels} level(s) with seed {args.seed}...")
    
    generated = generate_level(
        env_id=args.env_id,
        seed=args.seed,
        num_levels=args.num_levels
    )
    
    print(f"\nSuccessfully generated {len(generated)} levels:")
    for world_id in generated:
        print(f"  - {world_id}")

if __name__ == "__main__":
    main()