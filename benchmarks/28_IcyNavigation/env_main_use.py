#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_main import ReverseLakeNavEnv
import argparse

def generate_levels(num_levels: int = 5, seed_base: int = 42):
    """Generate multiple levels for the Reverse Lake Navigation environment."""
    
    # Create environment instance
    env = ReverseLakeNavEnv("reverse_lake_nav")
    
    print(f"Generating {num_levels} levels...")
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = seed_base + i
        print(f"\nGenerating level {i+1}/{num_levels} with seed {seed}")
        
        try:
            # Generate new world
            world_state = env.reset(mode="generate", seed=seed)
            
            # Get the world_id that was just generated
            world_id = f"world_{seed}_{int(__import__('time').time() * 1000) // 1000}"
            
            print(f"  ✓ Generated world: {world_id}")
            print(f"  - Agent start: {world_state['agent']['pos']}")
            print(f"  - Goal position: {world_state['objects']['goal_flag']['pos']}")
            print(f"  - Ice tiles: {len(world_state['objects']['ice_tiles'])}")
            
            generated_worlds.append(world_id)
            
        except Exception as e:
            print(f"  ✗ Failed to generate level {i+1}: {e}")
    
    print(f"\n=== Generation Complete ===")
    print(f"Successfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    return generated_worlds

def test_level(world_id: str):
    """Test a specific level by loading and displaying it."""
    
    env = ReverseLakeNavEnv("reverse_lake_nav")
    
    try:
        print(f"Testing level: {world_id}")
        
        # Load the world
        world_state = env.reset(mode="load", world_id=world_id)
        
        # Get initial observation
        obs = env.observe_semantic()
        rendered = env.render_skin(obs)
        
        print("\n=== Initial State ===")
        print(rendered)
        print(f"\nWorld loaded successfully!")
        print(f"Agent position: {world_state['agent']['pos']}")
        print(f"Goal position: {world_state['objects']['goal_flag']['pos']}")
        print(f"Number of ice tiles: {len(world_state['objects']['ice_tiles'])}")
        
    except Exception as e:
        print(f"Failed to test level {world_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate and test Reverse Lake Navigation levels")
    parser.add_argument("--generate", "-g", type=int, default=5, 
                       help="Number of levels to generate (default: 5)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Base seed for generation (default: 42)")
    parser.add_argument("--test", "-t", type=str, 
                       help="Test a specific world by world_id")
    
    args = parser.parse_args()
    
    if args.test:
        test_level(args.test)
    else:
        generated_worlds = generate_levels(args.generate, args.seed)
        
        # Test the first generated level
        if generated_worlds:
            print(f"\n=== Testing First Generated Level ===")
            test_level(generated_worlds[0])

if __name__ == "__main__":
    main()