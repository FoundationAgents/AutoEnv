#!/usr/bin/env python3

from env_main import UndergroundRuinEnv
import sys
import os

def generate_levels(num_levels=10, start_seed=0):
    """Generate multiple levels for the Underground Ruin environment."""
    
    print(f"Generating {num_levels} levels...")
    
    # Create environment
    env = UndergroundRuinEnv("underground_ruin_v1")
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        # Generate and reset with the new world
        env.reset(mode="generate", seed=seed)
        
        # Get a world ID based on the seed for tracking
        world_id = f"world_s{seed}"
        generated_worlds.append(world_id)
        
        print(f"  Generated world: {world_id}")
    
    print(f"\nGeneration complete! Created {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    print(f"\nLevels saved to: ./levels/")
    return generated_worlds

def test_level(world_id):
    """Test a specific level by loading and displaying it."""
    
    print(f"Testing level: {world_id}")
    
    env = UndergroundRuinEnv("underground_ruin_v1")
    
    try:
        # Load the world
        env.reset(mode="load", world_id=world_id)
        
        # Get initial observation
        raw_obs = env.observe_semantic()
        rendered = env.render_skin(raw_obs)
        
        print("Initial state:")
        print(rendered)
        print()
        
        # Try a few moves
        actions = [
            {"action": "MOVE_EAST", "params": {}},
            {"action": "ROTATE_RIGHT", "params": {}},
            {"action": "MOVE_SOUTH", "params": {}},
            {"action": "WAIT", "params": {}}
        ]
        
        for i, action in enumerate(actions):
            if env.done():
                break
                
            print(f"Action {i+1}: {action['action']}")
            state, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            print(info["skinned"])
            print(f"Done: {done}")
            print("-" * 40)
            
            if done:
                break
    
    except FileNotFoundError:
        print(f"Error: Level {world_id} not found!")
    except Exception as e:
        print(f"Error testing level: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Generate levels: python env_main_code_use.py generate [num_levels] [start_seed]")
        print("  Test level:      python env_main_code_use.py test [world_id]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        start_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        generate_levels(num_levels, start_seed)
        
    elif command == "test":
        if len(sys.argv) < 3:
            print("Please specify a world_id to test")
            sys.exit(1)
        world_id = sys.argv[2]
        test_level(world_id)
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'generate' or 'test'")
        sys.exit(1)