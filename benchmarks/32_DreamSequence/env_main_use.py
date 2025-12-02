#!/usr/bin/env python3

import os
import sys
from env_main import DreamNavEnv

def generate_levels():
    """Generate sample levels for Dream Navigation environment"""
    
    # Create levels directory if it doesn't exist
    os.makedirs("./levels", exist_ok=True)
    
    # Initialize environment
    env = DreamNavEnv(env_id=1)
    
    # Generate levels with different difficulties and seeds
    difficulties = ["Easy", "Medium", "Hard"]
    seeds = [42, 123, 456, 789, 101112]
    
    generated_worlds = []
    
    print("Generating Dream Navigation levels...")
    
    for difficulty in difficulties:
        # Update config for current difficulty
        env.configs['state_template']['globals']['difficulty'] = difficulty
        
        for i, seed in enumerate(seeds):
            print(f"Generating {difficulty} level {i+1} with seed {seed}...")
            
            # Reset in generate mode with specific seed
            try:
                env.reset(mode="generate", seed=seed)
                world_id = f"world_seed_{seed}"
                generated_worlds.append({
                    'world_id': world_id,
                    'difficulty': difficulty,
                    'seed': seed
                })
                print(f"✓ Generated: {world_id}")
                
            except Exception as e:
                print(f"✗ Failed to generate {difficulty} level with seed {seed}: {e}")
    
    # Test loading generated levels
    print(f"\nTesting generated levels...")
    for world_info in generated_worlds[:3]:  # Test first 3
        try:
            env.reset(mode="load", world_id=world_info['world_id'])
            obs = env.observe_semantic()
            rendered = env.render_skin(obs)
            print(f"✓ Successfully loaded and rendered: {world_info['world_id']}")
            print(f"  Start room: {obs['current_room']}, Key in room: {env._state['world']['key_location']}")
            
        except Exception as e:
            print(f"✗ Failed to load {world_info['world_id']}: {e}")
    
    print(f"\nLevel generation complete!")
    print(f"Generated {len(generated_worlds)} levels in ./levels/ directory")
    
    # List generated files
    if os.path.exists("./levels"):
        files = [f for f in os.listdir("./levels") if f.endswith('.yaml')]
        print(f"Level files: {files}")
    
    return generated_worlds

if __name__ == "__main__":
    generate_levels()