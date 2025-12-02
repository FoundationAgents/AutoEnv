#!/usr/bin/env python3

from env_main import SentientArchitectureEnv
import os
import sys
import random

def main():
    # Create environment
    env = SentientArchitectureEnv(env_id=15)
    
    # Ensure levels directory exists
    os.makedirs("./levels", exist_ok=True)
    
    # Generate 5 different levels
    print("Generating levels for Sentient Architecture Management Environment...")
    
    for i in range(5):
        seed = random.randint(1, 10000)
        print(f"\nGenerating level {i+1} with seed {seed}...")
        
        try:
            # Reset in generate mode to create new level
            env.reset(mode="generate", seed=seed)
            print(f"Successfully generated level {i+1}")
            
            # Test the level by running a few steps
            obs = env.observe_semantic()
            print(f"Level {i+1} initial state:")
            print(f"  - Buildings: {len(obs['buildings'])}")
            print(f"  - Bio-materials: {obs['city']['bio_material_stock']}")
            print(f"  - Energy capacity: {obs['city']['energy_grid_capacity']}")
            print(f"  - Conflicts: {len(obs['conflicts'])}")
            
            # Test a sample action
            test_action = {
                "action": "Negotiate", 
                "params": {"building_id": obs['buildings'][0]['building_id']}
            }
            env.step(test_action)
            print(f"  - Test action completed successfully")
            
        except Exception as e:
            print(f"Error generating level {i+1}: {str(e)}")
    
    print(f"\nLevel generation complete. Check ./levels/ directory for generated world files.")
    
    # List generated files
    if os.path.exists("./levels"):
        level_files = [f for f in os.listdir("./levels") if f.endswith('.yaml')]
        print(f"Generated {len(level_files)} level files:")
        for file in sorted(level_files):
            print(f"  - {file}")

if __name__ == "__main__":
    main()