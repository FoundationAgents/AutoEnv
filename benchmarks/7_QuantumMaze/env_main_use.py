#!/usr/bin/env python3
import os
import sys
import yaml
from env_main import QuantumMazeEnv

def generate_levels(num_levels=5):
    """Generate multiple levels for the Quantum Maze environment"""
    
    # Create levels directory if it doesn't exist
    os.makedirs('./levels', exist_ok=True)
    
    # Initialize environment
    env = QuantumMazeEnv(env_id=1)
    
    generated_levels = []
    
    for i in range(num_levels):
        print(f"Generating level {i+1}/{num_levels}...")
        
        # Generate a new level with different seeds
        seed = 1000 + i
        obs = env.reset(mode="generate", seed=seed)
        
        # Get the world ID from the generated level
        world_id = f"quantum_maze_seed_{seed}"
        
        # Test the level by taking a few actions
        print(f"Testing level {world_id}...")
        print(env.render_skin(obs))
        
        # Try a few moves to test functionality
        test_actions = [
            {"action": "OBSERVE", "params": {}},
            {"action": "MOVE_EAST", "params": {}},
            {"action": "MOVE_SOUTH", "params": {}}
        ]
        
        for j, action in enumerate(test_actions):
            if not env.done():
                state, reward, done, info = env.step(action)
                print(f"\nAction {j+1}: {action['action']}")
                print(f"Reward: {reward}, Done: {done}")
                if done:
                    print("Level completed or terminated!")
                    break
        
        generated_levels.append(world_id)
        print(f"Level {world_id} generated successfully!\n" + "="*50)
    
    print(f"\nGenerated {len(generated_levels)} levels:")
    for level in generated_levels:
        print(f"  - {level}")
    
    print(f"\nAll levels saved in ./levels/ directory")
    print("You can load any level using: env.reset(mode='load', world_id='LEVEL_NAME')")

if __name__ == "__main__":
    # Allow specifying number of levels as command line argument
    num_levels = 5
    if len(sys.argv) > 1:
        try:
            num_levels = int(sys.argv[1])
        except ValueError:
            print("Usage: python env_main_code_use.py [num_levels]")
            sys.exit(1)
    
    generate_levels(num_levels)