#!/usr/bin/env python3
import os
import sys
from env_main import BinaryWarehouseEnv

def generate_levels(num_levels=10, start_seed=42):
    """Generate warehouse levels and test basic functionality"""
    
    print(f"Generating {num_levels} warehouse levels...")
    
    env = BinaryWarehouseEnv()
    
    generated_levels = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"\nGenerating level {i+1}/{num_levels} with seed {seed}")
        
        try:
            state = env.reset(mode="generate", seed=seed)
            
            obs = env.observe_semantic()
            rendered = env.render_skin(obs)
            
            print(f"Successfully generated level with {obs['total_boxes']} boxes")
            print(f"Agent starts at position: {obs['agent_pos']}")
            
            world_id = f"warehouse_{seed}"
            generated_levels.append(world_id)
            
            if i < 3:  # Show first 3 levels
                print("\nLevel preview:")
                print(rendered)
            
        except Exception as e:
            print(f"Error generating level {i+1}: {e}")
    
    print(f"\n=== Generation Complete ===")
    print(f"Successfully generated {len(generated_levels)} levels")
    print(f"Levels saved to: ./levels/")
    
    print("\n=== Testing a generated level ===")
    if generated_levels:
        try:
            test_world_id = generated_levels[0].replace("warehouse_", "") + "_" + "test"
            env.reset(mode="generate", seed=42)
            
            obs = env.observe_semantic()
            print(f"Test level loaded with {obs['total_boxes']} boxes")
            
            for action_name in ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest"]:
                action = {"action": action_name, "params": {}}
                
                old_pos = obs['agent_pos'][:]
                state, reward, done, info = env.step(action)
                obs = env.observe_semantic()
                
                print(f"Action: {action_name}")
                print(f"  Position: {old_pos} -> {obs['agent_pos']}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")
                print(f"  Action result: {info.get('last_action_result', 'None')}")
                
                if done:
                    print("Level completed or maximum steps reached!")
                    break
                    
        except Exception as e:
            print(f"Error testing level: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_levels = int(sys.argv[1])
    else:
        num_levels = 10
    
    if len(sys.argv) > 2:
        start_seed = int(sys.argv[2])
    else:
        start_seed = 42
    
    generate_levels(num_levels, start_seed)