#!/usr/bin/env python3

import yaml
import argparse
import sys
import os

sys.path.append('.')

from env_main import SmartHomeEnv

def generate_levels(num_levels=5, seed_start=1000):
    """Generate multiple levels for the Smart Home Assistant environment"""
    
    config = {
        "meta": {
            "id": "smart_home_assistant",
            "name": "Smart Home Assistant Environment",
            "description": "Embodied AI agent completing household chores in a realistic apartment setting"
        },
        "state_template": {
            "globals": {"max_steps": 40, "grid_size": [12, 12], "num_chores": 3},
            "agent": {"pos": [0, 0], "facing": "north", "inventory": None},
            "apartment": {
                "walls": [], "doors": [],
                "rooms": {
                    "kitchen": {"bounds": [], "furniture": [], "appliances": []},
                    "living_room": {"bounds": [], "furniture": [], "appliances": []},
                    "bedroom": {"bounds": [], "furniture": [], "appliances": []},
                    "bathroom": {"bounds": [], "furniture": [], "appliances": []},
                    "corridor": {"bounds": [], "furniture": [], "appliances": []}
                }
            },
            "objects": [], "appliances": [], "containers": [],
            "chores": {"instructions": [], "completed": [False, False, False]}
        },
        "generator": {
            "mode": "procedural",
            "output_format": "yaml",
            "pipeline": [
                {"name": "init_from_template", "desc": "Initialize world with state_template as base"},
                {"name": "generate_apartment_layout", "desc": "Create 5-room layout with walls and doors"},
                {"name": "place_furniture_appliances", "desc": "Add appropriate furniture and appliances"},
                {"name": "populate_objects", "desc": "Distribute movable objects across rooms"},
                {"name": "generate_chore_instructions", "desc": "Create 3 random chore tasks"},
                {"name": "place_agent", "desc": "Spawn agent at random floor position"}
            ]
        },
        "termination": {"max_steps": 40}
    }
    
    os.makedirs('./levels', exist_ok=True)
    
    with open('./config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    env = SmartHomeEnv()
    generated_worlds = []
    
    print(f"Generating {num_levels} levels...")
    
    for i in range(num_levels):
        seed = seed_start + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        try:
            world_id = env._generate_world(seed)
            generated_worlds.append(world_id)
            print(f"  Generated world: {world_id}")
            
            obs = env.reset(mode="load", world_id=world_id)
            print(f"  Level validated successfully!")
            
        except Exception as e:
            print(f"  Error generating level {i+1}: {e}")
            continue
    
    print(f"\nSuccessfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    print(f"\nLevels saved in: ./levels/")
    
    return generated_worlds

def test_level(world_id):
    """Test a specific level by loading and running a few steps"""
    env = SmartHomeEnv()
    
    try:
        print(f"Testing level: {world_id}")
        obs = env.reset(mode="load", world_id=world_id)
        
        print("Initial observation:")
        print(env.render_skin(obs))
        
        test_actions = [
            {"action": "Wait", "params": {}},
            {"action": "TurnRight", "params": {}},
            {"action": "MoveForward", "params": {}}
        ]
        
        for i, action in enumerate(test_actions):
            print(f"\nStep {i+1}: {action}")
            state, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            print("New observation:")
            print(info["skinned"])
            
            if done:
                print("Episode finished!")
                break
                
    except Exception as e:
        print(f"Error testing level {world_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate levels for Smart Home Assistant Environment')
    parser.add_argument('--num-levels', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed-start', type=int, default=1000, help='Starting seed for generation')
    parser.add_argument('--test-world', type=str, help='Test a specific world by ID')
    
    args = parser.parse_args()
    
    if args.test_world:
        test_level(args.test_world)
    else:
        generated_worlds = generate_levels(args.num_levels, args.seed_start)
        
        if generated_worlds:
            print(f"\nTesting first generated level: {generated_worlds[0]}")
            test_level(generated_worlds[0])

if __name__ == "__main__":
    main()