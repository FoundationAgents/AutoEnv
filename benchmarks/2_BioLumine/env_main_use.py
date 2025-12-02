#!/usr/bin/env python3

import sys
sys.path.append("../../../")
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_main import BioluminescentEnv
import argparse

def generate_levels(env_id, num_levels=5, seed_start=42):
    env = BioluminescentEnv(env_id)
    
    print(f"Generating {num_levels} levels for environment {env_id}")
    
    for i in range(num_levels):
        seed = seed_start + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}")
        
        try:
            obs = env.reset(mode="generate", seed=seed)
            print(f"  ✓ Successfully generated level with world_id: world_{seed}")
        except Exception as e:
            print(f"  ✗ Failed to generate level {i+1}: {e}")
    
    print("Level generation complete!")

def test_level(env_id, world_id):
    env = BioluminescentEnv(env_id)
    
    print(f"Testing level {world_id} in environment {env_id}")
    
    try:
        obs = env.reset(mode="load", world_id=world_id)
        print(f"✓ Successfully loaded level {world_id}")
        
        rendered = env.render_skin(obs)
        print("Initial state:")
        print(rendered)
        
        sample_action = {
            "action": "RESPOND_PATTERN",
            "params": {
                "pattern_length": 3,
                "pulse_colors": ["blue", "green", "purple"],
                "pulse_durations": ["short", "long", "short"], 
                "pulse_intensities": ["low", "high", "low"]
            }
        }
        
        next_state, reward, done, info = env.step(sample_action)
        print(f"\nAfter sample action:")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Last action result: {info.get('last_action_result', 'None')}")
        
    except Exception as e:
        print(f"✗ Failed to test level {world_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and test bioluminescent environment levels")
    parser.add_argument("--env_id", default="bioluminescent_test", help="Environment ID")
    parser.add_argument("--generate", type=int, default=5, help="Number of levels to generate")
    parser.add_argument("--seed_start", type=int, default=42, help="Starting seed for generation")
    parser.add_argument("--test", type=str, help="World ID to test")
    
    args = parser.parse_args()
    
    if args.test:
        test_level(args.env_id, args.test)
    else:
        generate_levels(args.env_id, args.generate, args.seed_start)