#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from env_main import HiveMindEnv
import argparse

def generate_level(seed=None, level_name=None):
    env = HiveMindEnv()
    
    if level_name is None:
        world_id = env._generate_world(seed)
    else:
        world_id = level_name
        env.generator.generate(seed, f"./levels/{world_id}.yaml")
    
    print(f"Generated level: {world_id}")
    return world_id

def test_level(world_id):
    env = HiveMindEnv()
    
    try:
        state = env.reset(mode="load", world_id=world_id)
        obs = env.observe_semantic()
        
        print(f"Successfully loaded level: {world_id}")
        print(f"Initial Unity: {obs['unity']:.1f}%")
        print(f"Initial Diversity: {obs['diversity']:.1f}%") 
        print(f"Initial Knowledge: {obs['knowledge_score']}")
        print(f"Sub-streams: {len(obs['sub_streams'])}")
        
        test_action = {
            'action': 'STIMULATE',
            'params': {'stream_id': 0, 'energy_amount': 20}
        }
        
        next_state = env.transition(test_action)
        reward, events, reward_info = env.reward(test_action)
        
        print(f"Test action reward: {reward:.2f}")
        print(f"Events: {events}")
        
        return True
        
    except Exception as e:
        print(f"Error testing level {world_id}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and test Hive Mind levels")
    parser.add_argument("--generate", action="store_true", help="Generate a new level")
    parser.add_argument("--test", type=str, help="Test a specific level by world_id")
    parser.add_argument("--seed", type=int, help="Seed for level generation")
    parser.add_argument("--name", type=str, help="Custom name for generated level")
    
    args = parser.parse_args()
    
    if args.generate:
        world_id = generate_level(args.seed, args.name)
        if args.test is None:
            test_level(world_id)
    elif args.test:
        test_level(args.test)
    else:
        world_id = generate_level()
        test_level(world_id)