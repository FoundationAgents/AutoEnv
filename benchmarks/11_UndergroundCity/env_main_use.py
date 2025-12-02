#!/usr/bin/env python3

from env_main import SubterraneanMegacityEnv
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Generate levels for Subterranean Megacity Environment')
    parser.add_argument('--count', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed', type=int, default=None, help='Base seed for generation')
    parser.add_argument('--test', action='store_true', help='Test the environment after generation')
    
    args = parser.parse_args()
    
    env = SubterraneanMegacityEnv(env_id=1)
    
    print(f"Generating {args.count} levels...")
    
    generated_worlds = []
    
    for i in range(args.count):
        # Use incremental seeds if base seed provided
        seed = args.seed + i if args.seed is not None else None
        
        print(f"Generating level {i+1}/{args.count}...")
        
        try:
            world_id = env._generate_world(seed)
            generated_worlds.append(world_id)
            print(f"  Generated: {world_id}")
            
        except Exception as e:
            print(f"  Error generating level {i+1}: {e}")
    
    print(f"\nSuccessfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    if args.test and generated_worlds:
        print(f"\nTesting environment with level: {generated_worlds[0]}")
        try:
            obs = env.reset(mode="load", world_id=generated_worlds[0])
            print("Environment reset successful!")
            print("Initial observation keys:", list(obs.keys()))
            
            # Test a few actions
            test_actions = [
                {"action": "diagnostic_scan", "params": {"x": 2, "y": 2}},
                {"action": "excavate_cell", "params": {"x": 1, "y": 1}},
                {"action": "research_anomaly", "params": {"research_type": "gravity_mechanics"}}
            ]
            
            for i, action in enumerate(test_actions):
                print(f"\nTesting action {i+1}: {action['action']}")
                try:
                    state, reward, done, info = env.step(action)
                    print(f"  Reward: {reward:.2f}")
                    print(f"  Done: {done}")
                    print(f"  Events: {info.get('events', [])}")
                    
                    if done:
                        print("  Environment terminated!")
                        break
                        
                except Exception as e:
                    print(f"  Error: {e}")
            
            print("\nRendering final state:")
            final_obs = env.observe_semantic()
            rendered = env.render_skin(final_obs)
            print(rendered)
            
        except Exception as e:
            print(f"Error testing environment: {e}")

if __name__ == "__main__":
    main()