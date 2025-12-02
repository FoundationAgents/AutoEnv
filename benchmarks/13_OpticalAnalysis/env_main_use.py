import sys
sys.path.append('/root/yiran/AutoEnvV1.1')

from env_main import LightSpectrumEnv
import argparse
import random

def generate_levels(num_levels=5, seed_start=42):
    env = LightSpectrumEnv(env_id=1)
    env._dsl_config()
    
    print(f"Generating {num_levels} levels...")
    
    for i in range(num_levels):
        seed = seed_start + i
        print(f"Generating level {i+1} with seed {seed}...")
        
        world_id = env._generate_world(seed=seed)
        print(f"Generated world: {world_id}")
        
        env.reset(mode="load", world_id=world_id)
        print(f"Level {i+1} successfully created and validated")
        
        obs = env.observe_semantic()
        print(f"Target material ID: {env._state['sample']['true_material_id']}")
        print("---")
    
    print(f"All {num_levels} levels generated successfully!")

def test_level(world_id):
    env = LightSpectrumEnv(env_id=1)
    env._dsl_config()
    
    print(f"Testing level: {world_id}")
    env.reset(mode="load", world_id=world_id)
    
    print("Initial state:")
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print(rendered)
    
    print("\nTesting EmitUV action:")
    action = {"action": "EmitUV", "params": {}}
    state, reward, done, info = env.step(action)
    
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print(rendered)
    print(f"Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and test Light Spectrum Analysis levels')
    parser.add_argument('--generate', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed', type=int, default=42, help='Starting seed for generation')
    parser.add_argument('--test', type=str, help='Test specific world_id')
    
    args = parser.parse_args()
    
    if args.test:
        test_level(args.test)
    else:
        generate_levels(args.generate, args.seed)
