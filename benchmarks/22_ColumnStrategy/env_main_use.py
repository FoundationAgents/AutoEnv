from env_main import ConnectFourEnv
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='Generate Connect Four levels')
    parser.add_argument('--num_levels', type=int, default=5, help='Number of levels to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for generation')
    parser.add_argument('--env_id', type=str, default='connect_four_test', help='Environment ID')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    env = ConnectFourEnv(args.env_id)
    
    print(f"Generating {args.num_levels} Connect Four levels...")
    
    generated_worlds = []
    for i in range(args.num_levels):
        level_seed = random.randint(0, 999999) if args.seed is None else args.seed + i
        world_id = env._generate_world(level_seed)
        generated_worlds.append(world_id)
        print(f"Generated level {i+1}: {world_id}")
    
    print(f"\nSuccessfully generated {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    # Test loading and running one level
    print(f"\nTesting level: {generated_worlds[0]}")
    env.reset(mode="load", world_id=generated_worlds[0])
    
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print("Initial state:")
    print(rendered)
    
    # Test a sample action
    action = {"action": "drop_disk", "params": {"column": 3}}
    state = env.transition(action)
    reward, events, reward_info = env.reward(action)
    
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print(f"\nAfter dropping disk in column 3 (reward: {reward}):")
    print(rendered)

if __name__ == "__main__":
    main()