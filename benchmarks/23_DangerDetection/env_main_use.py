from env_main import InvertedTreasureEnv
import sys
import yaml
import os

def generate_levels():
    # Create config.yaml if it doesn't exist
    config = {
        "meta": {
            "id": "inverted_treasure_hunt",
            "name": "Inverted-Symbol Treasure Hunt Grid",
            "description": "Navigate 8x8 grid to find inverted-symbol treasure while avoiding deadly flower traps"
        },
        "state_template": {
            "globals": {"grid_size": [8, 8], "max_steps": 30},
            "agent": {"pos": [0, 0]},
            "tiles": {
                "size": [8, 8],
                "default_type": "unrevealed",
                "icons": {"bomb_count": 1, "flower_count": 10, "empty_count": 53}
            },
            "grid": {"revealed": {}, "icons": {}}
        },
        "termination": {"max_steps": 30},
        "reward": {"events": [{"trigger": "treasure_found", "value": 1.0}]},
        "generator": {"mode": "procedural", "output_format": "yaml"}
    }
    
    os.makedirs("./levels", exist_ok=True)
    
    if not os.path.exists("./config.yaml"):
        with open("./config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Create environment instance
    env = InvertedTreasureEnv(env_id=1)
    
    # Generate 10 levels with different seeds
    seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
    
    print("Generating levels for Inverted-Symbol Treasure Hunt...")
    
    for i, seed in enumerate(seeds):
        world_id = env._generate_world(seed)
        print(f"Generated level {i+1}: {world_id} (seed: {seed})")
    
    print(f"Successfully generated {len(seeds)} levels in ./levels/ directory")
    
    # Test loading and running one level
    print("\nTesting level generation and loading...")
    initial_obs = env.reset(mode="load", world_id="world_42")
    print("Successfully loaded world_42")
    
    # Take a few actions to test the environment
    print("\nTesting environment functionality...")
    print("Initial state:")
    obs_dict = env.observe_semantic()
    rendered = env.render_skin(obs_dict)
    print(rendered)
    
    # Test reveal action
    action = {"action": "REVEAL", "params": {}}
    state, reward, done, info = env.step(action)
    print(f"\nAfter REVEAL: reward={reward}, done={done}")
    print(f"Action result: {info.get('last_action_result', 'None')}")
    
    # Test movement
    action = {"action": "MOVE_EAST", "params": {}}
    state, reward, done, info = env.step(action)
    print(f"\nAfter MOVE_EAST: reward={reward}, done={done}")
    print(f"Action result: {info.get('last_action_result', 'None')}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    generate_levels()