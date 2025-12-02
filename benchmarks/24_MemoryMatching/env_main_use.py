import sys
import os

# Add autoenv path for imports
sys.path.insert(0, '../../../')
sys.path.insert(0, '.')

from env_main import ChaosSlideEnv
import argparse

def generate_levels(num_levels=10, start_seed=42):
    """Generate multiple levels for the chaos slide puzzle environment."""
    
    # Create environment instance
    env = ChaosSlideEnv(env_id=1)
    env._dsl_config()
    
    # Ensure levels directory exists
    os.makedirs('./levels/', exist_ok=True)
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = start_seed + i
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        try:
            # Generate new world - this returns the world_id directly
            world_id = env._generate_world(seed)
            obs = env.reset(mode="load", world_id=world_id)
            generated_worlds.append(world_id)
            
            # Display the generated level
            rendered = env.render_skin(obs)
            print(f"\nGenerated Level {i+1}:")
            print("-" * 50)
            print(rendered)
            print("-" * 50)
            
        except Exception as e:
            print(f"Error generating level {i+1}: {e}")
    
    print(f"\nSuccessfully generated {len(generated_worlds)} levels!")
    print("Generated world IDs:", generated_worlds)
    return generated_worlds

def test_level(world_id):
    """Test loading and displaying a specific level."""
    env = ChaosSlideEnv(env_id=1)
    env._dsl_config()
    
    try:
        obs = env.reset(mode="load", world_id=world_id)
        rendered = env.render_skin(obs)
        print(f"Loaded Level: {world_id}")
        print("-" * 50)
        print(rendered)
        print("-" * 50)
        
        # Test a few actions
        print("\nTesting actions:")
        actions = ["SLIDE_UP", "SLIDE_DOWN", "SLIDE_LEFT", "SLIDE_RIGHT"]
        
        for action_name in actions[:2]:  # Test first 2 actions
            action = {"action": action_name, "params": {}}
            
            # Use the proper environment methods
            env.transition(action)
            reward_info = env.reward(action)
            done = env.done()
            obs = env.observe_semantic()
            
            print(f"\nAction: {action_name}")
            print(f"Reward: {reward_info[0]}")
            print(f"Events: {reward_info[1]}")
            print(f"Done: {done}")
            print(f"Action Result: {getattr(env, '_last_action_result', 'N/A')}")
            
            if done:
                print("Episode finished!")
                break
    
    except Exception as e:
        print(f"Error testing level {world_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and test Chaos Slide Puzzle levels")
    parser.add_argument("--generate", type=int, default=5, help="Number of levels to generate")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed for generation")
    parser.add_argument("--test", type=str, help="World ID to test loading")
    
    args = parser.parse_args()
    
    if args.test:
        test_level(args.test)
    else:
        generated_worlds = generate_levels(args.generate, args.seed)
        
        # Test the first generated level
        if generated_worlds:
            print(f"\nTesting first generated level: {generated_worlds[0]}")
            test_level(generated_worlds[0])
