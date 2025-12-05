#!/usr/bin/env python3
import sys
import os

# Add the correct paths
sys.path.insert(0, '/Users/ruarua/HKUSTGZ/deepwisdom/AutoEnv')
sys.path.insert(0, '.')

try:
    from env_main import AlienColonyEnv
    from env_validator import validate_generated_level
    
    # Test environment with fixed config
    env = AlienColonyEnv(env_id=2)
    
    # Generate a new level with fixed rewards
    obs = env.reset(mode="generate", seed=123)
    print("✓ New level generated with fixed reward structure")
    
    # Find the newest level file
    import os
    levels_dir = "./levels/"
    world_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    newest_world = max(world_files, key=lambda x: os.path.getctime(os.path.join(levels_dir, x)))
    newest_path = os.path.join(levels_dir, newest_world)
    
    print(f"Validating new level: {newest_world}")
    
    # Validate the new level
    is_valid, issues = validate_generated_level(newest_path)
    
    print(f"✓ Level validation completed")
    print(f"Valid: {is_valid}")
    
    if issues:
        print("\nRemaining issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ No validation issues found - level is fully valid!")
    
    # Test a complete gameplay sequence to verify functionality
    print(f"\nTesting complete gameplay sequence...")
    
    # Reset with the new level
    world_id = newest_world.replace('.yaml', '')
    obs = env.reset(mode="load", world_id=world_id)
    
    # Run a strategic sequence to achieve the goal
    actions = [
        {"action": "gather_resource", "params": {"resource_type": "toxic_waste", "amount": 5}},
        {"action": "allocate_resource", "params": {"resource_type": "toxic_waste", "allocation_amount": 4, "target_system": "colony"}},
        {"action": "explore_area", "params": {"direction": "north", "investment_level": 3}},
        {"action": "gather_resource", "params": {"resource_type": "rotten_food", "amount": 8}},
        {"action": "allocate_resource", "params": {"resource_type": "rotten_food", "allocation_amount": 6, "target_system": "colony"}},
        {"action": "explore_area", "params": {"direction": "east", "investment_level": 2}},
    ]
    
    total_reward = 0
    for i, action in enumerate(actions):
        env.transition(action)
        reward, events, reward_info = env.reward(action)
        total_reward += reward
        env._t += 1
        
        obs_dict = env.observe_semantic()
        print(f"Step {i+1}: Pop={obs_dict['population']}, Happiness={obs_dict['happiness']}, Reward={reward:.3f}")
        
        if reward > 0:
            print(f"  Events: {events}")
    
    print(f"\nTotal reward accumulated: {total_reward:.3f}")
    print("✓ Complete gameplay sequence tested successfully!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)