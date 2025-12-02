import sys
import os
sys.path.append('.')

from env_main import MolecularTasteEnv

def generate_levels(num_levels=5):
    """Generate concrete levels for the Molecular Taste Navigation environment"""
    print("Generating Molecular Taste Navigation levels...")
    
    # Create environment instance
    env = MolecularTasteEnv("molecular_taste_nav")
    
    generated_worlds = []
    
    for i in range(num_levels):
        seed = 1000 + i  # Use deterministic seeds
        print(f"Generating level {i+1}/{num_levels} with seed {seed}...")
        
        try:
            # Generate and load new world
            world_id = env._generate_world(seed)
            generated_worlds.append(world_id)
            print(f"✓ Generated world: {world_id}")
            
            # Test the generated world
            env.reset(mode="load", world_id=world_id)
            obs = env.observe_semantic()
            print(f"  - Start position: {obs['agent_pos']}")
            print(f"  - Chemical signature: {[f'{x:.3f}' for x in obs['flavor_vector']]}")
            
        except Exception as e:
            print(f"✗ Error generating level {i+1}: {e}")
    
    print(f"\nGeneration complete! Created {len(generated_worlds)} levels:")
    for world_id in generated_worlds:
        print(f"  - {world_id}")
    
    return generated_worlds

def test_environment():
    """Test the environment with a sample level"""
    print("\nTesting environment...")
    
    env = MolecularTasteEnv("molecular_taste_nav")
    
    # Generate a test world
    world_id = env._generate_world(seed=42)
    
    # Reset and test basic functionality
    env.reset(mode="load", world_id=world_id)
    
    # Test a few actions
    actions = [
        {"action": "DO_NOTHING", "params": {}},
        {"action": "MOVE_NORTH", "params": {}},
        {"action": "MOVE_EAST", "params": {}},
        {"action": "MOVE_SOUTH", "params": {}},
        {"action": "MOVE_WEST", "params": {}}
    ]
    
    for i, action in enumerate(actions):
        state, reward, done, info = env.step(action)
        print(f"Step {i+1}: {action['action']} -> Reward: {reward}, Done: {done}")
        if done:
            break
    
    print("✓ Environment test complete!")

if __name__ == "__main__":
    # Generate levels
    generated_worlds = generate_levels(5)
    
    # Test environment
    test_environment()
    
    print("\n" + "="*50)
    print("Setup complete! You can now use the environment.")
    print("Generated worlds are saved in ./levels/")
    print("="*50)