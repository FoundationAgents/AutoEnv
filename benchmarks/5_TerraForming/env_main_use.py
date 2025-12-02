import os
import sys
from env_main import TerraformingEnv

def generate_levels():
    env = TerraformingEnv(1)
    
    os.makedirs("./levels", exist_ok=True)
    
    print("Generating 10 terraforming planets...")
    
    for i in range(10):
        try:
            world_id = env._generate_world(seed=i)
            print(f"Generated planet {i+1}: {world_id}")
        except Exception as e:
            print(f"Error generating planet {i+1}: {e}")
    
    print("Level generation complete!")
    
    print("\nTesting level loading...")
    try:
        obs = env.reset(mode="load", world_id="planet_0")
        print("Successfully loaded planet_0")
        print(f"Initial habitability: {obs['habitability_index']:.1f}%")
        print(f"Initial instability: {obs['instability_index']:.1f}%")
        
        action = {"action": "PASSIVE_OBSERVATION", "params": {}}
        state, reward, done, info = env.step(action)
        print(f"Step reward: {reward:.2f}")
        print(f"Episode done: {done}")
        
    except Exception as e:
        print(f"Error testing level: {e}")

if __name__ == "__main__":
    generate_levels()