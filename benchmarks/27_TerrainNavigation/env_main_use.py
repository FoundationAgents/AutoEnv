from env_main import IceLakeEnv
import random

def main():
    env = IceLakeEnv(env_id=83)
    
    print("Generating 5 levels...")
    for i in range(5):
        seed = random.randint(1000, 9999)
        world_id = env._generate_world(seed)
        print(f"Generated level {i+1}: {world_id}")
    
    print("\nTesting environment with generated level...")
    env.reset(mode="generate", seed=12345)
    
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    
    print("\nInitial state:")
    print(f"Position: {rendered['position']}")
    print(f"Steps remaining: {rendered['steps_remaining']}")
    print("Local view:")
    print(rendered['local_grid'])
    print(rendered['legend'])
    
    actions = ["MoveEast", "MoveNorth", "MoveSouth", "Wait"]
    
    for step in range(5):
        action = random.choice(actions)
        action_dict = {"action": action, "params": {}}
        
        next_state, reward, done, info = env.step(action_dict)
        
        print(f"\nStep {step + 1}: {action}")
        print(f"Reward: {reward}")
        print(f"Position: {info['skinned']['position']}")
        print(f"Steps remaining: {info['skinned']['steps_remaining']}")
        print("Local view:")
        print(info['skinned']['local_grid'])
        
        if done:
            print("Episode ended!")
            break

if __name__ == "__main__":
    main()