#!/usr/bin/env python3

import os
import sys
sys.path.append('.')

from env_main import BackwardsValleyFarmEnv
from env_obs import RadiusObserver
from env_generate import BackwardsValleyFarmGenerator
import yaml

def create_config_file():
    config = {
        "meta": {
            "id": "backwards_valley_farm_v0",
            "name": "BackwardsValleyFarm",
            "description": "Inverse-causality 10×10 farm where neglect is beneficial."
        },
        "state_template": {
            "globals": {
                "max_steps": 40,
                "farm_value": 0
            },
            "agent": {
                "pos": [0, 0],
                "facing": "N"
            },
            "tiles": {
                "size": [10, 10],
                "default_type": "grass"
            },
            "objects": {
                "fields": [],
                "pens": [],
                "villagers": [],
                "fences": []
            }
        },
        "generator": {
            "mode": "procedural",
            "output_format": "yaml",
            "pipeline": [
                {"name": "init_from_template", "desc": "Deep-copy state_template as base.", "args": {}},
                {"name": "place_zones", "desc": "Partition map into crop, pen, village zones; add fences.", "args": {"num_fields": 12, "num_pens": 4, "num_houses": 6}},
                {"name": "populate_entities", "desc": "Instantiate crops, animals, villagers into zones.", "args": {"crop_types": ["wheat", "corn", "rice", "pumpkin"], "animal_types": ["cow", "sheep", "pig"]}},
                {"name": "assign_initial_states", "desc": "Randomise growth/health/mood categories per spec.", "args": {"crop_stages": ["Seed", "Sprout", "Young", "HarvestReady"], "animal_states": ["Weak", "Okay", "Thriving"], "moods": ["Hostile", "Neutral", "Friendly"]}},
                {"name": "place_agent", "desc": "Pick random empty tile for agent spawn.", "args": {}}
            ]
        },
        "termination": {
            "max_steps": 40,
            "conditions": ["globals.farm_value >= 300"]
        },
        "skin": {
            "type": "text",
            "template": "Step {t}/{max_steps}   Farm Value: {farm_value}\nPosition: {agent_pos}  Facing: {agent_facing}\nRemaining steps: {remaining}\nVisible (5×5, A=Agent, C=Crop, P=Pen, V=Villager, #=Fence, .=Grass):\n{tiles_ascii}\nActions: MoveN/S/E/W, Wait, UseWateringCan, SpreadFertilizer, Feed, CleanPen, Compliment, Insult"
        }
    }
    
    with open('./config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Config file created: ./config.yaml")

def generate_levels():
    os.makedirs('./levels', exist_ok=True)
    
    env = BackwardsValleyFarmEnv("backwards_valley_farm")
    
    print("Generating sample levels...")
    
    for i in range(5):
        seed = 1000 + i
        world_id = env._generate_world(seed)
        print(f"Generated level {i+1}: {world_id}")
    
    print("\nLevel generation complete!")
    print("Generated levels are saved in ./levels/")

def test_environment():
    print("\nTesting environment...")
    
    env = BackwardsValleyFarmEnv("backwards_valley_farm")
    
    state = env.reset(mode="generate", seed=42)
    print("Environment reset successful")
    
    print("\nInitial observation:")
    obs = env.observe_semantic()
    rendered = env.render_skin(obs)
    print(rendered)
    
    print("\nTesting a few actions...")
    actions = [
        {"action": "MoveE", "params": {}},
        {"action": "Wait", "params": {}},
        {"action": "UseWateringCan", "params": {}}
    ]
    
    for i, action in enumerate(actions):
        print(f"\nAction {i+1}: {action['action']}")
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(f"Farm Value: {state['globals']['farm_value']}")
        
        if done:
            print("Episode terminated")
            break
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    print("Backwards Valley Farm Environment Setup")
    print("=" * 50)
    
    create_config_file()
    generate_levels()
    test_environment()
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now use the environment.")
    print("\nTo use the environment in your code:")
    print("from env_main import BackwardsValleyFarmEnv")
    print("env = BackwardsValleyFarmEnv('backwards_valley_farm')")
    print("state = env.reset(mode='generate', seed=42)")