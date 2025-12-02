#!/usr/bin/env python3

import yaml
import os
from env_main import GearRatioEnv

def create_config():
    config = {
        "meta": {
            "id": "gear_ratio_opt",
            "name": "Gear Ratio Optimization Environment",
            "description": "Design linear gear trains to achieve target mechanical advantages within tolerance"
        },
        "state_template": {
            "globals": {
                "max_steps": 30,
                "tolerance": 0.02
            },
            "agent": {
                "remaining_steps": 30
            },
            "gear_system": {
                "available_gears": [],
                "gear_chain": [],
                "current_ma": 1.0,
                "target_ma": 1.0,
                "episode_finished": False,
                "success": False
            }
        },
        "observation": {
            "policy": "full",
            "params": {},
            "expose": [
                "gear_system.available_gears",
                "gear_system.gear_chain", 
                "gear_system.current_ma",
                "gear_system.target_ma",
                "agent.remaining_steps",
                "globals.tolerance",
                "t"
            ]
        },
        "reward": {
            "events": [
                {"trigger": "finish_success", "value": 1.0},
                {"trigger": "finish_fail", "value": 0.0},
                {"trigger": "step", "value": 0.0}
            ]
        },
        "transition": {
            "actions": [
                {"name": "PlaceGear", "params": ["gear_index"]},
                {"name": "RemoveLast", "params": []},
                {"name": "Finish", "params": []},
                {"name": "Skip", "params": []}
            ]
        },
        "termination": {
            "max_steps": 30,
            "conditions": ["gear_system.episode_finished"]
        },
        "generator": {
            "mode": "procedural",
            "output_format": "yaml",
            "pipeline": [
                {"name": "init_from_template", "desc": "Initialize world with state_template as base", "args": {}},
                {"name": "generate_gear_library", "desc": "Generate 10 random gears with tooth counts between 6-60", "args": {"num_gears": 10, "min_teeth": 6, "max_teeth": 60}},
                {"name": "generate_target_ma", "desc": "Generate random target mechanical advantage that is achievable", "args": {"min_ratio": 0.1, "max_ratio": 10.0}},
                {"name": "validate_solvability", "desc": "Ensure target MA is theoretically achievable with given gears", "args": {"max_chain_length": 10}}
            ],
            "randomization": {
                "seed_based": True,
                "parameters": {
                    "gear_teeth_range": [6, 60],
                    "target_ma_range": [0.1, 10.0]
                }
            }
        }
    }
    return config

def generate_levels():
    os.makedirs("./levels", exist_ok=True)
    
    config = create_config()
    with open("./config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    env = GearRatioEnv(env_id=77)
    
    print("Generating levels...")
    for i in range(5):
        seed = 1000 + i
        print(f"Generating level with seed {seed}...")
        
        obs = env.reset(mode="generate", seed=seed)
        rendered = env.render_skin(obs)
        print(f"\nLevel {i+1} generated:")
        print("=" * 50)
        print(rendered)
        print("=" * 50)
        
        test_action = {"action": "PlaceGear", "params": {"gear_index": 0}}
        next_state, reward, done, info = env.step(test_action)
        print(f"Test action result: {info['last_action_result']}")
        print(f"New MA: {next_state['current_ma']:.4f}")
        print()

if __name__ == "__main__":
    generate_levels()