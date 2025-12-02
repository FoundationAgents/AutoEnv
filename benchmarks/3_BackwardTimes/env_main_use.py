import yaml
import os
from env_main import BackwardsInvestigationEnv

# Create config file
config = {
    'meta': {
        'id': 'backwards_investigation',
        'name': 'BackwardsTimeInvestigation',
        'description': 'Detective environment where agents investigate backwards through time to find the original crime trigger'
    },
    'state_template': {
        'globals': {'max_steps': 40},
        'agent': {'current_time_index': 40, 'investigation_status': 'active'},
        'timeline': {'time_indices': 40, 'events': {}, 'validated_connections': {}},
        'crime_scene': {'location': 'primary_scene', 'initial_evidence': [], 'witness_statements': [], 'suspect_list': []},
        'investigation': {'collected_clues': [], 'timeline_ledger': {}, 'unresolved_effects': [], 'proposed_connections': []},
        'ground_truth': {'root_cause_time': 0, 'root_cause_perpetrator': '', 'root_cause_action': '', 'essential_clues': [], 'decoy_suspects': []}
    },
    'generator': {
        'mode': 'procedural',
        'output_format': 'yaml',
        'pipeline': [
            {'name': 'init_from_template', 'desc': 'Initialize world with state_template as base', 'args': {}},
            {'name': 'generate_crime_scenario', 'desc': 'Create the root cause event with perpetrator and action at random early time', 'args': {'root_time_range': [5, 15], 'perpetrator_pool_size': [4, 6]}},
            {'name': 'build_causal_chain', 'desc': 'Create sequence of events leading from root cause to final crime scene', 'args': {'chain_length': [6, 8], 'complexity_factor': 0.7}},
            {'name': 'distribute_clues', 'desc': 'Place essential clues throughout timeline and add decoy evidence', 'args': {'essential_clues': [6, 8], 'decoy_clues': [2, 3]}},
            {'name': 'setup_crime_scene', 'desc': 'Initialize final state at time 40 with visible effects and initial information', 'args': {'initial_suspects': [4, 6], 'unresolved_effects': [2, 3]}},
            {'name': 'validate_solvability', 'desc': 'Ensure the generated scenario has exactly one valid solution path', 'args': {'require_unique_solution': True}}
        ]
    },
    'termination': {'max_steps': 40}
}

# Save config
with open('./config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Create levels directory
os.makedirs('./levels/', exist_ok=True)

# Generate and test levels
if __name__ == "__main__":
    env = BackwardsInvestigationEnv(env_id=1)
    
    print("Generating 3 test levels...")
    
    for i in range(3):
        print(f"\n=== Generating Level {i+1} ===")
        
        # Generate new level
        state = env.reset(mode="generate", seed=42 + i)
        
        # Get initial observation
        obs = env.observe_semantic()
        rendered = env.render_skin(obs)
        print(f"Level {i+1} generated successfully!")
        print("Initial state:")
        print(rendered[:200] + "..." if len(rendered) > 200 else rendered)
        
        # Test a few actions
        print(f"\nTesting actions on Level {i+1}:")
        
        # Test ExamineScene
        action = {"action": "ExamineScene", "params": {"location": "office"}}
        next_state, reward, done, info = env.step(action)
        print(f"ExamineScene result: {info.get('last_action_result', 'No result')}")
        
        # Test InterrogatePerson  
        action = {"action": "InterrogatePerson", "params": {"person": "person_1"}}
        next_state, reward, done, info = env.step(action)
        print(f"InterrogatePerson result: {info.get('last_action_result', 'No result')}")
        
        print(f"Level {i+1} testing completed. Done: {done}, Reward: {reward}")
    
    print("\n=== Level Generation Complete ===")
    print("All levels saved to ./levels/ directory")