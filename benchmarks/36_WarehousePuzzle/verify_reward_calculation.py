import yaml
from env_main import BinaryWarehouseEnv

def test_reward_structure():
    """Quick test to verify our understanding of the reward structure."""
    print("Testing reward structure with level_01...")
    
    # Initialize environment
    env = BinaryWarehouseEnv()
    
    # Load level_01
    state = env.reset(mode="load", world_id="level_01")
    
    print(f"Initial state:")
    print(f"  Agent position: {state['agent']['pos']}")
    print(f"  Total boxes: {state['level_info']['total_boxes']}")
    print(f"  Boxes on docks: {state['level_info']['boxes_on_docks']}")
    
    # Check initial reward
    reward, events, info = env.reward({})
    print(f"  Initial reward: {reward} (should be 0.0)")
    
    # Check what happens when we complete the level
    # Manually set all boxes on docks to test completion reward
    dock_positions = [dock['pos'] for dock in state['objects']['docks']]
    for i, box in enumerate(state['objects']['boxes']):
        if i < len(dock_positions):
            box['pos'] = dock_positions[i]
    
    env._state = state
    env._update_boxes_on_docks()
    
    reward, events, info = env.reward({})
    print(f"  Completion reward: {reward} (should be 1.0)")
    print(f"  Events: {events}")
    print(f"  Info: {info}")
    
    # Verify termination condition
    is_done = env.done()
    print(f"  Game done: {is_done} (should be True)")
    
    print("Reward structure verification complete!")

if __name__ == "__main__":
    test_reward_structure()