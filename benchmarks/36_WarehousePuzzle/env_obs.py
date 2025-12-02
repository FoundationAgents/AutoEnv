from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class FullObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        
        obs['agent_pos'] = env_state['agent']['pos']
        obs['tiles'] = env_state['tiles']
        obs['boxes'] = env_state['objects']['boxes']
        obs['docks'] = env_state['objects']['docks']
        obs['total_boxes'] = env_state['level_info']['total_boxes']
        obs['boxes_on_docks'] = env_state['level_info']['boxes_on_docks']
        obs['max_steps'] = env_state['globals']['max_steps']
        obs['t'] = t + 1  # Fix: Show step number starting from 1
        obs['boxes_remaining'] = obs['total_boxes'] - obs['boxes_on_docks']
        
        return obs