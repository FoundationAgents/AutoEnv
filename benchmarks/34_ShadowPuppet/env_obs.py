from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class ShadowObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        
        obs['grid_size'] = env_state['globals']['grid_size']
        obs['light_source'] = env_state['globals']['light_source']
        obs['goal_area'] = env_state['globals']['goal_area']
        obs['max_steps'] = env_state['globals']['max_steps']
        obs['t'] = t + 1
        
        obs['shadow_position'] = env_state['shadow']['position']
        obs['shadow_shape'] = env_state['shadow']['shape']
        obs['shadow_active'] = env_state['shadow']['active']
        
        obs['objects'] = []
        for obj in env_state['objects']:
            obj_info = {
                'id': obj['id'],
                'position': obj['position'],
                'property': obj['property'],
                'velocity': obj['velocity'],
                'is_target': obj['is_target']
            }
            obs['objects'].append(obj_info)
        
        return obs
