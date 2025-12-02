from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class SpectrumObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        
        obs['current_step'] = env_state['agent']['current_step']
        obs['max_steps'] = env_state['globals']['max_steps']
        obs['remaining_steps'] = env_state['globals']['max_steps'] - env_state['agent']['current_step']
        obs['illuminated_bands'] = deepcopy(env_state['sample']['illuminated_bands'])
        obs['observed_spectrum'] = deepcopy(env_state['sample']['observed_spectrum'])
        obs['reference_library'] = deepcopy(env_state['reference_library']['material_signatures'])
        obs['t'] = t + 1
        
        return obs