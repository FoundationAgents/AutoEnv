from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class ArchitectureObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {
            'city': {
                'bio_material_stock': env_state['city']['bio_material_stock'],
                'energy_grid_capacity': env_state['city']['energy_grid_capacity'],
                'harmony_index': env_state['city']['harmony_index'],
                'synergy_score': env_state['city']['synergy_score']
            },
            'buildings': deepcopy(env_state['buildings']),
            'conflicts': deepcopy(env_state['conflicts']),
            'max_steps': env_state['globals']['max_steps'],
            't': t,
            'target_synergy': env_state['globals']['target_synergy']
        }
        return obs