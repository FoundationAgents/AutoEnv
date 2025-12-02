from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class UndergroundObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        
        # Basic metrics and counters
        obs['districts_built'] = env_state['agent']['districts_built']
        obs['power_storage'] = env_state['agent']['power_storage']
        obs['available_materials'] = deepcopy(env_state['agent']['available_materials'])
        obs['structural_integrity'] = env_state['metrics']['structural_integrity']
        obs['breathable_air_index'] = env_state['metrics']['breathable_air_index']
        obs['total_power_usage'] = env_state['metrics']['total_power_usage']
        obs['current_airflow_phase'] = env_state['physics_state']['current_airflow_phase']
        obs['max_steps'] = env_state['globals']['max_steps']
        obs['t'] = t
        
        # Research status
        obs['research'] = deepcopy(env_state['research'])
        
        # Grid information - provide full observability of current state
        grid_size = env_state['grid']['size']
        obs['grid_size'] = grid_size
        obs['grid_cells'] = []
        
        for y in range(grid_size[1]):
            row = []
            for x in range(grid_size[0]):
                cell_idx = y * grid_size[0] + x
                cell_info = {
                    'rock_stress': env_state['grid']['cells']['rock_stress'][cell_idx],
                    'airflow_vector': env_state['grid']['cells']['airflow_vector'][cell_idx],
                    'structure_type': env_state['grid']['cells']['structure_type'][cell_idx],
                    'has_support': env_state['grid']['cells']['has_support'][cell_idx],
                    'excavated': env_state['grid']['cells']['excavated'][cell_idx],
                    'district_core': env_state['grid']['cells']['district_core'][cell_idx],
                    'power_conduit': env_state['grid']['cells']['power_conduit'][cell_idx],
                    'ventilation_shaft': env_state['grid']['cells']['ventilation_shaft'][cell_idx]
                }
                row.append(cell_info)
            obs['grid_cells'].append(row)
        
        return obs