from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class ColonyObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        observation = {}
        
        # Extract colony information
        observation['population'] = env_state['colony']['population']
        observation['happiness'] = env_state['colony']['happiness']
        observation['area_exploration'] = env_state['colony']['area_exploration']
        observation['target_population'] = env_state['globals']['target_population']
        observation['t'] = t + 1
        observation['max_steps'] = env_state['globals']['max_steps']
        
        # Extract resources
        observation['resources'] = env_state['resources'].copy()
        
        # Extract buildings
        observation['buildings'] = []
        for building in env_state['buildings']:
            observation['buildings'].append({
                'type': building['type'],
                'location': building['location'],
                'operational': building['operational']
            })
        
        # Extract environment conditions
        observation['season'] = env_state['environment']['season']
        observation['weather'] = env_state['environment']['weather']
        
        # Extract discovery information
        observation['resource_effects_found'] = env_state['discovery']['resource_effects_found'].copy()
        observation['building_effects_found'] = env_state['discovery']['building_effects_found'].copy()
        
        # Grid information for building placement
        observation['grid_size'] = env_state['grid']['size']
        observation['occupied_positions'] = env_state['grid']['occupied'].copy()
        
        return observation
