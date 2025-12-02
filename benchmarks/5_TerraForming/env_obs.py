from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class TerraformingObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        observation = {}
        
        observation['oxygen_pct'] = env_state['atmosphere']['oxygen_pct']
        observation['co2_pct'] = env_state['atmosphere']['co2_pct']
        observation['pressure'] = env_state['atmosphere']['pressure']
        observation['temperature'] = env_state['atmosphere']['temperature']
        
        observation['surface_water_pct'] = env_state['hydrosphere']['surface_water_pct']
        observation['subsurface_ice_pct'] = env_state['hydrosphere']['subsurface_ice_pct']
        observation['ph_level'] = env_state['hydrosphere']['ph_level']
        
        observation['soil_fertility'] = env_state['lithosphere']['soil_fertility']
        observation['tectonic_stress'] = env_state['lithosphere']['tectonic_stress']
        
        observation['dormant_microbes'] = env_state['biosphere_seeds']['dormant_microbes']
        observation['dormant_flora'] = env_state['biosphere_seeds']['dormant_flora']
        
        observation['terraforming_stations'] = env_state['infrastructure']['terraforming_stations']
        observation['station_upgrade_level'] = env_state['infrastructure']['station_upgrade_level']
        observation['energy_reserves'] = env_state['infrastructure']['energy_reserves']
        
        observation['habitability_index'] = env_state['global_metrics']['habitability_index']
        observation['instability_index'] = env_state['global_metrics']['instability_index']
        
        observation['max_steps'] = env_state.get('globals', {}).get('max_steps', 40)
        observation['t'] = t
        
        return observation