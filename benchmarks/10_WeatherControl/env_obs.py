from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class AtmosphereObservationPolicy(ObservationPolicy):
    def __init__(self):
        pass
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        atmosphere = env_state.get('atmosphere', {})
        agent = env_state.get('agent', {})
        globals_data = env_state.get('globals', {})
        
        observation = {
            'climate_stability_index': round(atmosphere.get('climate_stability_index', 50.0), 1),
            'temperature': round(atmosphere.get('temperature', 300.0), 1),
            'humidity': round(atmosphere.get('humidity', 50.0), 1),
            'atmospheric_pressure': round(atmosphere.get('atmospheric_pressure', 1.0), 1),
            'cloud_coverage': round(atmosphere.get('cloud_coverage', 50.0), 1),
            'storm_energy': round(atmosphere.get('storm_energy', 30.0), 1),
            'solar_flux': round(atmosphere.get('solar_flux', 1000.0), 1),
            'energy_budget': agent.get('energy_budget', 45),
            'step_counter': agent.get('step_counter', 0),
            't': t,
            'max_steps': globals_data.get('max_steps', 30)
        }
        
        return observation