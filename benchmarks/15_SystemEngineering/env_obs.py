from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
import copy

class FullObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        obs['valve_states'] = copy.deepcopy(env_state['valves']['states'])
        obs['pipe_pressures'] = copy.deepcopy(env_state['hydraulics']['pipe_pressures'])
        obs['sensor_readings'] = copy.deepcopy(env_state['hydraulics']['sensor_readings'])
        obs['target_pressures'] = copy.deepcopy(env_state['hydraulics']['target_pressures'])
        obs['step_count'] = env_state['agent']['step_count']
        obs['t'] = t + 1
        return obs