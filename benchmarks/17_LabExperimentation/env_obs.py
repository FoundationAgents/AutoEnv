from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class FullLabObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = deepcopy(env_state)
        # Expose 1-based timestep to observers
        obs["t"] = t + 1
        return obs
