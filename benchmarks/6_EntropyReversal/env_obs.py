from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class FullObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        return deepcopy(env_state)