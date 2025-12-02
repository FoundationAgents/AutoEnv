from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
from copy import deepcopy

class GearObservationPolicy(ObservationPolicy):
    def __init__(self, policy_type: str = "full", params: Dict[str, Any] = None):
        self.policy_type = policy_type
        self.params = params or {}
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        if self.policy_type == "full":
            return {
                "available_gears": deepcopy(env_state["gear_system"]["available_gears"]),
                "gear_chain": deepcopy(env_state["gear_system"]["gear_chain"]),
                "current_ma": env_state["gear_system"]["current_ma"],
                "target_ma": env_state["gear_system"]["target_ma"],
                "remaining_steps": env_state["agent"]["remaining_steps"],
                "tolerance": env_state["globals"]["tolerance"],
                "t": t + 1,
                "episode_finished": env_state["gear_system"]["episode_finished"],
                "success": env_state["gear_system"]["success"]
            }
        else:
            return deepcopy(env_state)