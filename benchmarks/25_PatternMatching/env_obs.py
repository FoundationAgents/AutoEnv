from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class MemoryObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {
            'card_states': env_state['game']['card_states'],
            'current_revealed_symbol': env_state['game']['current_revealed_symbol'],
            'steps_remaining': env_state['agent']['steps_remaining'],
            't': t + 1
        }