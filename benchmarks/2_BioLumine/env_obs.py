import sys
sys.path.append("../../../")
from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class CommunicationObserver(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        observation = {
            'current_incoming_pattern': env_state['session']['current_incoming_pattern'],
            'handshakes_completed': env_state['session']['handshakes_completed'],
            'energy': env_state['session']['energy'],
            'exchanges': env_state['history']['exchanges'],
            'max_steps': env_state['globals']['max_steps'],
            'colors': env_state['globals']['colors'],
            'durations': env_state['globals']['durations'],
            'intensities': env_state['globals']['intensities'],
            't': t
        }
        return observation