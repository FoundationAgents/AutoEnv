from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class ConnectFourObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {
            'board_grid': env_state['board']['grid'],
            'opponent_last_move': env_state['opponent']['last_move'],
            'max_steps': env_state['globals']['max_steps'],
            'moves_made': env_state['game']['moves_made'],
            't': t + 1
        }