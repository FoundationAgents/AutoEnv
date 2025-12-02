from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class EMFieldObservationPolicy(ObservationPolicy):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state['agent']['pos']
        agent_x, agent_y = agent_pos
        grid_size = env_state['grid']['size']
        em_field_values = env_state['em_field']['values']
        
        # Create 3x3 window centered on agent
        half_window = self.window_size // 2
        local_field = []
        
        for dy in range(-half_window, half_window + 1):
            row = []
            for dx in range(-half_window, half_window + 1):
                x, y = agent_x + dx, agent_y + dy
                if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                    row.append(em_field_values[y][x])
                else:
                    row.append(0)
            local_field.append(row)
        
        return {
            'local_em_field': local_field,
            'agent_facing': env_state['agent']['facing'],
            'steps_remaining': env_state['steps_remaining'],
            't': t + 1
        }