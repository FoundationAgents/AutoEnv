from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class LocalWindowObservation(ObservationPolicy):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state['agent']['pos']
        grid_size = env_state['globals']['grid_size']
        max_steps = env_state['globals']['max_steps']
        
        # Create local grid centered on agent
        local_grid = []
        half_window = self.window_size // 2
        
        for dy in range(-half_window, half_window + 1):
            row = []
            for dx in range(-half_window, half_window + 1):
                new_x = agent_pos[0] + dx
                new_y = agent_pos[1] + dy
                
                # Check boundaries
                if new_x < 0 or new_x >= grid_size[0] or new_y < 0 or new_y >= grid_size[1]:
                    row.append('#')
                    continue
                
                # Check for goal flag
                goal_pos = env_state['objects']['goal_flag']['pos']
                if [new_x, new_y] == goal_pos and not env_state['objects']['goal_flag']['collected']:
                    row.append('G')
                    continue
                
                # Check for ice tiles
                is_ice = False
                for ice_tile in env_state['objects']['ice_tiles']:
                    if ice_tile['pos'] == [new_x, new_y]:
                        row.append('I')
                        is_ice = True
                        break
                
                if not is_ice:
                    # Check if it's agent position (center)
                    if dx == 0 and dy == 0:
                        row.append('A')
                    else:
                        row.append('H')  # Hole (safe terrain)
            
            local_grid.append(row)
        
        remaining_steps = max_steps - t
        
        return {
            'local_grid': local_grid,
            'remaining_steps': remaining_steps,
            'agent_pos': agent_pos,
            't': t + 1  # Fix: Show step number starting from 1
        }