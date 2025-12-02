from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
import numpy as np

class EgoRadiusFixed(ObservationPolicy):
    def __init__(self, radius: int = 2):
        self.radius = radius
        self.window_size = 2 * radius + 1
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state['agent']['pos']
        grid_size = env_state['globals']['size']
        tiles = env_state['tiles']
        
        # Create window around agent
        window = [[' ' for _ in range(self.window_size)] for _ in range(self.window_size)]
        
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                world_y = agent_pos[0] + dy
                world_x = agent_pos[1] + dx
                window_y = dy + self.radius
                window_x = dx + self.radius
                
                if 0 <= world_y < grid_size[0] and 0 <= world_x < grid_size[1]:
                    if world_y < len(tiles['data']) and world_x < len(tiles['data'][world_y]):
                        window[window_y][window_x] = tiles['data'][world_y][world_x]
                    else:
                        window[window_y][window_x] = tiles['default_type']
                
        # Place agent in center
        center = self.radius
        window[center][center] = '🧍'
        
        return {
            "visible_tiles": window,
            "agent.steps_remaining": env_state['agent']['steps_remaining'],
            # Expose 1-based timestep to observers
            "t": t + 1,
        }
