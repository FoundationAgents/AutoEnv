from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class RadiusObservationPolicy(ObservationPolicy):
    def __init__(self, radius: int = 1):
        self.radius = radius
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        agent_row, agent_col = agent_pos
        tiles_layout = env_state["tiles"]["layout"]
        grid_size = env_state["tiles"]["size"]
        
        visible_tiles = {}
        
        for dr in range(-self.radius, self.radius + 1):
            for dc in range(-self.radius, self.radius + 1):
                r = agent_row + dr
                c = agent_col + dc
                
                if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
                    if r < len(tiles_layout) and c < len(tiles_layout[r]):
                        visible_tiles[(dr, dc)] = tiles_layout[r][c]
                    else:
                        visible_tiles[(dr, dc)] = "unknown"
                else:
                    visible_tiles[(dr, dc)] = "unknown"
        
        return {
            "agent_pos": agent_pos,
            "steps_remaining": env_state["agent"]["steps_remaining"],
            "visible_tiles": visible_tiles,
            "t": t + 1,  # Fix: Show step number starting from 1
            "goal_pos": env_state["goal_pos"]
        }