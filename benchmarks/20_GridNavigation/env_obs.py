from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class EgocentricCropPolicy(ObservationPolicy):
    def __init__(self, crop_size=5, mask_token="Unknown"):
        self.crop_size = crop_size
        self.mask_token = mask_token
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        agent_facing = env_state["agent"]["facing"]
        steps_left = env_state["agent"]["steps_left"]
        tiles = env_state["tiles"]
        
        height = len(tiles)
        width = len(tiles[0])
        
        visible_tiles = []
        half_crop = self.crop_size // 2
        
        for dy in range(-half_crop, half_crop + 1):
            row = []
            for dx in range(-half_crop, half_crop + 1):
                y = agent_pos[1] + dy
                x = agent_pos[0] + dx
                
                if 0 <= y < height and 0 <= x < width:
                    if y == agent_pos[1] and x == agent_pos[0]:
                        row.append("Agent")
                    else:
                        row.append(tiles[y][x])
                else:
                    row.append(self.mask_token)
            visible_tiles.append(row)
        
        return {
            "visible_tiles": visible_tiles,
            "agent_facing": agent_facing,
            "agent_steps_left": steps_left,
            # display step as 1-based
            "t": t + 1,
        }
