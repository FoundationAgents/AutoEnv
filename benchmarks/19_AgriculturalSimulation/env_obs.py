from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class RadiusObserver(ObservationPolicy):
    def __init__(self, radius: int = 2):
        self.radius = radius
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        map_size = env_state["tiles"]["size"]
        
        visible_tiles = []
        center_x, center_y = agent_pos
        
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                world_x = center_x + dx
                world_y = center_y + dy
                
                if 0 <= world_x < map_size[0] and 0 <= world_y < map_size[1]:
                    tile_type = "grass"
                    detail = None
                    
                    for fence in env_state["objects"]["fences"]:
                        if fence["pos"] == [world_x, world_y]:
                            tile_type = "fence"
                            break
                    
                    for field in env_state["objects"]["fields"]:
                        if field["pos"] == [world_x, world_y]:
                            tile_type = "crop"
                            detail = {"type": field["crop_type"], "stage": field["stage"]}
                            break
                    
                    for pen in env_state["objects"]["pens"]:
                        if pen["pos"] == [world_x, world_y]:
                            tile_type = "pen"
                            detail = {"type": pen["animal_type"], "state": pen["health_state"]}
                            break
                    
                    for villager in env_state["objects"]["villagers"]:
                        if villager["pos"] == [world_x, world_y]:
                            tile_type = "villager"
                            detail = {"mood": villager["mood"]}
                            break
                    
                    visible_tiles.append({
                        "dx": dx,
                        "dy": dy,
                        "type": tile_type,
                        "detail": detail
                    })
                else:
                    visible_tiles.append({
                        "dx": dx,
                        "dy": dy,
                        "type": "out_of_bounds",
                        "detail": None
                    })
        
        max_steps = env_state["globals"]["max_steps"]
        # Keep remaining based on internal 0-based t to show full budget initially
        remaining = max_steps - t
        
        return {
            "visible_tiles": visible_tiles,
            "agent": env_state["agent"],
            # Expose 1-based t for display
            "t": t + 1,
            "remaining": remaining,
            "farm_value": env_state["globals"]["farm_value"]
        }
