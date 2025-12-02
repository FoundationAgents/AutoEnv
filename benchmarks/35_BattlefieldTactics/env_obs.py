from base.env.base_observation import ObservationPolicy
from typing import Dict, Any, List
import numpy as np

class TacticalObservationPolicy(ObservationPolicy):
    def __init__(self, sensor_radius: int = 3):
        self.sensor_radius = sensor_radius
        
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        grid_size = env_state["globals"]["grid_size"]
        squads = env_state["squads"]
        enemy_camps = env_state["enemy_camps"]
        terrain = env_state["terrain"]
        
        # Get active squads
        active_squads = [squad for squad in squads if squad["active"]]
        
        # Create tactical map by merging sensor views
        tactical_map = np.zeros(grid_size, dtype=bool)
        
        for squad in active_squads:
            pos = squad["pos"]
            # Add 7x7 sensor area around squad
            for dx in range(-self.sensor_radius, self.sensor_radius + 1):
                for dy in range(-self.sensor_radius, self.sensor_radius + 1):
                    x, y = pos[0] + dx, pos[1] + dy
                    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                        tactical_map[x][y] = True
        
        # Update visibility map in env_state
        env_state["visibility_map"]["explored"] = tactical_map.tolist()
        
        # Find discovered enemy camps (within sensor range of any active squad)
        discovered_camps = []
        for camp in enemy_camps:
            camp_pos = camp["pos"]
            for squad in active_squads:
                squad_pos = squad["pos"]
                dx = abs(camp_pos[0] - squad_pos[0])
                dy = abs(camp_pos[1] - squad_pos[1])
                if dx <= self.sensor_radius and dy <= self.sensor_radius:
                    camp_copy = camp.copy()
                    camp_copy["discovered"] = True
                    discovered_camps.append(camp_copy)
                    break
        
        # Calculate total friendly strength
        total_friendly_strength = sum(squad["strength"] for squad in active_squads)
        
        return {
            "squads": active_squads,
            "discovered_enemy_camps": discovered_camps,
            "tactical_map": tactical_map.tolist(),
            "eliminated_camps": env_state["globals"]["eliminated_camps"],
            "total_enemy_camps": env_state["globals"]["total_enemy_camps"],
            "max_steps": env_state["globals"]["max_steps"],
            "total_friendly_strength": total_friendly_strength,
            "t": t+1
        }