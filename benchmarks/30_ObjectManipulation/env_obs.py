import sys
sys.path.append('../../../')
from typing import Dict, Any
from base.env.base_observation import ObservationPolicy

class LocalGridObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        vision_radius = 2
        grid_size = 5
        
        visible_grid = []
        for dy in range(-vision_radius, vision_radius + 1):
            row = []
            for dx in range(-vision_radius, vision_radius + 1):
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                
                if dx == 0 and dy == 0:
                    row.append("agent")
                elif x < 0 or x >= 12 or y < 0 or y >= 12:
                    row.append("unknown")
                else:
                    cell_content = self._get_cell_content([x, y], env_state)
                    row.append(cell_content)
            visible_grid.append(row)
        
        current_room = self._get_current_room(agent_pos, env_state)
        
        visible_appliances = []
        for appliance in env_state["appliances"]:
            app_pos = appliance["pos"]
            if (abs(app_pos[0] - agent_pos[0]) <= vision_radius and 
                abs(app_pos[1] - agent_pos[1]) <= vision_radius):
                visible_appliances.append({
                    "type": appliance["type"],
                    "state": appliance["state"],
                    "pos": appliance["pos"]
                })
        
        observation = {
            "agent": env_state["agent"],
            "visible_grid": visible_grid,
            "current_room": current_room,
            "visible_appliances": visible_appliances,
            "chores": env_state["chores"],
            "globals": env_state["globals"],
            "t": t + 1  # Fix: Show step number starting from 1
        }
        
        return observation
    
    def _get_cell_content(self, pos, env_state):
        if pos in env_state["apartment"]["walls"]:
            return "wall"
        
        if pos in env_state["apartment"]["doors"]:
            return "door"
        
        for obj in env_state["objects"]:
            if obj["pos"] == pos:
                return {
                    "type": "object",
                    "name": f"{obj['color']} {obj['type']}",
                    "color": obj["color"],
                    "object_type": obj["type"]
                }
        
        for appliance in env_state["appliances"]:
            if appliance["pos"] == pos:
                return {
                    "type": "appliance",
                    "name": appliance["type"],
                    "state": appliance["state"]
                }
        
        for container in env_state["containers"]:
            if container["pos"] == pos:
                return {
                    "type": "container",
                    "name": container["type"],
                    "open": container["open"]
                }
        
        for room_data in env_state["apartment"]["rooms"].values():
            for furniture in room_data["furniture"]:
                if furniture["pos"] == pos:
                    return {
                        "type": "furniture",
                        "name": furniture["type"]
                    }
        
        return "floor"
    
    def _get_current_room(self, agent_pos, env_state):
        for room_name, room_data in env_state["apartment"]["rooms"].items():
            if "bounds" in room_data:
                bounds = room_data["bounds"]
                if (bounds[0] <= agent_pos[0] <= bounds[2] and 
                    bounds[1] <= agent_pos[1] <= bounds[3]):
                    return room_name
        return "corridor"