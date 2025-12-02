import sys
from typing import Dict, Any

# Add the AutoEnv path to sys.path for imports
sys.path.append('../../../')
from base.env.base_observation import ObservationPolicy

class ChemicalSensorPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = tuple(env_state["agent"]["pos"])
        chemical_map = env_state["maze"]["chemical_map"]
        maze_size = env_state["maze"]["size"]
        walls = set(tuple(wall) for wall in env_state["maze"]["walls"])
        max_steps = env_state["globals"]["max_steps"]
        
        # Get current chemical signature
        flavor_vector = chemical_map.get(f"{agent_pos[0]},{agent_pos[1]}", [0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Calculate wall bitmask for movement options
        x, y = agent_pos
        wall_bitmask = [
            self._can_move_to(x, y-1, maze_size, walls),  # North
            self._can_move_to(x+1, y, maze_size, walls),  # East  
            self._can_move_to(x, y+1, maze_size, walls),  # South
            self._can_move_to(x-1, y, maze_size, walls)   # West
        ]
        
        remaining_steps = max_steps - t
        
        return {
            "flavor_vector": flavor_vector,
            "remaining_steps": remaining_steps,
            "wall_bitmask": wall_bitmask,
            "agent_pos": agent_pos,
            "t": t
        }
    
    def _can_move_to(self, x, y, maze_size, walls):
        if not (0 <= x < maze_size[0] and 0 <= y < maze_size[1]):
            return False
        if (x, y) in walls:
            return False
        return True
