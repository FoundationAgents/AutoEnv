from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class TreasureObservationPolicy(ObservationPolicy):
    def _pos_to_key(self, pos: list) -> str:
        """Convert position [x, y] to string key 'x,y'"""
        return f"{pos[0]},{pos[1]}"
    
    def _key_to_pos(self, key: str) -> tuple:
        """Convert string key 'x,y' to position tuple (x, y)"""
        parts = key.split(',')
        return (int(parts[0]), int(parts[1]))
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        agent_x, agent_y = agent_pos
        
        # Create 5x5 window centered on agent
        visible_tiles = {}
        for dy in range(-2, 3):  # -2 to +2 inclusive
            for dx in range(-2, 3):
                world_x = agent_x + dx
                world_y = agent_y + dy
                
                # Check if position is within grid bounds
                if 0 <= world_x < 8 and 0 <= world_y < 8:
                    pos_key_str = f"{world_x},{world_y}"
                    pos_key_tuple = (world_x, world_y)
                    
                    if env_state["grid"]["revealed"].get(pos_key_str, False):
                        # Tile is revealed, show actual icon
                        visible_tiles[pos_key_tuple] = env_state["grid"]["icons"].get(pos_key_str, "empty")
                    else:
                        # Tile is not revealed
                        visible_tiles[pos_key_tuple] = "unrevealed"
                else:
                    # Out of bounds
                    visible_tiles[(world_x, world_y)] = "out_of_bounds"
        
        remaining_steps = max(0, env_state["globals"]["max_steps"] - t)
        
        return {
            "agent_pos": agent_pos,
            "remaining_steps": remaining_steps,
            "visible_tiles": visible_tiles,
            "t": t + 1
        }
