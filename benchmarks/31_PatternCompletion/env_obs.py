from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class CursorObservationPolicy(ObservationPolicy):
    def __init__(self, neighborhood_size: int = 3):
        self.neighborhood_size = neighborhood_size

    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        cursor_pos = env_state["agent"]["cursor_pos"]
        canvas = env_state["canvas"]["pixels"]
        masked_positions = env_state["canvas"]["masked_positions"]
        
        # Extract local neighborhood
        local_neighborhood = self._extract_neighborhood(canvas, cursor_pos, masked_positions)
        
        # Calculate visible colors mask
        visible_colors_mask = self._get_visible_colors_mask(canvas, masked_positions)
        return {
            "cursor_pos": cursor_pos,
            "local_neighborhood": local_neighborhood,
            "visible_colors_mask": visible_colors_mask,
            "max_steps": env_state["globals"]["max_steps"],
            "t": t + 1
        }
    
    def _extract_neighborhood(self, canvas, cursor_pos, masked_positions):
        x, y = cursor_pos
        neighborhood = []
        
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 10 and 0 <= ny < 10:
                    pos = (nx, ny)
                    if pos in masked_positions:
                        row.append("□")
                    else:
                        row.append(canvas[ny][nx])
                else:
                    row.append(0)  # Background color for out of bounds
            neighborhood.append(row)
        
        return neighborhood
    
    def _get_visible_colors_mask(self, canvas, masked_positions):
        visible_colors = set()
        for y in range(10):
            for x in range(10):
                if (x, y) not in masked_positions:
                    visible_colors.add(canvas[y][x])
        
        mask = [False] * 16
        for color in visible_colors:
            if 0 <= color <= 15:
                mask[color] = True
        return mask
