from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class QuantumObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent_pos = env_state["agent"]["pos"]
        grid_size = env_state["globals"]["grid_size"]
        collapsed_walls = env_state["maze"]["collapsed_walls"]
        exit_pos = env_state["globals"]["exit_pos"]
        max_steps = env_state.get("max_steps", 40)
        
        # Create 3x3 local view centered on agent
        local_view = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                x = agent_pos[0] + dx
                y = agent_pos[1] + dy
                
                # Check bounds
                if x < 0 or x >= grid_size[0] or y < 0 or y >= grid_size[1]:
                    row.append("boundary")
                elif [x, y] == agent_pos:
                    row.append("agent")
                elif [x, y] == exit_pos:
                    if f"{x},{y}" in collapsed_walls:
                        if collapsed_walls[f"{x},{y}"] == "wall":
                            row.append("wall")
                        else:
                            row.append("exit")
                    else:
                        row.append("unknown")
                else:
                    cell_key = f"{x},{y}"
                    if cell_key in collapsed_walls:
                        row.append(collapsed_walls[cell_key])
                    else:
                        row.append("unknown")
            local_view.append(row)
        
        remaining_steps = max_steps - t
        
        return {
            "agent_pos": agent_pos,
            "local_view": local_view,
            "remaining_steps": remaining_steps,
            "t": t
        }