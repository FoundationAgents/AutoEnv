from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import CursorObservationPolicy
from env_generate import PixelArtGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class MaskedPixelArtEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = CursorObservationPolicy(neighborhood_size=3)
        super().__init__(env_id, obs_policy)
        self.generator = None
    
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        # Initialize generator with config
        self.generator = PixelArtGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            raise ValueError("world_id must be provided in load mode")
        
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)

        masked_positions = world_state.get("canvas", {}).get("masked_positions", [])
        if masked_positions:
            world_state["canvas"]["masked_positions"] = [
                (int(pos[0]), int(pos[1])) for pos in masked_positions
            ]

        # Override max_steps if specified in level
        if "max_steps" in world_state.get("globals", {}):
            self.configs["termination"]["max_steps"] = world_state["globals"]["max_steps"]

        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        action_name = action["action"]
        params = action.get("params", {})
        
        cursor_pos = self._state["agent"]["cursor_pos"]
        
        if action_name == "MoveNorth":
            new_y = max(0, cursor_pos[1] - 1)
            self._state["agent"]["cursor_pos"] = [cursor_pos[0], new_y]
            self._last_action_result = f"Moved to ({cursor_pos[0]}, {new_y})"
        
        elif action_name == "MoveSouth":
            new_y = min(9, cursor_pos[1] + 1)
            self._state["agent"]["cursor_pos"] = [cursor_pos[0], new_y]
            self._last_action_result = f"Moved to ({cursor_pos[0]}, {new_y})"
        
        elif action_name == "MoveEast":
            new_x = min(9, cursor_pos[0] + 1)
            self._state["agent"]["cursor_pos"] = [new_x, cursor_pos[1]]
            self._last_action_result = f"Moved to ({new_x}, {cursor_pos[1]})"
        
        elif action_name == "MoveWest":
            new_x = max(0, cursor_pos[0] - 1)
            self._state["agent"]["cursor_pos"] = [new_x, cursor_pos[1]]
            self._last_action_result = f"Moved to ({new_x}, {cursor_pos[1]})"
        
        elif action_name.startswith("WriteColor"):
            color_index = int(action_name.replace("WriteColor", ""))
            x, y = cursor_pos
            self._state["canvas"]["pixels"][y][x] = color_index
            self._last_action_result = f"Wrote color {color_index} at ({x}, {y})"
        
        elif action_name == "Skip":
            self._last_action_result = f"Skipped at ({cursor_pos[0]}, {cursor_pos[1]})"
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        action_name = action["action"]
        reward = 0.0
        events = []
        
        if action_name.startswith("WriteColor") or action_name == "Skip":
            cursor_pos = self._state["agent"]["cursor_pos"]
            x, y = cursor_pos
            
            masked_positions = self._state["canvas"]["masked_positions"]
            ground_truth = self._state["canvas"]["ground_truth"]
            canvas = self._state["canvas"]["pixels"]
            
            # Check if this position was originally masked
            if (x, y) in masked_positions:
                # Check if current canvas value matches ground truth
                if canvas[y][x] == ground_truth[y][x]:
                    reward = 1.0
                    events.append("correct_restoration")
                    self._state["episode"]["correct_restorations"] += 1
                else:
                    events.append("incorrect_action")
            else:
                events.append("incorrect_action")
        
        reward_info = {
            "position": self._state["agent"]["cursor_pos"],
            "action": action_name,
            "correct_restorations": self._state["episode"]["correct_restorations"]
        }
        
        return reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        cursor_pos = omega["cursor_pos"]
        local_neighborhood = omega["local_neighborhood"]
        visible_colors_mask = omega["visible_colors_mask"]
        max_steps = omega["max_steps"]
        t = omega["t"]
        
        # Format local grid with cursor marked as 'A'
        grid_str = ""
        for i, row in enumerate(local_neighborhood):
            row_str = ""
            for j, cell in enumerate(row):
                if i == 1 and j == 1:  # Center cell (cursor position)
                    row_str += "A "
                else:
                    if cell == "□":
                        row_str += "□ "
                    else:
                        row_str += f"{cell} "
            grid_str += row_str.strip() + "\n"
        
        # Format visible colors
        visible_colors = [str(i) for i, visible in enumerate(visible_colors_mask) if visible]
        visible_colors_str = ", ".join(visible_colors) if visible_colors else "None"
        
        return f"""Step {t}/{max_steps} | Cursor: {cursor_pos}
Local 3x3 view (□=masked, A=agent):
{grid_str.strip()}
Visible colors: {visible_colors_str}
Actions: MoveNorth/South/East/West, WriteColor0-15, Skip"""
    
    def done(self, state=None) -> bool:
        # Check if max steps reached
        if self._t >= self.configs["termination"]["max_steps"]:
            return True
        
        # Check if all masked positions are filled (with any color)
        masked_positions = self._state["canvas"]["masked_positions"]
        canvas = self._state["canvas"]["pixels"]
        
        for x, y in masked_positions:
            if canvas[y][x] == -1:  # Still has masked placeholder
                return False
        
        return True
