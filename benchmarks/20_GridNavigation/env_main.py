from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import EgocentricCropPolicy
from env_generate import MazeGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class UndergroundRuinEnv(SkinEnv):
    def __init__(self, env_id: str):
        obs_policy = EgocentricCropPolicy(crop_size=5, mask_token="Unknown")
        super().__init__(env_id, obs_policy)
    
    def _dsl_config(self):
        config_path = "./config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.configs = yaml.safe_load(f)
        else:
            # Fallback config
            self.configs = {
                "meta": {
                    "id": "underground_ruin_v1",
                    "name": "UndergroundRuin"
                },
                "termination": {
                    "max_steps": 40
                },
                "generator": {
                    "mode": "procedural",
                    "output_format": "yaml"
                }
            }
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load" and world_id is not None:
            world_state = self._load_world(world_id)
        else:
            generated_world_id = self._generate_world(seed)
            world_state = self._load_world(generated_world_id)
        
        self._state = deepcopy(world_state)
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        # Ensure steps_left is set correctly
        max_steps = self.configs.get("termination", {}).get("max_steps", 40)
        self._state["agent"]["steps_left"] = max_steps
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        file_path = f"./levels/{world_id}.yaml"
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = MazeGenerator(self.env_id, self.configs.get("generator", {}))
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action["action"]
        params = action.get("params", {})
        
        # Store previous state
        self._history.append(deepcopy(self._state))
        
        # Get current agent position and facing
        agent_pos = self._state["agent"]["pos"]
        agent_facing = self._state["agent"]["facing"]
        tiles = self._state["tiles"]
        
        # Direction mappings
        directions = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
        dir_names = ["N", "E", "S", "W"]
        
        # Process action
        if action_name in ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]:
            # Determine movement direction
            if action_name == "MOVE_NORTH":
                dx, dy = 0, -1
            elif action_name == "MOVE_SOUTH":
                dx, dy = 0, 1
            elif action_name == "MOVE_EAST":
                dx, dy = 1, 0
            else:  # MOVE_WEST
                dx, dy = -1, 0
            
            new_x = agent_pos[0] + dx
            new_y = agent_pos[1] + dy
            
            # Check bounds and tile type
            width = len(tiles[0])
            height = len(tiles)
            
            if (0 <= new_x < width and 0 <= new_y < height and 
                tiles[new_y][new_x] not in ["Wall", "Water"]):
                
                self._state["agent"]["pos"] = [new_x, new_y]
                
                # Check for special tiles
                target_tile = tiles[new_y][new_x]
                if target_tile == "Fire":
                    self._last_action_result = "fire"
                elif target_tile == "Treasure":
                    self._last_action_result = "treasure"
        
        elif action_name == "ROTATE_LEFT":
            current_idx = dir_names.index(agent_facing)
            new_idx = (current_idx - 1) % 4
            self._state["agent"]["facing"] = dir_names[new_idx]
        
        elif action_name == "ROTATE_RIGHT":
            current_idx = dir_names.index(agent_facing)
            new_idx = (current_idx + 1) % 4
            self._state["agent"]["facing"] = dir_names[new_idx]
        
        # WAIT does nothing
        
        # Decrement steps
        self._state["agent"]["steps_left"] -= 1
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if self._last_action_result == "treasure":
            return 1.0, ["goal"], {}
        return 0.0, [], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        visible_tiles = omega["visible_tiles"]
        agent_facing = omega["agent_facing"]
        steps_left = omega["agent_steps_left"]
        t = omega["t"]
        
        # Tile to character mapping
        tile_chars = {
            "Empty": ".",
            "Wall": "#",
            "Water": "~",
            "Fire": "^",
            "Treasure": "$",
            "Agent": "@",
            "Unknown": "?"
        }
        
        # Build ASCII representation
        ascii_lines = []
        for row in visible_tiles:
            line = ""
            for tile in row:
                line += tile_chars.get(tile, "?")
            ascii_lines.append(line)
        
        ascii_crop = "\n".join(ascii_lines)
        
        max_steps = self.configs.get("termination", {}).get("max_steps", 40)
        
        result = f"Step {t}/{max_steps}\n"
        result += f"Facing: {agent_facing}\n"
        result += f"Steps left: {steps_left}\n"
        result += f"Local view:\n{ascii_crop}\n"
        result += "Legend: @=Agent #=Wall ~=Water ^=Fire $=Treasure"
        
        return result
    
    def done(self, state=None) -> bool:
        if self._last_action_result in ["fire", "treasure"]:
            return True
        
        max_steps = self.configs.get("termination", {}).get("max_steps", 40)
        if self._t >= max_steps:
            return True
            
        if self._state["agent"]["steps_left"] <= 0:
            return True
            
        return False