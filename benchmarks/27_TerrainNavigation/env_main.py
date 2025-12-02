from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from base.env.base_observation import ObservationPolicy
from env_obs import RadiusObservationPolicy
from env_generate import IceLakeGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class IceLakeEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = RadiusObservationPolicy(radius=1)
        super().__init__(env_id, obs_policy)
        self.generator = None
    
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = IceLakeGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        else:
            self._state = self._load_world(world_id)
        
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        
        action_name = action["action"]
        params = action.get("params", {})
        
        current_pos = self._state["agent"]["pos"]
        new_pos = current_pos.copy()
        
        if action_name == "MoveNorth":
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action_name == "MoveSouth":
            new_pos[0] = min(self._state["tiles"]["size"][0] - 1, new_pos[0] + 1)
        elif action_name == "MoveEast":
            new_pos[1] = min(self._state["tiles"]["size"][1] - 1, new_pos[1] + 1)
        elif action_name == "MoveWest":
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action_name == "Wait":
            pass
        
        if new_pos != current_pos:
            target_tile = self._state["tiles"]["layout"][new_pos[0]][new_pos[1]]
            if target_tile in ["ice", "water"]:
                self._state["agent"]["pos"] = new_pos
                self._last_action_result = f"Moved to {target_tile}"
            else:
                self._last_action_result = "Move blocked"
        else:
            self._last_action_result = "Stayed in place"
        
        self._state["agent"]["steps_remaining"] -= 1
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        current_pos = self._state["agent"]["pos"]
        goal_pos = self._state["goal_pos"]
        
        if current_pos == goal_pos:
            return 1.0, ["goal_reached"], {"success": True}
        
        return 0.0, [], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> Any:
        visible_tiles = omega["visible_tiles"]
        agent_pos = omega["agent_pos"]
        steps_remaining = omega["steps_remaining"]
        goal_pos = omega["goal_pos"]
        
        grid_str = ""
        for dr in range(-1, 2):
            row_str = ""
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    row_str += "🤖"
                else:
                    tile_type = visible_tiles.get((dr, dc), "unknown")
                    if tile_type == "ice":
                        row_str += "🧊"
                    elif tile_type == "water":
                        row_str += "💧"
                    elif agent_pos[0] + dr == goal_pos[0] and agent_pos[1] + dc == goal_pos[1]:
                        row_str += "🏁"
                    else:
                        row_str += "⬛"
            grid_str += row_str + "\n"
        
        return {
            "position": f"({agent_pos[0]}, {agent_pos[1]})",
            "steps_remaining": steps_remaining,
            "local_grid": grid_str.strip(),
            "legend": "🧊=Ice 💧=Water 🏁=Goal ⬛=Unknown 🤖=You"
        }
    
    def done(self, state: Optional[Dict[str, Any]] = None) -> bool:
        if state is None:
            state = self._state
        
        current_pos = state["agent"]["pos"]
        goal_pos = state["goal_pos"]
        
        if current_pos == goal_pos:
            return True
        
        if state["agent"]["steps_remaining"] <= 0:
            return True
        
        current_tile = state["tiles"]["layout"][current_pos[0]][current_pos[1]]
        if current_tile == "water":
            return True
        
        return False
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Transition to next state
        next_state = self.transition(action)
        
        # Calculate reward
        reward, events, reward_info = self.reward(action)
        
        # Check if done
        done = self.done()
        
        # Get observation
        obs = self.observe_semantic()
        
        # Create info dict with skinned rendering
        info = {
            'events': events,
            'reward_info': reward_info,
            'skinned': self.render_skin(obs),
            'last_action_result': self._last_action_result
        }
        
        return next_state, reward, done, info
