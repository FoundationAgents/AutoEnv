from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import TreasureObservationPolicy
from env_generate import TreasureWorldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class InvertedTreasureEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = TreasureObservationPolicy()
        super().__init__(env_id, obs_policy)
        self.generator = TreasureWorldGenerator(str(env_id), self.configs.get("generator", {}))
    
    def _dsl_config(self):
        config_path = "./config.yaml" 
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.configs = yaml.safe_load(f)
        else:
            # Default configuration
            self.configs = {
                "meta": {
                    "id": "inverted_treasure_hunt",
                    "name": "Inverted-Symbol Treasure Hunt Grid"
                },
                "termination": {
                    "max_steps": 30
                },
                "reward": {
                    "events": [{"trigger": "treasure_found", "value": 1.0}]
                }
            }
    
    def _pos_to_key(self, pos: List[int]) -> str:
        """Convert position [x, y] to string key 'x,y'"""
        return f"{pos[0]},{pos[1]}"
    
    def _key_to_pos(self, key: str) -> Tuple[int, int]:
        """Convert string key 'x,y' to position tuple (x, y)"""
        parts = key.split(',')
        return (int(parts[0]), int(parts[1]))
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        else:
            raise ValueError("mode must be either 'load' or 'generate'")
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World file not found: {world_path}")
        
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Standard step method required by base environment"""
        # Transition to next state
        next_state = self.transition(action)
        
        # Calculate reward
        reward_value, reward_events, reward_info = self.reward(action)
        
        # Check if done
        is_done = self.done()
        
        # Increment time step
        self._t += 1
        
        # Get observations
        observation = self.observe_semantic()
        agent_obs = self.render_skin(observation)
        
        # Create info dict
        info = {
            'raw_obs': observation,
            'skinned': agent_obs,
            'events': reward_events,
            'reward_info': reward_info,
            'last_action_result': self._last_action_result
        }
        
        return observation, reward_value, is_done, info
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        # Store previous state in history
        self._history.append(deepcopy(self._state))
        
        agent_pos = self._state["agent"]["pos"]
        x, y = agent_pos
        
        if action_name == "MOVE_NORTH":
            new_y = min(7, y + 1)
            new_pos = [x, new_y]
            self._move_to_position(new_pos)
        elif action_name == "MOVE_SOUTH":
            new_y = max(0, y - 1)
            new_pos = [x, new_y]
            self._move_to_position(new_pos)
        elif action_name == "MOVE_EAST":
            new_x = min(7, x + 1)
            new_pos = [new_x, y]
            self._move_to_position(new_pos)
        elif action_name == "MOVE_WEST":
            new_x = max(0, x - 1)
            new_pos = [new_x, y]
            self._move_to_position(new_pos)
        elif action_name == "REVEAL":
            pos_key = self._pos_to_key(agent_pos)
            self._state["grid"]["revealed"][pos_key] = True
            self._last_action_result = f"Revealed tile at {agent_pos}: {self._state['grid']['icons'].get(pos_key, 'empty')}"
        elif action_name == "WAIT":
            self._last_action_result = "Waited one turn"
        else:
            self._last_action_result = f"Unknown action: {action_name}"
        
        return self._state
    
    def _move_to_position(self, new_pos: List[int]):
        old_pos = self._state["agent"]["pos"]
        
        # Check if actually moved
        if new_pos != old_pos:
            # Auto-reveal new position if not already revealed
            pos_key = self._pos_to_key(new_pos)
            if not self._state["grid"]["revealed"].get(pos_key, False):
                self._state["grid"]["revealed"][pos_key] = True
            
            # Update agent position
            self._state["agent"]["pos"] = new_pos
            
            icon_type = self._state["grid"]["icons"].get(pos_key, "empty")
            self._last_action_result = f"Moved to {new_pos}, revealed: {icon_type}"
        else:
            self._last_action_result = f"Tried to move but stayed at {old_pos} (boundary)"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        agent_pos_key = self._pos_to_key(self._state["agent"]["pos"])
        icon = self._state["grid"]["icons"].get(agent_pos_key, "empty")
        
        if icon == "bomb":
            return (1.0, ["treasure_found"], {"found_treasure": True})
        
        return (0.0, [], {})
    
    def done(self, state=None) -> bool:
        """Fixed signature to match base class requirement"""
        # Check if stepped on flower (death)
        agent_pos_key = self._pos_to_key(self._state["agent"]["pos"])
        icon = self._state["grid"]["icons"].get(agent_pos_key, "empty")
        if icon == "flower":
            return True
        
        # Check if stepped on bomb (treasure found)
        if icon == "bomb":
            return True
        
        # Check if max steps reached
        max_steps = self._state.get("globals", {}).get("max_steps", self.configs["termination"]["max_steps"])
        if self._t >= max_steps:
            return True
        
        return False
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        agent_pos = omega["agent_pos"]
        remaining_steps = omega["remaining_steps"]
        visible_tiles = omega["visible_tiles"]
        t = omega["t"]
        
        # Create grid view string
        grid_lines = []
        agent_x, agent_y = agent_pos
        
        # Render 5x5 grid from top to bottom
        for dy in range(2, -3, -1):  # 2, 1, 0, -1, -2
            line = ""
            for dx in range(-2, 3):  # -2, -1, 0, 1, 2
                world_x = agent_x + dx
                world_y = agent_y + dy
                
                if world_x == agent_x and world_y == agent_y:
                    line += "A "  # Agent
                else:
                    pos_key = (world_x, world_y)
                    tile = visible_tiles.get(pos_key, "out_of_bounds")
                    
                    if tile == "out_of_bounds":
                        line += "# "
                    elif tile == "unrevealed":
                        line += "? "
                    elif tile == "bomb":
                        line += "B "
                    elif tile == "flower":
                        line += "F "
                    elif tile == "empty":
                        line += ". "
                    else:
                        line += "? "
            
            grid_lines.append(line.rstrip())
        
        grid_view = "\n".join(grid_lines)
        
        max_steps = self._state.get("globals", {}).get("max_steps", self.configs["termination"]["max_steps"])
        
        return f"""Step {t}/{max_steps} - Inverted Symbol Hunt
Position: {agent_pos} | Remaining Steps: {remaining_steps}

Local View (5x5 centered on agent):
{grid_view}

Legend: A=Agent, B=Bomb(treasure!), F=Flower(danger!), .=Empty, ?=Unrevealed
Remember: BOMBS are treasure, FLOWERS are deadly traps!

Available: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, REVEAL, WAIT"""
