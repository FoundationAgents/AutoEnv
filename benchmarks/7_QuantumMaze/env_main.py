from base.env.base_env import SkinEnv
from env_obs import QuantumObservationPolicy
from env_generate import QuantumMazeGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List

class QuantumMazeEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = QuantumObservationPolicy()
        super().__init__(env_id, obs_policy)
        # Load config immediately during initialization
        self._dsl_config()
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        # Initialize state if not exists
        if self._state is None:
            self._state = {
                "globals": {"grid_size": [10, 10], "start_pos": [0, 0], "exit_pos": [9, 9]},
                "agent": {"pos": [0, 0]},
                "maze": {"size": [10, 10], "quantum_walls": {}, "collapsed_walls": {}, "wall_probabilities": {}}
            }
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = QuantumMazeGenerator(str(self.env_id), self.configs)
        world_id = generator.generate(seed)
        return world_id
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            self._state = self._load_world(world_id)
        
        # Reset agent position and timestep
        self._state["agent"]["pos"] = self._state["globals"]["start_pos"].copy()
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        return self.observe_semantic()
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action["action"]
        params = action.get("params", {})
        
        current_pos = self._state["agent"]["pos"]
        
        if action_name == "MOVE_NORTH":
            target_pos = [current_pos[0], current_pos[1] - 1]
            self._attempt_move(target_pos)
        elif action_name == "MOVE_SOUTH":
            target_pos = [current_pos[0], current_pos[1] + 1]
            self._attempt_move(target_pos)
        elif action_name == "MOVE_EAST":
            target_pos = [current_pos[0] + 1, current_pos[1]]
            self._attempt_move(target_pos)
        elif action_name == "MOVE_WEST":
            target_pos = [current_pos[0] - 1, current_pos[1]]
            self._attempt_move(target_pos)
        elif action_name == "OBSERVE":
            self._observe_adjacent()
        
        return self._state
    
    def _attempt_move(self, target_pos: List[int]):
        grid_size = self._state["globals"]["grid_size"]
        
        # Check bounds
        if (target_pos[0] < 0 or target_pos[0] >= grid_size[0] or 
            target_pos[1] < 0 or target_pos[1] >= grid_size[1]):
            self._last_action_result = "boundary_blocked"
            return
        
        # Collapse quantum state if unknown
        cell_key = f"{target_pos[0]},{target_pos[1]}"
        if cell_key not in self._state["maze"]["collapsed_walls"]:
            self._collapse_cell(cell_key)
        
        # Check if passable
        if self._state["maze"]["collapsed_walls"][cell_key] == "wall":
            self._last_action_result = "wall_blocked"
        else:
            self._state["agent"]["pos"] = target_pos
            self._last_action_result = "moved"
    
    def _observe_adjacent(self):
        current_pos = self._state["agent"]["pos"]
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        grid_size = self._state["globals"]["grid_size"]
        
        for dx, dy in directions:
            adj_x = current_pos[0] + dx
            adj_y = current_pos[1] + dy
            
            # Check bounds
            if (adj_x >= 0 and adj_x < grid_size[0] and 
                adj_y >= 0 and adj_y < grid_size[1]):
                cell_key = f"{adj_x},{adj_y}"
                if cell_key not in self._state["maze"]["collapsed_walls"]:
                    self._collapse_cell(cell_key)
        
        self._last_action_result = "observed"
    
    def _collapse_cell(self, cell_key: str):
        wall_prob = self._state["maze"]["wall_probabilities"][cell_key]
        if random.random() < wall_prob:
            self._state["maze"]["collapsed_walls"][cell_key] = "wall"
        else:
            self._state["maze"]["collapsed_walls"][cell_key] = "empty"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if self._state["agent"]["pos"] == self._state["globals"]["exit_pos"]:
            return (1.0, ["reach_exit"], {})
        return (0.0, [], {})
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        local_view = omega["local_view"]
        agent_pos = omega["agent_pos"]
        remaining_steps = omega["remaining_steps"]
        exit_pos = self._state["globals"]["exit_pos"]
        max_steps = self._state.get("max_steps", self.configs["termination"]["max_steps"])
        
        # Create display grid
        display_lines = []
        for row in local_view:
            line = ""
            for cell in row:
                if cell == "agent":
                    line += "A"
                elif cell == "unknown":
                    line += "?"
                elif cell == "wall":
                    line += "#"
                elif cell == "empty":
                    line += "."
                elif cell == "exit":
                    line += "E"
                elif cell == "boundary":
                    line += "#"
                else:
                    line += "?"
            display_lines.append(line)
        
        local_view_display = "\n".join(display_lines)
        
        skin_output = f"""Step {self._t + 1}/{max_steps} - Quantum Maze Escape
Position: {agent_pos} | Target: {exit_pos}

Local View (3x3 around agent):
{local_view_display}

Legend: A=Agent, ?=Unknown(Quantum), #=Wall, .=Empty, E=Exit

Available Actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, OBSERVE
Remaining Steps: {remaining_steps}"""
        
        return skin_output
    
    def done(self, state=None) -> bool:
        max_steps = self._state.get("max_steps", self.configs["termination"]["max_steps"])
        return (self._state["agent"]["pos"] == self._state["globals"]["exit_pos"] or 
                self._t >= max_steps)
