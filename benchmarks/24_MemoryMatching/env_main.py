from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import ChaosSlideObservation
from env_generate import ChaosSlideGenerator
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import yaml
import os
from copy import deepcopy

class ChaosSlideEnv(SkinEnv):
    """Chaos Slide Puzzle Environment."""
    
    def __init__(self, env_id: int):
        obs_policy = ChaosSlideObservation()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        """Load DSL configuration from YAML file."""
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        # Convert patterns to numpy arrays for efficient comparison
        self.chaos_pattern = np.array(self.configs['state_template']['targets']['chaos_pattern'])
        self.forbidden_pattern = np.array(self.configs['state_template']['targets']['forbidden_pattern'])
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        """Reset environment by loading or generating world."""
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided in load mode")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """Load world state from file."""
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        """Generate complete world using generator pipeline."""
        generator = ChaosSlideGenerator(str(self.env_id), self.configs)
        world_id = generator.generate(seed)
        return world_id
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """State transition function."""
        self._history.append(deepcopy(self._state))
        
        action_name = action.get('action')
        
        # Find blank space position
        board = self._state['board']['grid']
        blank_pos = self._find_blank(board)
        
        # Calculate target position based on action
        new_pos = None
        if action_name == 'SLIDE_UP' and blank_pos[0] > 0:
            new_pos = (blank_pos[0] - 1, blank_pos[1])
        elif action_name == 'SLIDE_DOWN' and blank_pos[0] < 2:
            new_pos = (blank_pos[0] + 1, blank_pos[1])
        elif action_name == 'SLIDE_LEFT' and blank_pos[1] > 0:
            new_pos = (blank_pos[0], blank_pos[1] - 1)
        elif action_name == 'SLIDE_RIGHT' and blank_pos[1] < 2:
            new_pos = (blank_pos[0], blank_pos[1] + 1)
        
        if new_pos is None:
            # Illegal move
            self._last_action_result = "illegal_move"
        else:
            # Valid move - swap blank with target tile
            board[blank_pos[0]][blank_pos[1]], board[new_pos[0]][new_pos[1]] = \
                board[new_pos[0]][new_pos[1]], board[blank_pos[0]][blank_pos[1]]
            self._last_action_result = "legal_move"
        
        # Decrement steps remaining
        self._state['agent']['steps_remaining'] -= 1
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """Reward function."""
        current_board = np.array(self._state['board']['grid'])
        
        # Check for chaos pattern (success)
        if np.array_equal(current_board, self.chaos_pattern):
            return (1.0, ["chaos_pattern_reached"], {"pattern": "chaos"})
        
        # Check for forbidden pattern (failure)
        if np.array_equal(current_board, self.forbidden_pattern):
            return (0.0, ["forbidden_pattern_reached"], {"pattern": "forbidden"})
        
        # Default case
        return (0.0, ["step_taken"], {"pattern": "none"})
    
    def observe_semantic(self) -> Dict[str, Any]:
        """Semantic-level observation."""
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        """Render the final input from semantic observation."""
        board = omega['board']
        steps_remaining = omega['steps_remaining']
        chaos_pattern = omega['chaos_pattern']
        t = omega['t']
        max_steps = self.configs['state_template']['globals']['max_steps']
        
        # Format board display
        board_display = ""
        for row in board:
            board_display += " ".join([str(x) if x != 0 else " " for x in row]) + "\n"
        board_display = board_display.strip()
        
        # Format chaos pattern display
        chaos_pattern_display = ""
        for row in chaos_pattern:
            chaos_pattern_display += " ".join([str(x) if x != 0 else " " for x in row]) + "\n"
        chaos_pattern_display = chaos_pattern_display.strip()
        
        template = f"""Step {t}/{max_steps} (Remaining: {steps_remaining})

Current Board:
{board_display}

Target Chaos Pattern:
{chaos_pattern_display}

Available actions: SLIDE_UP, SLIDE_DOWN, SLIDE_LEFT, SLIDE_RIGHT
(0 = blank space that will move in the specified direction)"""
        
        return template
    
    def done(self, s_next: Dict[str, Any] = None) -> bool:
        """Check if episode is done."""
        if s_next is None:
            s_next = self._state
        
        current_board = np.array(s_next['board']['grid'])
        
        # Check success condition
        if np.array_equal(current_board, self.chaos_pattern):
            return True
        
        # Check failure condition
        if np.array_equal(current_board, self.forbidden_pattern):
            return True
        
        # Check step limit
        if s_next['agent']['steps_remaining'] <= 0:
            return True
        
        return False
    
    def _find_blank(self, board: List[List[int]]) -> Tuple[int, int]:
        """Find position of blank space (0)."""
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return (i, j)
        return (0, 0)