from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import EgoRadiusFixed
from env_generate import DeceptiveGridGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os
from copy import deepcopy

class DeceptiveGridWorld(SkinEnv):
    def __init__(self, env_id):
        obs_policy = EgoRadiusFixed(radius=2)
        super().__init__(env_id, obs_policy)
    
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Override max_steps if specified in world
        if 'termination' in self._state and 'max_steps' in self._state['termination']:
            self.configs['termination']['max_steps'] = self._state['termination']['max_steps']
        
        # Reset time and history
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        file_path = f"./levels/{world_id}.yaml"
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = DeceptiveGridGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # Store previous state in history
        if self._state is not None:
            self._history.append(deepcopy(self._state))
        
        # Decrement steps remaining
        self._state['agent']['steps_remaining'] -= 1
        
        # Parse action
        action_name = action.get('action', '')
        
        # Define movement mappings
        moves = {
            'MoveNorth': (-1, 0),
            'MoveSouth': (1, 0), 
            'MoveEast': (0, 1),
            'MoveWest': (0, -1),
            'Wait': (0, 0)
        }
        
        if action_name in moves:
            dy, dx = moves[action_name]
            current_pos = self._state['agent']['pos']
            new_y = current_pos[0] + dy
            new_x = current_pos[1] + dx
            
            # Check bounds
            grid_size = self._state['globals']['size']
            if 0 <= new_y < grid_size[0] and 0 <= new_x < grid_size[1]:
                self._state['agent']['pos'] = [new_y, new_x]
                # Record what tile was entered
                tile = self._get_tile_at([new_y, new_x])
                self._last_action_result = tile
            else:
                # Out of bounds - agent stays in place but action still consumed
                self._last_action_result = "boundary"
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        agent_pos = self._state['agent']['pos']
        goal_pos = self._state['special']['goal_pos']
        
        if agent_pos == goal_pos:
            return 1.0, ["success"], {"reason": "reached_goal"}
        else:
            return 0.0, [], {}
    
    def done(self, state=None) -> bool:
        if state is None:
            state = self._state
        
        agent_pos = state['agent']['pos']
        goal_pos = state['special']['goal_pos']
        
        # Success condition
        if agent_pos == goal_pos:
            return True
        
        # Trap condition
        tile = self._get_tile_at(agent_pos)
        if tile == '💰':
            return True
        
        # Step limit condition
        if state['agent']['steps_remaining'] <= 0:
            return True
        
        return False
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> Any:
        visible_tiles = omega.get('visible_tiles', [])
        steps_remaining = omega.get('agent.steps_remaining', 0)
        t = omega.get('t', 0)
        max_steps = self.configs['termination']['max_steps']
        
        # Convert 2D tile array to ASCII string
        visible_ascii = '\n'.join([''.join(row) for row in visible_tiles])
        
        # Use skin template from config
        template = self.configs['skin']['template']
        
        # Create a namespace for template formatting
        format_dict = {
            't': t,
            'max_steps': max_steps,
            'agent': {'steps_remaining': steps_remaining},
            'visible_ascii': visible_ascii
        }
        
        # Handle the dot notation in template by replacing with underscore format
        template_fixed = template.replace('{agent.steps_remaining}', '{agent_steps_remaining}')
        format_dict['agent_steps_remaining'] = steps_remaining
        
        return template_fixed.format(**format_dict)
    
    def _get_tile_at(self, pos: List[int]) -> str:
        row, col = pos
        tiles = self._state['tiles']
        
        if 'data' in tiles and row < len(tiles['data']) and col < len(tiles['data'][row]):
            return tiles['data'][row][col]
        else:
            return tiles['default_type']