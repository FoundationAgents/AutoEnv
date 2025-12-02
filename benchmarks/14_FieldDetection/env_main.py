from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import EMFieldObservationPolicy
from env_generate import EMWorldGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os
from copy import deepcopy

class ElectromagneticEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = EMFieldObservationPolicy(window_size=3)
        super().__init__(env_id, obs_policy)
        self.generator = None
        self.mark_executed = False
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = EMWorldGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self.mark_executed = False
        self._last_action_result = None
        if self.generator is None:
            self._dsl_config()
        
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
            
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action')
        params = action.get('params', {})
        
        agent_pos = self._state['agent']['pos']
        agent_x, agent_y = agent_pos
        grid_size = self._state['grid']['size']
        walls = set(tuple(wall) if isinstance(wall, list) else wall for wall in self._state['walls'])
        
        new_pos = [agent_x, agent_y]
        new_facing = self._state['agent']['facing']
        
        if action_name == 'move_north':
            temp_pos = [agent_x, agent_y - 1]
            if self._is_valid_move(temp_pos, grid_size, walls):
                new_pos = temp_pos
                new_facing = 'north'
                self._last_action_result = 'moved'
            else:
                self._last_action_result = 'collision'
                
        elif action_name == 'move_south':
            temp_pos = [agent_x, agent_y + 1]
            if self._is_valid_move(temp_pos, grid_size, walls):
                new_pos = temp_pos
                new_facing = 'south'
                self._last_action_result = 'moved'
            else:
                self._last_action_result = 'collision'
                
        elif action_name == 'move_east':
            temp_pos = [agent_x + 1, agent_y]
            if self._is_valid_move(temp_pos, grid_size, walls):
                new_pos = temp_pos
                new_facing = 'east'
                self._last_action_result = 'moved'
            else:
                self._last_action_result = 'collision'
                
        elif action_name == 'move_west':
            temp_pos = [agent_x - 1, agent_y]
            if self._is_valid_move(temp_pos, grid_size, walls):
                new_pos = temp_pos
                new_facing = 'west'
                self._last_action_result = 'moved'
            else:
                self._last_action_result = 'collision'
                
        elif action_name == 'rotate_left':
            facing_map = {'north': 'west', 'west': 'south', 'south': 'east', 'east': 'north'}
            new_facing = facing_map[self._state['agent']['facing']]
            self._last_action_result = 'rotated'
            
        elif action_name == 'rotate_right':
            facing_map = {'north': 'east', 'east': 'south', 'south': 'west', 'west': 'north'}
            new_facing = facing_map[self._state['agent']['facing']]
            self._last_action_result = 'rotated'
            
        elif action_name == 'mark':
            self.mark_executed = True
            self._last_action_result = 'marked'
        
        # Update state
        self._state['agent']['pos'] = new_pos
        self._state['agent']['facing'] = new_facing
        self._state['steps_remaining'] = max(0, self._state['steps_remaining'] - 1)
        
        return self._state
    
    def _is_valid_move(self, pos: List[int], grid_size: List[int], walls: set) -> bool:
        x, y = pos
        if x < 0 or x >= grid_size[0] or y < 0 or y >= grid_size[1]:
            return False
        if (x, y) in walls:
            return False
        return True
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if action.get('action') == 'mark':
            agent_pos = self._state['agent']['pos']
            node_pos = self._state['vulnerability_node']['pos']
            
            # Calculate Manhattan distance
            distance = abs(agent_pos[0] - node_pos[0]) + abs(agent_pos[1] - node_pos[1])
            
            if distance <= 1:
                return 1.0, ['mark_success'], {'success': True, 'distance': distance}
            else:
                return 0.0, ['mark_failure'], {'success': False, 'distance': distance}
        
        return 0.0, [], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        max_steps = self._state.get('max_steps', self.configs['state_template']['globals']['max_steps'])
        steps_remaining = omega['steps_remaining']
        agent_facing = omega['agent_facing']
        local_field = omega['local_em_field']
        t = omega['t']
        
        # Format EM field display
        field_display = ""
        for row in local_field:
            field_display += " ".join(str(val) for val in row) + "\n"
        
        return f"""Step {t}/{max_steps} - Steps Remaining: {steps_remaining}
Facing: {agent_facing}

Electromagnetic Field Readings (3x3 around you):
{field_display}
Legend: 0=No Field, 1=Weak, 2=Moderate, 3=Strong
Actions: move_north, move_south, move_east, move_west, rotate_left, rotate_right, mark"""
    
    def done(self, state=None) -> bool:
        max_steps = self._state.get('max_steps', self.configs['state_template']['globals']['max_steps'])
        return self.mark_executed or self._t >= max_steps