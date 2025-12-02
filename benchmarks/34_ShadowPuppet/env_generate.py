from base.env.base_generator import WorldGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional
from copy import deepcopy
import time

class ShadowWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.property_mappings = {
            'square': 'Heavy',
            'circle': 'Light',
            'triangle': 'Bouncy',
            'cross': 'Sticky'
        }
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = deepcopy(self.config['state_template'])
        
        generated_state = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(generated_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = deepcopy(base_state)
        
        pipeline = self.config['generator']['pipeline']
        
        for step in pipeline:
            if step['name'] == 'init_from_template':
                continue
            elif step['name'] == 'randomize_positions':
                state = self._randomize_positions(state, step['args'])
            elif step['name'] == 'set_goal_area':
                state = self._set_goal_area(state, step['args'])
            elif step['name'] == 'assign_properties':
                state = self._assign_properties(state, step['args'])
            elif step['name'] == 'validate_solvability':
                if not self._validate_solvability(state, step['args']):
                    state = self._execute_pipeline(base_state, seed)
                    break
        
        return state
    
    def _randomize_positions(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        grid_size = state['globals']['grid_size']
        min_distance_from_goal = args.get('min_distance_from_goal', 3)
        
        goal_area = state['globals']['goal_area']
        goal_positions = set()
        for x in range(goal_area[0][0], goal_area[1][0] + 1):
            for y in range(goal_area[0][1], goal_area[1][1] + 1):
                goal_positions.add((x, y))
        
        occupied_positions = set()
        occupied_positions.add(tuple(state['globals']['light_source']))
        occupied_positions.update(goal_positions)
        
        for obj in state['objects']:
            valid_positions = []
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
                    if (x, y) not in occupied_positions:
                        min_dist_to_goal = min(
                            abs(x - gx) + abs(y - gy) 
                            for gx, gy in goal_positions
                        )
                        if min_dist_to_goal >= min_distance_from_goal:
                            valid_positions.append([x, y])
            
            if valid_positions:
                new_pos = random.choice(valid_positions)
                obj['position'] = new_pos
                occupied_positions.add(tuple(new_pos))
        
        shadow_valid_positions = [
            [x, y] for x in range(grid_size[0]) for y in range(grid_size[1])
            if (x, y) not in occupied_positions
        ]
        if shadow_valid_positions:
            state['shadow']['position'] = random.choice(shadow_valid_positions)
        
        return state
    
    def _set_goal_area(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        size = args.get('size', [2, 2])
        min_distance_from_start = args.get('min_distance_from_start', 4)
        grid_size = state['globals']['grid_size']
        
        target_pos = None
        for obj in state['objects']:
            if obj.get('is_target', False):
                target_pos = obj['position']
                break
        
        if target_pos:
            valid_goal_positions = []
            for x in range(grid_size[0] - size[0] + 1):
                for y in range(grid_size[1] - size[1] + 1):
                    distance = abs(x - target_pos[0]) + abs(y - target_pos[1])
                    if distance >= min_distance_from_start:
                        valid_goal_positions.append([[x, y], [x + size[0] - 1, y + size[1] - 1]])
            
            if valid_goal_positions:
                state['globals']['goal_area'] = random.choice(valid_goal_positions)
        
        return state
    
    def _assign_properties(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        target_property = args.get('target_property', 'Heavy')
        randomize_others = args.get('randomize_others', True)
        
        properties = ['Heavy', 'Light', 'Bouncy', 'Sticky']
        
        for obj in state['objects']:
            if obj.get('is_target', False):
                obj['property'] = target_property
            elif randomize_others:
                obj['property'] = random.choice(properties)
        
        shapes = ['square', 'circle', 'triangle', 'cross']
        state['shadow']['shape'] = random.choice(shapes)
        
        return state
    
    def _validate_solvability(self, state: Dict[str, Any], args: Dict[str, Any]) -> bool:
        max_steps = args.get('max_validation_steps', 40)
        
        target_pos = None
        for obj in state['objects']:
            if obj.get('is_target', False):
                target_pos = obj['position']
                break
        
        if not target_pos:
            return False
        
        goal_area = state['globals']['goal_area']
        goal_x_range = range(goal_area[0][0], goal_area[1][0] + 1)
        goal_y_range = range(goal_area[0][1], goal_area[1][1] + 1)
        
        if target_pos[0] in goal_x_range and target_pos[1] in goal_y_range:
            return False
        
        manhattan_distance = min(
            abs(target_pos[0] - gx) + abs(target_pos[1] - gy)
            for gx in goal_x_range for gy in goal_y_range
        )
        
        return manhattan_distance <= max_steps // 2
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000) % 100000
        if seed is not None:
            return f"world_s{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"