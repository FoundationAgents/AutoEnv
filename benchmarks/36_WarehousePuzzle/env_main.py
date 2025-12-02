from base.env.base_env import SkinEnv
from env_obs import FullObservationPolicy
from env_generate import WarehouseGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class BinaryWarehouseEnv(SkinEnv):
    def __init__(self, env_id: str = "binary_warehouse_sorting"):
        obs_policy = FullObservationPolicy()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load" and world_id:
            self._state = self._load_world(world_id)
        else:
            raise ValueError("Invalid reset mode or missing world_id")
        
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
        generator = WarehouseGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._deep_copy(self._state))
        
        action_name = action.get('action', '')
        agent_pos = self._state['agent']['pos']
        
        direction_map = {
            'MoveNorth': [-1, 0],
            'MoveSouth': [1, 0],
            'MoveEast': [0, 1],
            'MoveWest': [0, -1]
        }
        
        if action_name not in direction_map:
            self._last_action_result = "Invalid action"
            return self._state
        
        dx, dy = direction_map[action_name]
        new_x, new_y = agent_pos[0] + dx, agent_pos[1] + dy
        
        if not (0 <= new_x < 10 and 0 <= new_y < 10):
            self._last_action_result = "Out of bounds"
            return self._state
        
        grid = self._state['tiles']['grid']
        
        if grid[new_x][new_y] == 'wall':
            self._last_action_result = "Hit wall"
            return self._state
        
        box_at_target = None
        for i, box in enumerate(self._state['objects']['boxes']):
            if box['pos'] == [new_x, new_y]:
                box_at_target = i
                break
        
        if box_at_target is not None:
            box_new_x, box_new_y = new_x + dx, new_y + dy
            
            if not (0 <= box_new_x < 10 and 0 <= box_new_y < 10):
                self._last_action_result = "Cannot push box out of bounds"
                return self._state
            
            if grid[box_new_x][box_new_y] == 'wall':
                self._last_action_result = "Cannot push box into wall"
                return self._state
            
            for other_box in self._state['objects']['boxes']:
                if other_box['pos'] == [box_new_x, box_new_y]:
                    self._last_action_result = "Cannot push box into another box"
                    return self._state
            
            self._state['objects']['boxes'][box_at_target]['pos'] = [box_new_x, box_new_y]
            self._last_action_result = "Pushed box"
        else:
            self._last_action_result = "Moved"
        
        self._state['agent']['pos'] = [new_x, new_y]
        
        self._update_boxes_on_docks()
        
        return self._state
    
    def _update_boxes_on_docks(self):
        boxes_on_docks = 0
        dock_positions = {tuple(dock['pos']) for dock in self._state['objects']['docks']}
        
        for box in self._state['objects']['boxes']:
            if tuple(box['pos']) in dock_positions:
                boxes_on_docks += 1
        
        self._state['level_info']['boxes_on_docks'] = boxes_on_docks
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_boxes = self._state['level_info']['total_boxes']
        boxes_on_docks = self._state['level_info']['boxes_on_docks']
        
        if boxes_on_docks == total_boxes:
            return 1.0, ["task_completed"], {"completion_reward": 1.0}
        else:
            return 0.0, [], {"completion_reward": 0.0}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        grid = [['.' for _ in range(10)] for _ in range(10)]
        
        tiles = omega['tiles']['grid']
        for i in range(10):
            for j in range(10):
                if tiles[i][j] == 'wall':
                    grid[i][j] = '#'
        
        dock_positions = {tuple(dock['pos']) for dock in omega['docks']}
        for dock in omega['docks']:
            pos = dock['pos']
            grid[pos[0]][pos[1]] = 'D'
        
        for box in omega['boxes']:
            pos = box['pos']
            if tuple(pos) in dock_positions:
                grid[pos[0]][pos[1]] = 'X'
            else:
                grid[pos[0]][pos[1]] = 'B'
        
        agent_pos = omega['agent_pos']
        grid[agent_pos[0]][agent_pos[1]] = 'A'
        
        warehouse_grid = '\n'.join([''.join(row) for row in grid])
        
        output = f"""Warehouse Box Sorting - Step {omega['t']}/{omega['max_steps']}
Agent Position: {omega['agent_pos']}
Boxes Remaining: {omega['boxes_remaining']}/{omega['total_boxes']}

Warehouse Layout:
{warehouse_grid}

Legend: A=Agent, #=Wall, .=Floor, B=Box, D=Dock, X=BoxOnDock
Actions: MoveNorth, MoveSouth, MoveEast, MoveWest"""
        
        return output
    
    def done(self, s_next: Dict[str, Any] = None) -> bool:
        if s_next is None:
            s_next = self._state
        
        max_steps = s_next.get('globals', {}).get('max_steps', self.configs["termination"]["max_steps"])
        
        if self._t >= max_steps:
            return True
        
        if s_next['level_info']['boxes_on_docks'] == s_next['level_info']['total_boxes']:
            return True
        
        return False
    
    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        else:
            return obj