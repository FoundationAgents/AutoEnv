from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import DreamObservationPolicy
from env_generate import DreamGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class DreamNavEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = DreamObservationPolicy()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Reset episode state
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        # Reset agent state for episode
        self._state['agent']['has_key'] = False
        self._state['agent']['wait_used_in_room'] = False
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        # Override max_steps if specified in loaded world
        if 'globals' in world_state and 'max_steps' in world_state['globals']:
            self.configs["termination"]["max_steps"] = world_state['globals']['max_steps']
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = DreamGenerator(str(self.env_id), self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action', '')
        params = action.get('params', {})
        
        # Store previous state in history
        self._history.append(self._state.copy())
        
        current_room = self._state['agent']['current_room']
        connections = self._state['world']['connections']
        room_type = self._state['world']['rooms'][current_room]['type']
        
        if action_name == "ENTER_RED_DOOR":
            self._handle_door_entry(current_room, 'red', room_type)
        elif action_name == "ENTER_BLUE_DOOR":
            self._handle_door_entry(current_room, 'blue', room_type)
        elif action_name == "ENTER_GREEN_DOOR":
            self._handle_door_entry(current_room, 'green', room_type)
        elif action_name == "PICK_UP_KEY":
            if self._state['world']['key_location'] == current_room and not self._state['agent']['has_key']:
                self._state['agent']['has_key'] = True
                self._last_action_result = "Key picked up successfully"
            else:
                self._last_action_result = "No key to pick up here"
        elif action_name == "WAIT":
            self._state['agent']['wait_used_in_room'] = True
            self._last_action_result = "Waited in room"
        
        return self._state
    
    def _handle_door_entry(self, current_room: int, door_color: str, room_type: str):
        connections = self._state['world']['connections']
        room_connections = connections.get(current_room, {})
        
        # Check if door exists
        if door_color not in room_connections:
            self._last_action_result = f"No {door_color} door in this room"
            return
        
        # Check Time-Slow restriction
        if room_type == "Time-Slow" and not self._state['agent']['wait_used_in_room']:
            self._last_action_result = "Must wait before using doors in Time-Slow room"
            return
        
        # Apply Anti-Gravity effect
        effective_color = door_color
        if room_type == "Anti-Gravity":
            if door_color == "red":
                effective_color = "blue"
            elif door_color == "blue":
                effective_color = "red"
        
        # Get destination (use original door color for connection lookup)
        destination = room_connections.get(door_color)
        if destination is not None:
            self._state['agent']['current_room'] = destination
            self._state['agent']['wait_used_in_room'] = False  # Reset for new room
            self._last_action_result = f"Moved to room {destination}"
        else:
            self._last_action_result = f"No {door_color} door in this room"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        # Check win condition
        if (self._state['agent']['current_room'] == self._state['world']['portal_room'] and 
            self._state['agent']['has_key']):
            events.append("reach_portal_with_key")
            reward = 1.0
            reward_info['win'] = True
        else:
            reward = 0.0
            reward_info['win'] = False
        
        return reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        template = self.configs['skin']['template']
        
        # Add max_steps to omega for template rendering
        omega_with_max = omega.copy()
        omega_with_max['max_steps'] = self.configs['termination']['max_steps']
        
        return template.format(**omega_with_max)
    
    def done(self, state: Dict[str, Any] = None) -> bool:
        if state is None:
            state = self._state
        
        # Check max steps
        if self._t >= self.configs["termination"]["max_steps"]:
            return True
        
        # Check win condition
        if (state['agent']['current_room'] == state['world']['portal_room'] and 
            state['agent']['has_key']):
            return True
        
        # Check for dead-end rooms (rooms with no exits)
        current_room = state['agent']['current_room']
        connections = state['world']['connections']
        if current_room in connections and len(connections[current_room]) == 0:
            return True
        
        return False