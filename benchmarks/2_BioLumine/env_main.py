import sys
sys.path.append("../../../")
from base.env.base_env import SkinEnv
from env_obs import CommunicationObserver
from env_generate import ProtocolGenerator, ProtocolEvaluator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class BioluminescentEnv(SkinEnv):
    def __init__(self, env_id: str):
        obs_policy = CommunicationObserver()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            raise ValueError("world_id must be provided when mode is 'load'")
            
        world_state = self._load_world(world_id)
        self._state = world_state
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if 'max_steps' in world_state.get('globals', {}):
            self.configs["termination"]["max_steps"] = world_state['globals']['max_steps']
        
        if mode == "generate":
            return world_id
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = ProtocolGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if action['action'] != 'RESPOND_PATTERN':
            self._last_action_result = f"Unknown action: {action['action']}"
            return self._state
        
        params = action['params']
        pattern_length = params.get('pattern_length')
        pulse_colors = params.get('pulse_colors', [])
        pulse_durations = params.get('pulse_durations', [])
        pulse_intensities = params.get('pulse_intensities', [])
        
        if not (2 <= pattern_length <= 6):
            self._state['session']['session_active'] = False
            self._last_action_result = "Invalid pattern length"
            return self._state
        
        if len(pulse_colors) != pattern_length or len(pulse_durations) != pattern_length or len(pulse_intensities) != pattern_length:
            self._state['session']['session_active'] = False
            self._last_action_result = "Mismatched pulse attribute lengths"
            return self._state
        
        response_pattern = []
        for i in range(pattern_length):
            pulse = {
                'color': pulse_colors[i],
                'duration': pulse_durations[i],
                'intensity': pulse_intensities[i]
            }
            response_pattern.append(pulse)
        
        self._state['agent']['last_response'] = response_pattern
        
        protocol = self._state['session']['active_protocol']
        params_vals = self._state['session']['protocol_params']
        incoming_pattern = self._state['session']['current_incoming_pattern']
        
        is_valid = ProtocolEvaluator.validate_response(
            incoming_pattern, response_pattern, protocol, params_vals
        )
        
        exchange = {
            'incoming_pattern': incoming_pattern,
            'response_pattern': response_pattern,
            'accepted': is_valid
        }
        
        self._state['history']['exchanges'].append(exchange)
        if len(self._state['history']['exchanges']) > self._state['history']['max_history']:
            self._state['history']['exchanges'] = self._state['history']['exchanges'][-self._state['history']['max_history']:]
        
        if is_valid:
            self._state['session']['handshakes_completed'] += 1
            self._last_action_result = "Communication accepted"
            
            if self._state['session']['handshakes_completed'] < 3:
                next_pattern = ProtocolEvaluator.generate_next_pattern(
                    protocol, params_vals,
                    self._state['globals']['colors'],
                    self._state['globals']['durations'],
                    self._state['globals']['intensities']
                )
                self._state['session']['current_incoming_pattern'] = next_pattern
        else:
            self._state['session']['session_active'] = False
            self._last_action_result = "Communication rejected"
        
        self._state['session']['energy'] = max(0, self._state['session']['energy'] - 1)
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        if len(self._state['history']['exchanges']) > 0:
            last_exchange = self._state['history']['exchanges'][-1]
            if last_exchange['accepted']:
                events.append("handshake_success")
                reward_info['handshake_result'] = 'accepted'
                return 1.0, events, reward_info
            else:
                events.append("handshake_failure")
                reward_info['handshake_result'] = 'rejected'
                return 0.0, events, reward_info
        
        return 0.0, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t + 1)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        pattern_display = ""
        if omega['current_incoming_pattern']:
            pattern_display = "\n".join([
                f"  Pulse {i+1}: {pulse['color']} | {pulse['duration']} | {pulse['intensity']}"
                for i, pulse in enumerate(omega['current_incoming_pattern'])
            ])
        else:
            pattern_display = "  No active pattern"
        
        history_display = ""
        if omega['exchanges']:
            for i, exchange in enumerate(omega['exchanges'][-3:]):
                result = "✓ Accepted" if exchange['accepted'] else "✗ Rejected"
                history_display += f"  Exchange {i+1}: {len(exchange['incoming_pattern'])} pulses -> {len(exchange['response_pattern'])} pulses ({result})\n"
        else:
            history_display = "  No previous exchanges\n"
        
        return f"""=== DEEP-SEA COMMUNICATION SESSION ===
Step: {omega['t']}/{omega['max_steps']} | Energy: {omega['energy']} | Handshakes: {omega['handshakes_completed']}/3

INCOMING PATTERN:
{pattern_display}

COMMUNICATION HISTORY:
{history_display}
Available colors: {omega['colors']}
Available durations: {omega['durations']}
Available intensities: {omega['intensities']}

Action: RESPOND_PATTERN(pattern_length, pulse_colors, pulse_durations, pulse_intensities)"""
    
    def done(self, state=None) -> bool:
        return (self._t >= self.configs["termination"]["max_steps"] or 
                self._state['session']['handshakes_completed'] >= 3 or
                not self._state['session']['session_active'])