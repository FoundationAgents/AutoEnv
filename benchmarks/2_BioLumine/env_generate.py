import sys
sys.path.append("../../../")
from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy
import uuid

class ProtocolGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        
        base_state = deepcopy(self.config['state_template'])
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        self._save_world(world_state, save_path)
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = deepcopy(base_state)
        
        protocol_families = world_state['globals']['protocol_families']
        protocol = random.choice(protocol_families)
        param1 = random.randint(0, 10)
        param2 = random.randint(0, 10)
        
        world_state['session']['active_protocol'] = protocol
        world_state['session']['protocol_params'] = [param1, param2]
        
        pattern_length = random.randint(2, 6)
        colors = world_state['globals']['colors']
        durations = world_state['globals']['durations']
        intensities = world_state['globals']['intensities']
        
        first_pattern = []
        for _ in range(pattern_length):
            pulse = {
                'color': random.choice(colors),
                'duration': random.choice(durations),
                'intensity': random.choice(intensities)
            }
            first_pattern.append(pulse)
        
        world_state['session']['current_incoming_pattern'] = first_pattern
        
        return world_state
    
    def _save_world(self, world_state: Dict[str, Any], save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        return f"world_{str(uuid.uuid4())[:8]}"

class ProtocolEvaluator:
    @staticmethod
    def validate_response(incoming_pattern, response_pattern, protocol, params):
        if protocol == "color_mirroring":
            return ProtocolEvaluator._validate_color_mirroring(incoming_pattern, response_pattern, params)
        elif protocol == "duration_inversion":
            return ProtocolEvaluator._validate_duration_inversion(incoming_pattern, response_pattern, params)
        elif protocol == "intensity_parity":
            return ProtocolEvaluator._validate_intensity_parity(incoming_pattern, response_pattern, params)
        elif protocol == "sequence_fibonacci":
            return ProtocolEvaluator._validate_sequence_fibonacci(incoming_pattern, response_pattern, params)
        return False
    
    @staticmethod
    def _validate_color_mirroring(incoming, response, params):
        colors = ["blue", "green", "purple", "white"]
        offset = params[0] % len(colors)
        
        if len(incoming) != len(response):
            return False
        
        for i, pulse in enumerate(incoming):
            expected_color_idx = (colors.index(pulse['color']) + offset) % len(colors)
            expected_color = colors[expected_color_idx]
            if response[i]['color'] != expected_color:
                return False
        return True
    
    @staticmethod
    def _validate_duration_inversion(incoming, response, params):
        if len(incoming) != len(response):
            return False
            
        invert = params[1] % 2 == 1
        
        for i, pulse in enumerate(incoming):
            expected_duration = pulse['duration']
            if invert:
                expected_duration = "long" if pulse['duration'] == "short" else "short"
            if response[i]['duration'] != expected_duration:
                return False
        return True
    
    @staticmethod
    def _validate_intensity_parity(incoming, response, params):
        if len(incoming) != len(response):
            return False
            
        incoming_sum = sum(1 if pulse['intensity'] == 'high' else 0 for pulse in incoming)
        response_sum = sum(1 if pulse['intensity'] == 'high' else 0 for pulse in response)
        
        modulo = max(1, params[0])
        return (incoming_sum % modulo) == (response_sum % modulo)
    
    @staticmethod
    def _validate_sequence_fibonacci(incoming, response, params):
        fib_a, fib_b = params[0], params[1]
        expected_length = (fib_a + fib_b) % 5 + 2
        return len(response) == expected_length
    
    @staticmethod
    def generate_next_pattern(protocol, params, colors, durations, intensities):
        pattern_length = random.randint(2, 6)
        pattern = []
        
        for _ in range(pattern_length):
            pulse = {
                'color': random.choice(colors),
                'duration': random.choice(durations),
                'intensity': random.choice(intensities)
            }
            pattern.append(pulse)
        
        return pattern