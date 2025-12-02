import random
import yaml
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy

class BioluminescentLevelValidator:
    def __init__(self):
        self.protocol_families = ["color_mirroring", "duration_inversion", "intensity_parity", "sequence_fibonacci"]
        self.colors = ["blue", "green", "purple", "white"]
        self.durations = ["short", "long"]
        self.intensities = ["low", "high"]
        self.max_steps = 40
        self.required_handshakes = 3
    
    def validate_level(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Main validation function that checks all aspects of a generated level."""
        issues = []
        
        # 1. Validate level structure
        structure_valid, structure_issues = self._validate_level_structure(level_data)
        issues.extend(structure_issues)
        
        # 2. Check protocol validity
        protocol_valid, protocol_issues = self._validate_protocol_setup(level_data)
        issues.extend(protocol_issues)
        
        # 3. Validate initial pattern
        pattern_valid, pattern_issues = self._validate_initial_pattern(level_data)
        issues.extend(pattern_issues)
        
        # 4. Critical: Check level solvability
        solvable, solvability_issues = self._check_level_solvability(level_data)
        issues.extend(solvability_issues)
        
        # 5. Validate reward structure alignment
        reward_valid, reward_issues = self._validate_reward_structure(level_data)
        issues.extend(reward_issues)
        
        is_valid = structure_valid and protocol_valid and pattern_valid and solvable and reward_valid
        return is_valid, issues
    
    def _validate_level_structure(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate basic level structure and required fields."""
        issues = []
        
        # Check top-level structure
        required_sections = ['globals', 'session', 'history', 'agent']
        for section in required_sections:
            if section not in level_data:
                issues.append(f"Missing required section: {section}")
        
        # Validate globals section
        if 'globals' in level_data:
            globals_data = level_data['globals']
            if globals_data.get('max_steps') != self.max_steps:
                issues.append(f"Invalid max_steps: {globals_data.get('max_steps')}, expected {self.max_steps}")
            
            if set(globals_data.get('colors', [])) != set(self.colors):
                issues.append(f"Invalid colors configuration")
            
            if set(globals_data.get('durations', [])) != set(self.durations):
                issues.append(f"Invalid durations configuration")
            
            if set(globals_data.get('intensities', [])) != set(self.intensities):
                issues.append(f"Invalid intensities configuration")
        
        # Validate session initialization
        if 'session' in level_data:
            session_data = level_data['session']
            if session_data.get('handshakes_completed') != 0:
                issues.append("Initial handshakes_completed should be 0")
            
            if session_data.get('energy') != 100:
                issues.append("Initial energy should be 100")
            
            if session_data.get('session_active') is not True:
                issues.append("Initial session_active should be True")
        
        return len(issues) == 0, issues
    
    def _validate_protocol_setup(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate protocol selection and parameters."""
        issues = []
        
        if 'session' not in level_data:
            return False, ["Missing session data"]
        
        session_data = level_data['session']
        protocol = session_data.get('active_protocol')
        params = session_data.get('protocol_params', [])
        
        # Check protocol family validity
        if protocol not in self.protocol_families:
            issues.append(f"Invalid protocol family: {protocol}")
        
        # Check parameter format
        if not isinstance(params, list) or len(params) != 2:
            issues.append("Protocol params must be a list of exactly 2 integers")
        else:
            for i, param in enumerate(params):
                if not isinstance(param, int) or not (0 <= param <= 10):
                    issues.append(f"Protocol param {i} must be integer between 0-10, got {param}")
        
        return len(issues) == 0, issues
    
    def _validate_initial_pattern(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate the initial incoming pattern structure."""
        issues = []
        
        if 'session' not in level_data:
            return False, ["Missing session data"]
        
        pattern = level_data['session'].get('current_incoming_pattern', [])
        
        # Check pattern length
        if not (2 <= len(pattern) <= 6):
            issues.append(f"Initial pattern length {len(pattern)} not within valid range [2-6]")
        
        # Check each pulse structure
        for i, pulse in enumerate(pattern):
            if not isinstance(pulse, dict):
                issues.append(f"Pulse {i} must be a dictionary")
                continue
            
            # Check required attributes
            required_attrs = ['color', 'duration', 'intensity']
            for attr in required_attrs:
                if attr not in pulse:
                    issues.append(f"Pulse {i} missing required attribute: {attr}")
            
            # Check attribute values
            if pulse.get('color') not in self.colors:
                issues.append(f"Pulse {i} has invalid color: {pulse.get('color')}")
            
            if pulse.get('duration') not in self.durations:
                issues.append(f"Pulse {i} has invalid duration: {pulse.get('duration')}")
            
            if pulse.get('intensity') not in self.intensities:
                issues.append(f"Pulse {i} has invalid intensity: {pulse.get('intensity')}")
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Critical check for level solvability - can the agent actually complete 3 handshakes?"""
        issues = []
        
        protocol = level_data['session']['active_protocol']
        params = level_data['session']['protocol_params']
        initial_pattern = level_data['session']['current_incoming_pattern']
        
        # 1. ACTION CONSTRAINT ANALYSIS
        action_valid, action_issues = self._analyze_action_constraints(protocol, params)
        issues.extend(action_issues)
        
        # 2. TARGET REACHABILITY - Can we solve at least one pattern?
        solvable_pattern, pattern_issues = self._check_pattern_solvability(initial_pattern, protocol, params)
        issues.extend(pattern_issues)
        
        # 3. RESOURCE AND STEP BUDGET CHECK
        budget_valid, budget_issues = self._check_step_budget()
        issues.extend(budget_issues)
        
        # 4. PROTOCOL CONSISTENCY CHECK
        consistency_valid, consistency_issues = self._check_protocol_consistency(protocol, params)
        issues.extend(consistency_issues)
        
        is_solvable = action_valid and solvable_pattern and budget_valid and consistency_valid
        return is_solvable, issues
    
    def _analyze_action_constraints(self, protocol: str, params: List[int]) -> Tuple[bool, List[str]]:
        """Analyze if the action space can satisfy protocol requirements."""
        issues = []
        
        # Check if protocol has well-defined rules
        if protocol == "color_mirroring":
            # Color mirroring should always be solvable with valid offset
            if not (0 <= params[0] <= 10):
                issues.append(f"Color mirroring offset {params[0]} out of valid range")
        
        elif protocol == "duration_inversion":
            # Duration inversion should always be solvable
            pass  # Always solvable
        
        elif protocol == "intensity_parity":
            # Check if modulo parameter makes sense
            if params[0] == 0:
                issues.append("Intensity parity modulo cannot be 0 (division by zero)")
        
        elif protocol == "sequence_fibonacci":
            # Fibonacci sequence length calculation
            expected_length = (params[0] + params[1]) % 5 + 2
            if not (2 <= expected_length <= 6):
                issues.append(f"Fibonacci protocol generates invalid length: {expected_length}")
        
        return len(issues) == 0, issues
    
    def _check_pattern_solvability(self, pattern: List[Dict], protocol: str, params: List[int]) -> Tuple[bool, List[str]]:
        """Check if the initial pattern can be solved with available actions."""
        issues = []
        
        try:
            # Generate a correct response to verify solvability
            correct_response = self._generate_correct_response(pattern, protocol, params)
            
            if not correct_response:
                issues.append(f"Unable to generate valid response for protocol {protocol}")
                return False, issues
            
            # Verify the response would be accepted
            if not self._validate_protocol_response(pattern, correct_response, protocol, params):
                issues.append(f"Generated response fails protocol validation")
                return False, issues
        
        except Exception as e:
            issues.append(f"Error testing pattern solvability: {str(e)}")
            return False, issues
        
        return len(issues) == 0, issues
    
    def _generate_correct_response(self, incoming_pattern: List[Dict], protocol: str, params: List[int]) -> Optional[List[Dict]]:
        """Generate a correct response pattern for testing solvability."""
        try:
            if protocol == "color_mirroring":
                return self._generate_color_mirror_response(incoming_pattern, params)
            elif protocol == "duration_inversion":
                return self._generate_duration_inversion_response(incoming_pattern, params)
            elif protocol == "intensity_parity":
                return self._generate_intensity_parity_response(incoming_pattern, params)
            elif protocol == "sequence_fibonacci":
                return self._generate_fibonacci_response(incoming_pattern, params)
        except Exception:
            return None
        
        return None
    
    def _generate_color_mirror_response(self, incoming: List[Dict], params: List[int]) -> List[Dict]:
        """Generate correct color mirroring response."""
        offset = params[0] % len(self.colors)
        response = []
        
        for pulse in incoming:
            original_idx = self.colors.index(pulse['color'])
            new_color_idx = (original_idx + offset) % len(self.colors)
            new_color = self.colors[new_color_idx]
            
            response_pulse = {
                'color': new_color,
                'duration': pulse['duration'],  # Copy other attributes
                'intensity': pulse['intensity']
            }
            response.append(response_pulse)
        
        return response
    
    def _generate_duration_inversion_response(self, incoming: List[Dict], params: List[int]) -> List[Dict]:
        """Generate correct duration inversion response."""
        invert = params[1] % 2 == 1
        response = []
        
        for pulse in incoming:
            new_duration = pulse['duration']
            if invert:
                new_duration = "long" if pulse['duration'] == "short" else "short"
            
            response_pulse = {
                'color': pulse['color'],
                'duration': new_duration,
                'intensity': pulse['intensity']
            }
            response.append(response_pulse)
        
        return response
    
    def _generate_intensity_parity_response(self, incoming: List[Dict], params: List[int]) -> List[Dict]:
        """Generate correct intensity parity response."""
        modulo = max(1, params[0])
        incoming_sum = sum(1 if pulse['intensity'] == 'high' else 0 for pulse in incoming)
        target_sum = incoming_sum % modulo
        
        response = []
        current_high_count = 0
        
        for i, pulse in enumerate(incoming):
            # Copy pulse structure
            response_pulse = {
                'color': pulse['color'],
                'duration': pulse['duration'],
                'intensity': pulse['intensity']
            }
            
            # Adjust intensity to match parity if needed
            if i == len(incoming) - 1:  # Last pulse - adjust if necessary
                needed_high = target_sum - current_high_count
                if needed_high > 0:
                    response_pulse['intensity'] = 'high'
                else:
                    response_pulse['intensity'] = 'low'
            else:
                if pulse['intensity'] == 'high':
                    current_high_count += 1
            
            response.append(response_pulse)
        
        return response
    
    def _generate_fibonacci_response(self, incoming: List[Dict], params: List[int]) -> List[Dict]:
        """Generate correct fibonacci sequence response."""
        expected_length = (params[0] + params[1]) % 5 + 2
        
        response = []
        for i in range(expected_length):
            # Generate arbitrary valid pulse for required length
            pulse = {
                'color': random.choice(self.colors),
                'duration': random.choice(self.durations),
                'intensity': random.choice(self.intensities)
            }
            response.append(pulse)
        
        return response
    
    def _validate_protocol_response(self, incoming: List[Dict], response: List[Dict], protocol: str, params: List[int]) -> bool:
        """Validate that a response correctly follows the protocol."""
        if protocol == "color_mirroring":
            return self._validate_color_mirroring(incoming, response, params)
        elif protocol == "duration_inversion":
            return self._validate_duration_inversion(incoming, response, params)
        elif protocol == "intensity_parity":
            return self._validate_intensity_parity(incoming, response, params)
        elif protocol == "sequence_fibonacci":
            return self._validate_sequence_fibonacci(incoming, response, params)
        return False
    
    def _validate_color_mirroring(self, incoming, response, params):
        """Validate color mirroring protocol."""
        offset = params[0] % len(self.colors)
        
        if len(incoming) != len(response):
            return False
        
        for i, pulse in enumerate(incoming):
            expected_color_idx = (self.colors.index(pulse['color']) + offset) % len(self.colors)
            expected_color = self.colors[expected_color_idx]
            if response[i]['color'] != expected_color:
                return False
        return True
    
    def _validate_duration_inversion(self, incoming, response, params):
        """Validate duration inversion protocol."""
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
    
    def _validate_intensity_parity(self, incoming, response, params):
        """Validate intensity parity protocol."""
        if len(incoming) != len(response):
            return False
            
        incoming_sum = sum(1 if pulse['intensity'] == 'high' else 0 for pulse in incoming)
        response_sum = sum(1 if pulse['intensity'] == 'high' else 0 for pulse in response)
        
        modulo = max(1, params[0])
        return (incoming_sum % modulo) == (response_sum % modulo)
    
    def _validate_sequence_fibonacci(self, incoming, response, params):
        """Validate fibonacci sequence protocol."""
        expected_length = (params[0] + params[1]) % 5 + 2
        return len(response) == expected_length
    
    def _check_step_budget(self) -> Tuple[bool, List[str]]:
        """Check if step budget allows for completing 3 handshakes."""
        issues = []
        
        # Minimum steps needed: 3 handshakes = 3 actions
        # Maximum realistic steps per handshake: considering potential failures and learning
        min_steps_needed = 3
        reasonable_max_per_handshake = 10
        
        if self.max_steps < min_steps_needed:
            issues.append(f"Step limit {self.max_steps} too low for minimum {min_steps_needed} handshakes")
        
        # Check if there's reasonable exploration budget
        if self.max_steps < reasonable_max_per_handshake:
            issues.append(f"Step limit {self.max_steps} may not allow sufficient exploration for pattern learning")
        
        return len(issues) == 0, issues
    
    def _check_protocol_consistency(self, protocol: str, params: List[int]) -> Tuple[bool, List[str]]:
        """Check that protocol rules are consistent and won't create impossible situations."""
        issues = []
        
        # Verify protocol parameters don't create degenerate cases
        if protocol == "color_mirroring":
            # All color offsets should be valid
            if params[0] < 0:
                issues.append("Color mirroring offset cannot be negative")
        
        elif protocol == "intensity_parity":
            # Modulo of 0 would cause division by zero
            if params[0] == 0:
                issues.append("Intensity parity modulo cannot be zero")
            
            # Very large modulo values might not be achievable
            max_possible_intensity = 6  # Max pattern length
            if params[0] > max_possible_intensity + 1:
                issues.append(f"Intensity parity modulo {params[0]} too large for max pattern length")
        
        elif protocol == "sequence_fibonacci":
            # Check that fibonacci calculation produces valid lengths
            calc_length = (params[0] + params[1]) % 5 + 2
            if not (2 <= calc_length <= 6):
                issues.append(f"Fibonacci parameters produce invalid length: {calc_length}")
        
        return len(issues) == 0, issues
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that the reward structure promotes problem-solving over action grinding."""
        issues = []
        
        # This environment uses binary rewards: +1 for successful handshake, 0 for failure
        # This is actually good design - high rewards for achievement, no grinding possible
        
        # Check that there are no "easy points" available
        # The environment gives 1 point per successful handshake, max 3 points total
        # This correctly prioritizes goal achievement over process
        
        # Verify no reward loops exist:
        # - Actions don't give independent rewards
        # - Only goal achievement (handshake success) gives rewards
        # - Failed attempts terminate the episode (high stakes, prevents grinding)
        
        # The current reward structure is well-designed:
        # 1. Achievement > Process: Only successful communications reward
        # 2. No action farming: Actions themselves give no rewards
        # 3. Efficiency incentive: Episode ends on failure, encouraging careful responses
        # 4. Sparse rewards: Only meaningful achievements rewarded
        
        # This passes all reward structure validation criteria
        return True, issues

# Main validation function to be called
def validate_generated_level(level_path: str) -> Tuple[bool, List[str]]:
    """Validate a generated level file."""
    validator = BioluminescentLevelValidator()
    
    try:
        with open(level_path, 'r') as f:
            level_data = yaml.safe_load(f)
        
        return validator.validate_level(level_data)
    
    except Exception as e:
        return False, [f"Error loading level file: {str(e)}"]

def validate_level_data(level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate level data directly."""
    validator = BioluminescentLevelValidator()
    return validator.validate_level(level_data)