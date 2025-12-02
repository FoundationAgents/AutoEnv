import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
import copy

class PressureValveValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get('termination', {}).get('max_steps', 40)
        self.num_valves = config.get('state_template', {}).get('globals', {}).get('num_valves', 9)
        self.num_sensors = config.get('state_template', {}).get('globals', {}).get('num_sensors', 4)
        
    def validate_world(self, world_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated world file for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load world file: {str(e)}"]
        
        issues = []
        
        # 1. Basic structure validation
        structure_issues = self._validate_structure(world_state)
        issues.extend(structure_issues)
        
        # 2. Level solvability analysis
        solvability_issues = self._validate_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 4. Target reachability validation
        reachability_issues = self._validate_target_reachability(world_state)
        issues.extend(reachability_issues)
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic world structure and data integrity"""
        issues = []
        
        # Check required sections exist
        required_sections = ['valves', 'hydraulics', 'agent', 'globals']
        for section in required_sections:
            if section not in world_state:
                issues.append(f"Missing required section: {section}")
        
        if 'valves' in world_state and 'states' in world_state['valves']:
            valve_states = world_state['valves']['states']
            if len(valve_states) != self.num_valves:
                issues.append(f"Invalid number of valves: expected {self.num_valves}, got {len(valve_states)}")
            
            # Check valve states are boolean
            for i, state in enumerate(valve_states):
                if not isinstance(state, bool):
                    issues.append(f"Valve {i} state must be boolean, got {type(state)}")
        
        if 'hydraulics' in world_state:
            hydraulics = world_state['hydraulics']
            
            # Check sensor readings and targets
            if 'sensor_readings' in hydraulics:
                if len(hydraulics['sensor_readings']) != self.num_sensors:
                    issues.append(f"Invalid number of sensor readings: expected {self.num_sensors}, got {len(hydraulics['sensor_readings'])}")
            
            if 'target_pressures' in hydraulics:
                if len(hydraulics['target_pressures']) != self.num_sensors:
                    issues.append(f"Invalid number of target pressures: expected {self.num_sensors}, got {len(hydraulics['target_pressures'])}")
                
                # Check target pressure ranges are reasonable
                for i, target in enumerate(hydraulics['target_pressures']):
                    if not isinstance(target, (int, float)):
                        issues.append(f"Target pressure {i} must be numeric, got {type(target)}")
                    elif target < 0:
                        issues.append(f"Target pressure {i} cannot be negative: {target}")
                    elif target > 1000:  # Reasonable upper bound
                        issues.append(f"Target pressure {i} seems unreasonably high: {target}")
        
        return issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for level solvability using action constraint analysis"""
        issues = []
        
        initial_valve_states = world_state['valves']['states']
        target_pressures = world_state['hydraulics']['target_pressures']
        pump_speed = world_state['hydraulics']['pump_speed']
        reservoir_pressure = world_state['hydraulics']['reservoir_pressure']
        
        # Check if target pressures are achievable with any valve configuration
        achievable_pressures = self._get_all_achievable_pressure_profiles(pump_speed, reservoir_pressure)
        
        target_achievable = False
        min_distance = float('inf')
        
        for pressure_profile in achievable_pressures:
            # Check if this profile matches targets (within small tolerance for numerical precision)
            distance = sum(abs(pressure_profile[i] - target_pressures[i]) for i in range(self.num_sensors))
            min_distance = min(min_distance, distance)
            
            if distance < 1e-3:  # Very small tolerance for achievability check
                target_achievable = True
                break
        
        if not target_achievable:
            issues.append(f"Target pressures {target_pressures} are not achievable with any valve configuration. Closest achievable distance: {min_distance:.6f}")
        
        # Check if solution is reachable within step limit
        if target_achievable:
            min_steps_needed = self._estimate_min_steps_to_solution(initial_valve_states, target_pressures, pump_speed, reservoir_pressure)
            if min_steps_needed > self.max_steps:
                issues.append(f"Solution requires at least {min_steps_needed} steps, but only {self.max_steps} steps available")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate reward structure design for proper incentive alignment"""
        issues = []
        
        # The environment uses binary rewards (1.0 for complete success, 0.0 otherwise)
        # This is good design, but we need to verify no exploitation is possible
        
        # Check that targets are not trivially achievable (initial state == target state)
        initial_valve_states = world_state['valves']['states']
        target_pressures = world_state['hydraulics']['target_pressures']
        pump_speed = world_state['hydraulics']['pump_speed']
        reservoir_pressure = world_state['hydraulics']['reservoir_pressure']
        
        # Simulate initial pressures
        initial_pressures = self._simulate_hydraulics(initial_valve_states, pump_speed, reservoir_pressure)
        
        # Check if already at target (would give immediate reward without effort)
        if all(abs(initial_pressures[i] - target_pressures[i]) < 1e-6 for i in range(self.num_sensors)):
            issues.append("Initial state already matches target pressures - no challenge presented")
        
        # Check that targets require meaningful valve changes
        changes_needed = 0
        for valve_config in self._generate_valve_configurations():
            pressures = self._simulate_hydraulics(valve_config, pump_speed, reservoir_pressure)
            if all(abs(pressures[i] - target_pressures[i]) < 1e-3 for i in range(self.num_sensors)):
                # Count how many valves need to change
                changes = sum(1 for i in range(self.num_valves) if valve_config[i] != initial_valve_states[i])
                if changes_needed == 0 or changes < changes_needed:
                    changes_needed = changes
        
        if changes_needed == 0:
            issues.append("No valve changes needed to reach target - invalid level")
        elif changes_needed == 1:
            issues.append("Only 1 valve change needed - level may be too easy")
        
        return issues
    
    def _validate_target_reachability(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that target state is reachable through available actions"""
        issues = []
        
        initial_valve_states = world_state['valves']['states']
        target_pressures = world_state['hydraulics']['target_pressures']
        pump_speed = world_state['hydraulics']['pump_speed']
        reservoir_pressure = world_state['hydraulics']['reservoir_pressure']
        
        # Use breadth-first search to find if target is reachable within step limit
        solution_path = self._find_solution_path(initial_valve_states, target_pressures, pump_speed, reservoir_pressure, self.max_steps)
        
        if solution_path is None:
            issues.append("No solution path found within step limit using BFS")
        else:
            if len(solution_path) > self.max_steps * 0.8:  # Solution uses >80% of available steps
                issues.append(f"Solution path requires {len(solution_path)} steps, leaving little room for exploration (max steps: {self.max_steps})")
        
        return issues
    
    def _get_all_achievable_pressure_profiles(self, pump_speed: float, reservoir_pressure: float) -> List[List[float]]:
        """Generate all possible sensor pressure profiles from all valve configurations"""
        profiles = []
        for valve_config in self._generate_valve_configurations():
            pressure_profile = self._simulate_hydraulics(valve_config, pump_speed, reservoir_pressure)
            profiles.append(pressure_profile)
        return profiles
    
    def _generate_valve_configurations(self):
        """Generate all possible valve configurations (2^9 = 512 combinations)"""
        for i in range(2 ** self.num_valves):
            config = []
            for j in range(self.num_valves):
                config.append(bool(i & (1 << j)))
            yield config
    
    def _simulate_hydraulics(self, valve_states: List[bool], pump_speed: float, reservoir_pressure: float) -> List[float]:
        """Simulate hydraulic system to get sensor readings - matches environment logic"""
        base_pressure = reservoir_pressure
        pipe_pressures = []
        
        # Calculate pressure for each pipe section based on valve states and network topology
        for i in range(12):
            pressure_modifier = 1.0
            
            # Apply valve influence based on network topology
            for j, valve_open in enumerate(valve_states):
                if valve_open:
                    # Open valve increases pressure in downstream pipes
                    if (i + j) % 3 == 0:
                        pressure_modifier += 0.1
                else:
                    # Closed valve creates backpressure
                    if (i + j) % 4 == 0:
                        pressure_modifier += 0.05
            
            # Add pump influence
            pump_influence = (pump_speed / 1500.0) * (1.0 + 0.1 * (i % 3))
            
            pipe_pressure = base_pressure * pressure_modifier * pump_influence
            pipe_pressures.append(pipe_pressure)
        
        # Extract sensor readings from specific pipe locations
        sensor_readings = [
            pipe_pressures[2],   # Sensor 1 at pipe 2
            pipe_pressures[5],   # Sensor 2 at pipe 5  
            pipe_pressures[8],   # Sensor 3 at pipe 8
            pipe_pressures[11]   # Sensor 4 at pipe 11
        ]
        
        return sensor_readings
    
    def _estimate_min_steps_to_solution(self, initial_valves: List[bool], target_pressures: List[float], 
                                      pump_speed: float, reservoir_pressure: float) -> int:
        """Estimate minimum steps needed to reach solution"""
        # Find solution valve configuration
        solution_config = None
        for valve_config in self._generate_valve_configurations():
            pressures = self._simulate_hydraulics(valve_config, pump_speed, reservoir_pressure)
            if all(abs(pressures[i] - target_pressures[i]) < 1e-3 for i in range(self.num_sensors)):
                solution_config = valve_config
                break
        
        if solution_config is None:
            return float('inf')
        
        # Count minimum valve changes needed (Hamming distance)
        changes_needed = sum(1 for i in range(self.num_valves) if initial_valves[i] != solution_config[i])
        return changes_needed
    
    def _find_solution_path(self, initial_valves: List[bool], target_pressures: List[float],
                          pump_speed: float, reservoir_pressure: float, max_steps: int) -> Optional[List[int]]:
        """Use BFS to find shortest solution path"""
        from collections import deque
        
        # Convert valve state to tuple for hashing
        def valves_to_tuple(valves):
            return tuple(valves)
        
        # Check if state matches target
        def is_target_state(valves):
            pressures = self._simulate_hydraulics(valves, pump_speed, reservoir_pressure)
            return all(abs(pressures[i] - target_pressures[i]) < 1e-6 for i in range(self.num_sensors))
        
        if is_target_state(initial_valves):
            return []
        
        queue = deque([(initial_valves, [])])
        visited = {valves_to_tuple(initial_valves)}
        
        while queue:
            current_valves, path = queue.popleft()
            
            if len(path) >= max_steps:
                continue
            
            # Try toggling each valve
            for valve_id in range(self.num_valves):
                new_valves = current_valves[:]
                new_valves[valve_id] = not new_valves[valve_id]
                new_valves_tuple = valves_to_tuple(new_valves)
                
                if new_valves_tuple in visited:
                    continue
                
                visited.add(new_valves_tuple)
                new_path = path + [valve_id]
                
                if is_target_state(new_valves):
                    return new_path
                
                if len(new_path) < max_steps:
                    queue.append((new_valves, new_path))
        
        return None

def validate_generated_world(world_path: str, config_path: str) -> Tuple[bool, List[str]]:
    """Main validation function"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validator = PressureValveValidator(config)
    return validator.validate_world(world_path)

# Example usage:
# is_valid, issues = validate_generated_world("./levels/world_123.yaml", "./config.yaml")
# if not is_valid:
#     print("Validation failed:")
#     for issue in issues:
#         print(f"  - {issue}")