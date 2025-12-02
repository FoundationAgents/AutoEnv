from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import copy
import hashlib
import time

class PressureValveGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.state_template = config.get('state_template', {})
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = copy.deepcopy(self.state_template)
        world_state = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
            
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = copy.deepcopy(base_state)
        
        # Step 1: Generate target valve configuration that will give us our target pressures
        target_valve_config = self._generate_target_valve_config()
        
        # Step 2: Calculate what pressures this configuration achieves
        target_pipe_pressures, target_sensor_readings = self._simulate_pressures(
            target_valve_config,
            world_state['hydraulics']['pump_speed'],
            world_state['hydraulics']['reservoir_pressure']
        )
        
        # Step 3: Use these actual achievable pressures as our targets
        world_state['hydraulics']['target_pressures'] = target_sensor_readings
        
        # Step 4: Generate initial valve states that are different from target
        initial_valve_states = self._generate_initial_valves(target_valve_config)
        world_state['valves']['states'] = initial_valve_states
        
        # Step 5: Calculate initial pressures based on initial valve states
        initial_pipe_pressures, initial_sensor_readings = self._simulate_pressures(
            initial_valve_states,
            world_state['hydraulics']['pump_speed'],
            world_state['hydraulics']['reservoir_pressure']
        )
        world_state['hydraulics']['pipe_pressures'] = initial_pipe_pressures
        world_state['hydraulics']['sensor_readings'] = initial_sensor_readings
        
        return world_state
    
    def _generate_target_valve_config(self) -> list:
        """Generate a random valve configuration that will serve as our solution"""
        return [random.choice([True, False]) for _ in range(9)]
    
    def _generate_initial_valves(self, target_valve_config: list) -> list:
        """Generate initial valve states that differ from target by 2-5 valves"""
        initial_valves = target_valve_config[:]
        
        # Ensure we change at least 2 and at most 5 valves for reasonable difficulty
        num_changes = random.randint(2, 5)
        valve_indices = list(range(9))
        random.shuffle(valve_indices)
        
        # Change the selected valves
        for i in range(num_changes):
            valve_id = valve_indices[i]
            initial_valves[valve_id] = not initial_valves[valve_id]
        
        return initial_valves
    
    def _simulate_pressures(self, valve_states: list, pump_speed: float, reservoir_pressure: float):
        """Hydraulic simulation - must match the simulation in env_main.py exactly"""
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
        
        return pipe_pressures, sensor_readings
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = str(int(time.time() * 1000))
        if seed is not None:
            return f"world_{seed}_{timestamp}"
        else:
            return f"world_random_{timestamp}"
