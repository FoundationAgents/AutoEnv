from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import FullObservationPolicy
from env_generate import PressureValveGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
import copy

class PressureValveEnv(SkinEnv):
    def __init__(self, env_id: int):
        self.hydraulic_simulator = HydraulicSimulator()
        obs_policy = FullObservationPolicy()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            # Default to generating a new world if no world_id provided
            world_id = self._generate_world(seed)
            
        self._state = self._load_world(world_id)
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
        generator = PressureValveGenerator(str(self.env_id), self.configs)
        world_id = generator.generate(seed)
        return world_id
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action', '')
        params = action.get('params', {})
        
        # Store previous state in history
        self._history.append(copy.deepcopy(self._state))
        
        if action_name == "TOGGLE_VALVE":
            valve_id = params.get('valve_id', 0)
            if 0 <= valve_id < 9:
                # Toggle the valve state
                self._state['valves']['states'][valve_id] = not self._state['valves']['states'][valve_id]
                self._last_action_result = f"Toggled valve {valve_id}"
        elif action_name == "NO_OP":
            self._last_action_result = "No operation performed"
        
        # Recalculate hydraulic pressures
        pipe_pressures, sensor_readings = self.hydraulic_simulator.calculate_steady_state(
            self._state['valves']['states'],
            self._state['hydraulics']['pump_speed'],
            self._state['hydraulics']['reservoir_pressure']
        )
        
        self._state['hydraulics']['pipe_pressures'] = pipe_pressures
        self._state['hydraulics']['sensor_readings'] = sensor_readings
        
        # Decrement step count
        self._state['agent']['step_count'] -= 1
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        sensor_readings = self._state['hydraulics']['sensor_readings']
        target_pressures = self._state['hydraulics']['target_pressures']
        
        # Check if all sensors match targets exactly (zero tolerance)
        all_match = True
        for i in range(4):
            if abs(sensor_readings[i] - target_pressures[i]) > 1e-6:  # Essentially zero tolerance
                all_match = False
                break
        
        if all_match:
            return 1.0, ["pressure_target_achieved"], {"perfect_match": True}
        else:
            return 0.0, [], {"perfect_match": False}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        valve_states_str = "".join(["1" if v else "0" for v in omega['valve_states']])
        
        skin_output = f"=== Pressure Valve Control System ===\n"
        skin_output += f"Step: {omega['t']}/{self.configs['termination']['max_steps']} | Remaining: {omega['step_count']}\n\n"
        skin_output += f"Valve States: {valve_states_str}\n\n"
        skin_output += f"Sensor Readings vs Targets:\n"
        
        for i in range(4):
            skin_output += f"Sensor {i+1}: {omega['sensor_readings'][i]:.2f} kPa (target: {omega['target_pressures'][i]:.2f} kPa)\n"
        
        skin_output += f"\nPipe Section Pressures: {[f'{p:.2f}' for p in omega['pipe_pressures']]}\n\n"
        skin_output += f"Available Actions: NO_OP(), TOGGLE_VALVE(valve_id: 0-8)"
        
        return skin_output
    
    def done(self, state=None) -> bool:
        # Check if max steps reached
        if self._state['agent']['step_count'] <= 0:
            return True
            
        # Check if all sensors match targets
        sensor_readings = self._state['hydraulics']['sensor_readings']
        target_pressures = self._state['hydraulics']['target_pressures']
        
        all_match = True
        for i in range(4):
            if abs(sensor_readings[i] - target_pressures[i]) > 1e-6:
                all_match = False
                break
                
        return all_match

class HydraulicSimulator:
    def calculate_steady_state(self, valve_states: list, pump_speed: float, reservoir_pressure: float):
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