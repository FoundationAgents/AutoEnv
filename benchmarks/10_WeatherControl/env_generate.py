import random
import yaml
import os
from typing import Dict, Any, Optional
from base.env.base_generator import WorldGenerator
from copy import deepcopy

class AtmosphereGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.state_template = config.get('state_template', {})
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = deepcopy(self.state_template)
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
            
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = deepcopy(base_state)
        pipeline = self.config.get('generator', {}).get('pipeline', [])
        
        for step in pipeline:
            step_name = step.get('name', '')
            args = step.get('args', {})
            
            if step_name == 'init_from_template':
                pass
            elif step_name == 'randomize_initial_conditions':
                self._randomize_initial_conditions(world_state, args)
            elif step_name == 'calculate_initial_csi':
                self._calculate_initial_csi(world_state)
            elif step_name == 'initialize_drift_directions':
                self._initialize_drift_directions(world_state)
                
        return world_state
    
    def _randomize_initial_conditions(self, world_state: Dict[str, Any], args: Dict[str, Any]):
        atmosphere = world_state.setdefault('atmosphere', {})
        
        csi_range = args.get('csi_range', [40, 60])
        temperature_range = args.get('temperature_range', [200, 400])
        humidity_range = args.get('humidity_range', [30, 70])
        pressure_range = args.get('pressure_range', [0.8, 1.2])
        cloud_range = args.get('cloud_range', [20, 80])
        storm_range = args.get('storm_range', [10, 50])
        solar_range = args.get('solar_range', [800, 1200])
        
        atmosphere['temperature'] = random.uniform(*temperature_range)
        atmosphere['humidity'] = random.uniform(*humidity_range)
        atmosphere['atmospheric_pressure'] = random.uniform(*pressure_range)
        atmosphere['cloud_coverage'] = random.uniform(*cloud_range)
        atmosphere['storm_energy'] = random.uniform(*storm_range)
        atmosphere['solar_flux'] = random.uniform(*solar_range)
    
    def _calculate_initial_csi(self, world_state: Dict[str, Any]):
        atmosphere = world_state.get('atmosphere', {})
        
        temp_norm = (atmosphere.get('temperature', 300) - 200) / 200
        humidity_norm = atmosphere.get('humidity', 50) / 100
        pressure_norm = atmosphere.get('atmospheric_pressure', 1.0)
        cloud_norm = atmosphere.get('cloud_coverage', 50) / 100
        storm_norm = (atmosphere.get('storm_energy', 30) - 10) / 40
        solar_norm = (atmosphere.get('solar_flux', 1000) - 800) / 400
        
        csi = 50 + 10 * (temp_norm - 0.5) + 8 * (humidity_norm - 0.5) + 12 * (pressure_norm - 1.0) + 6 * (cloud_norm - 0.5) + 4 * (storm_norm - 0.5) + 5 * (solar_norm - 0.5)
        atmosphere['climate_stability_index'] = max(10, min(90, csi))
    
    def _initialize_drift_directions(self, world_state: Dict[str, Any]):
        physics = world_state.setdefault('physics', {})
        drift_directions = {}
        
        for var in ['temperature', 'humidity', 'atmospheric_pressure', 'cloud_coverage', 'storm_energy', 'solar_flux']:
            drift_directions[var] = random.choice([-1, 1])
            
        physics['drift_directions'] = drift_directions
        physics['action_effects_queue'] = []
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        else:
            return f"world_{random.randint(10000, 99999)}"