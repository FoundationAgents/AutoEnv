from base.env.base_generator import WorldGenerator
import yaml
import random
import os
from typing import Dict, Any, Optional
from copy import deepcopy

class TerraformingGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = deepcopy(self.config['state_template'])
        world_state = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = deepcopy(base_state)
        generator_config = self.config.get('generator', {})
        
        variance = generator_config.get('pipeline', [{}])[1].get('args', {})
        atm_var = variance.get('atmosphere_variance', 0.05)  # Less variance
        geo_var = variance.get('geology_variance', 0.1)   # Less variance
        res_var = variance.get('resources_variance', 0.1)  # Less variance
        
        state['atmosphere']['co2_pct'] = max(70.0, min(85.0, 
            state['atmosphere']['co2_pct'] * (1 + random.uniform(-atm_var, atm_var))))
        state['atmosphere']['temperature'] = max(-70.0, min(-30.0,
            state['atmosphere']['temperature'] * (1 + random.uniform(-atm_var, atm_var))))
        state['atmosphere']['pressure'] = max(0.1, min(0.8,
            state['atmosphere']['pressure'] * (1 + random.uniform(-atm_var, atm_var))))
        
        state['lithosphere']['tectonic_stress'] = max(40.0, min(80.0,
            state['lithosphere']['tectonic_stress'] * (1 + random.uniform(-geo_var, geo_var))))
        
        state['hydrosphere']['subsurface_ice_pct'] = max(25.0, min(50.0,
            state['hydrosphere']['subsurface_ice_pct'] * (1 + random.uniform(-res_var, res_var))))
        state['hydrosphere']['ph_level'] = max(2.5, min(4.0,
            state['hydrosphere']['ph_level'] * (1 + random.uniform(-res_var, res_var))))
        
        state['infrastructure']['energy_reserves'] = max(1400.0, min(2000.0,
            state['infrastructure']['energy_reserves'] * (1 + random.uniform(-res_var, res_var))))
        
        state['global_metrics']['instability_index'] = max(10.0, min(25.0,
            state['global_metrics']['instability_index'] * (1 + random.uniform(-0.3, 0.3))))
        
        state['biosphere_seeds']['dormant_microbes'] = max(80.0, min(120.0,
            state['biosphere_seeds']['dormant_microbes'] * (1 + random.uniform(-res_var, res_var))))
        state['biosphere_seeds']['dormant_flora'] = max(30.0, min(80.0,
            state['biosphere_seeds']['dormant_flora'] * (1 + random.uniform(-res_var, res_var))))
        
        return state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"planet_{seed}"
        else:
            return f"planet_{random.randint(1000, 9999)}"