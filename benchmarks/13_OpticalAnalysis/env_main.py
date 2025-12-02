from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import SpectrumObservationPolicy
from env_generate import SpectrumWorldGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class LightSpectrumEnv(SkinEnv):
    def __init__(self, env_id: int = 0):
        obs_policy = SpectrumObservationPolicy()
        super().__init__(env_id, obs_policy)
        self.generator = None
        
    def _dsl_config(self):
        with open("./config.yaml", 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = SpectrumWorldGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
            
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World file not found: {world_path}")
            
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed=seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        action_name = action.get('action')
        
        if action_name in ['EmitUV', 'EmitBlue', 'EmitGreen', 'EmitRed', 'EmitIR']:
            wavelength_map = {
                'EmitUV': 0, 'EmitBlue': 1, 'EmitGreen': 2, 'EmitRed': 3, 'EmitIR': 4
            }
            band_idx = wavelength_map[action_name]
            
            if not self._state['sample']['illuminated_bands'][band_idx]:
                self._state['sample']['illuminated_bands'][band_idx] = True
                self._recalculate_spectrum()
            
            self._last_action_result = f"Illuminating {action_name.replace('Emit', '')} wavelength"
            
        elif action_name.startswith('Declare_'):
            material_id = int(action_name.split('_')[1])
            self._state['agent']['has_declared'] = True
            self._state['agent']['declared_material'] = material_id
            self._last_action_result = f"Declared material {material_id}"
        
        self._state['agent']['current_step'] += 1
        return self._state
    
    def _recalculate_spectrum(self):
        true_material = self._state['sample']['true_material_id']
        reference_signature = self._state['reference_library']['material_signatures'][true_material]
        
        fluorescence = [0.0] * 5
        reflection = [0.0] * 5
        
        for i, illuminated in enumerate(self._state['sample']['illuminated_bands']):
            if illuminated:
                fluorescence[i] = max(fluorescence[i], reference_signature[i])
                reflection[i] = max(reflection[i], reference_signature[i + 5])
        
        self._state['sample']['observed_spectrum']['fluorescence'] = fluorescence
        self._state['sample']['observed_spectrum']['reflection'] = reflection
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_value = 0.0
        reward_info = {}
        
        if self._state['agent']['has_declared']:
            declared = self._state['agent']['declared_material']
            true_material = self._state['sample']['true_material_id']
            
            if declared == true_material:
                reward_value = 1.0
                events.append('correct_declaration')
                reward_info['result'] = 'correct'
            else:
                reward_value = 0.0
                events.append('incorrect_declaration')
                reward_info['result'] = 'incorrect'
        elif self._state['agent']['current_step'] >= self._state['globals']['max_steps']:
            reward_value = 0.0
            events.append('timeout')
            reward_info['result'] = 'timeout'
        
        return reward_value, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        illuminated = omega['illuminated_bands']
        fluor = omega['observed_spectrum']['fluorescence']
        refl = omega['observed_spectrum']['reflection']
        
        skin_text = f"""=== Light Spectrum Analysis Lab ===
Step: {omega['t']}/{omega['max_steps']}

Illuminated Bands: UV={illuminated[0]} Blue={illuminated[1]} Green={illuminated[2]} Red={illuminated[3]} IR={illuminated[4]}

Observed Spectrum:
Fluorescence: UV={fluor[0]:.3f} Blue={fluor[1]:.3f} Green={fluor[2]:.3f} Red={fluor[3]:.3f} IR={fluor[4]:.3f}
Reflection:   UV={refl[0]:.3f} Blue={refl[1]:.3f} Green={refl[2]:.3f} Red={refl[3]:.3f} IR={refl[4]:.3f}

Available Actions:
Illumination: EmitUV, EmitBlue, EmitGreen, EmitRed, EmitIR
Declaration: Declare_0 through Declare_9

Reference Library Available: 10 material signatures with full spectral data"""
        
        return skin_text
    
    def done(self, state=None) -> bool:
        return (self._state['agent']['has_declared'] or 
                self._state['agent']['current_step'] >= self._state['globals']['max_steps'])