from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy
import time

class SpectrumWorldGenerator(WorldGenerator):
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
        world_state = deepcopy(base_state)
        
        material_signatures = self._generate_reference_library()
        world_state['reference_library']['material_signatures'] = material_signatures
        
        target_material = random.randint(0, 9)
        world_state['sample']['true_material_id'] = target_material
        
        self._validate_spectral_consistency(material_signatures)
        
        return world_state
    
    def _generate_reference_library(self) -> list:
        materials = [
            "water_gel", "crystalline_silica", "metallic_copper", "organic_dye_red", 
            "organic_dye_blue", "polymer_plastic", "calcium_carbonate", "titanium_oxide",
            "carbon_black", "liquid_mercury"
        ]
        
        signatures = []
        
        for material in materials:
            if material == "water_gel":
                fluorescence = [0.1, 0.8, 0.3, 0.2, 0.1]
                reflection = [0.2, 0.6, 0.7, 0.5, 0.1]
            elif material == "crystalline_silica":
                fluorescence = [0.0, 0.1, 0.1, 0.1, 0.0]
                reflection = [0.3, 0.7, 0.8, 0.6, 0.4]
            elif material == "metallic_copper":
                fluorescence = [0.0, 0.0, 0.1, 0.1, 0.0]
                reflection = [0.4, 0.6, 0.9, 0.8, 0.7]
            elif material == "organic_dye_red":
                fluorescence = [0.9, 0.2, 0.1, 0.8, 0.1]
                reflection = [0.1, 0.2, 0.3, 0.7, 0.2]
            elif material == "organic_dye_blue":
                fluorescence = [0.8, 0.9, 0.3, 0.1, 0.1]
                reflection = [0.1, 0.7, 0.4, 0.2, 0.1]
            elif material == "polymer_plastic":
                fluorescence = [0.2, 0.3, 0.2, 0.2, 0.1]
                reflection = [0.5, 0.6, 0.7, 0.6, 0.5]
            elif material == "calcium_carbonate":
                fluorescence = [0.1, 0.1, 0.0, 0.0, 0.0]
                reflection = [0.8, 0.9, 0.9, 0.8, 0.7]
            elif material == "titanium_oxide":
                fluorescence = [0.0, 0.0, 0.0, 0.0, 0.0]
                reflection = [0.9, 0.9, 0.9, 0.9, 0.8]
            elif material == "carbon_black":
                fluorescence = [0.0, 0.0, 0.0, 0.0, 0.0]
                reflection = [0.1, 0.1, 0.1, 0.1, 0.1]
            elif material == "liquid_mercury":
                fluorescence = [0.0, 0.0, 0.0, 0.0, 0.0]
                reflection = [0.8, 0.9, 0.9, 0.8, 0.7]
            
            signature = fluorescence + reflection
            signatures.append(signature)
        
        return signatures
    
    def _validate_spectral_consistency(self, signatures: list) -> bool:
        for signature in signatures:
            if len(signature) != 10:
                raise ValueError("Each signature must have exactly 10 values")
            for value in signature:
                if not (0.0 <= value <= 1.0):
                    raise ValueError("All spectral values must be between 0.0 and 1.0")
        return True
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = str(int(time.time()))
        seed_str = str(seed) if seed is not None else "noseed"
        return f"spectrum_world_{timestamp}_{seed_str}"