from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import random
import yaml
import os
import time

class EntropyWorldGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = self.config["state_template"]
        world_state = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = {
            "globals": {
                "step": 0,
                "global_order_score": 0,
                "entropy_tokens": 200
            },
            "domains": {}
        }
        
        domain_names = ["thermal_grid", "data_archive", "crystal_lattice", "bio_habitat"]
        
        for domain_name in domain_names:
            order = random.randint(40, 60)
            energy = random.randint(80, 120)
            chaos = random.randint(20, 40)
            
            world_state["domains"][domain_name] = {
                "order": order,
                "energy": energy,
                "chaos": chaos,
                "locked": False
            }
        
        total_order = sum(domain["order"] for domain in world_state["domains"].values())
        world_state["globals"]["global_order_score"] = total_order - 200
        
        for domain in world_state["domains"].values():
            if domain["chaos"] > 70:
                domain["chaos"] = 70
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000000)
        if seed is not None:
            return f"entropy_world_s{seed}_{timestamp}"
        else:
            return f"entropy_world_{timestamp}"