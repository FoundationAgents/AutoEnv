from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
from copy import deepcopy
import random
import yaml
import os
import time

class BizarroLabGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.state_template = config.get("state_template", {})
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = deepcopy(self.state_template)
        complete_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.safe_dump(complete_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        pipeline = self.config.get("generator", {}).get("pipeline", [])
        
        for step in pipeline:
            if step["name"] == "init_from_template":
                pass
            elif step["name"] == "assign_target_compound":
                compound_pool = step["args"]["compound_pool"]
                base_state["globals"]["target_compound"] = random.choice(compound_pool)
            elif step["name"] == "set_random_seed":
                if seed is not None:
                    base_state["globals"]["seed"] = seed
        
        return base_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"world_{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"