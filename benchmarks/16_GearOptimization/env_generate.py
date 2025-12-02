from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional

class GearWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
            
        world_id = self._generate_world_id(seed)
        
        base_state = self.config.get("state_template", {})
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
            
        self._save_world(world_state, save_path)
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = {
            "globals": base_state.get("globals", {}).copy(),
            "agent": base_state.get("agent", {}).copy(),
            "gear_system": base_state.get("gear_system", {}).copy()
        }
        
        num_gears = 10
        min_teeth = 6
        max_teeth = 60
        available_gears = [random.randint(min_teeth, max_teeth) for _ in range(num_gears)]
        world_state["gear_system"]["available_gears"] = available_gears
        
        min_ratio = 0.1
        max_ratio = 10.0
        target_ma = random.uniform(min_ratio, max_ratio)
        world_state["gear_system"]["target_ma"] = target_ma
        
        if not self._validate_solvability(available_gears, target_ma, world_state["globals"]["tolerance"]):
            target_ma = self._find_achievable_target(available_gears)
            world_state["gear_system"]["target_ma"] = target_ma
            
        return world_state
    
    def _validate_solvability(self, available_gears, target_ma, tolerance):
        def calculate_ma(chain):
            if len(chain) == 0:
                return 1.0
            ma = 1.0
            for i in range(0, len(chain) - 1, 2):
                if i + 1 < len(chain):
                    ma *= chain[i] / chain[i + 1]
            return ma
        
        def check_combinations(current_chain, depth):
            if depth > 6:
                return False
            
            current_ma = calculate_ma(current_chain)
            if abs(current_ma - target_ma) / target_ma <= tolerance:
                return True
                
            if depth < 6:
                for gear in available_gears:
                    if check_combinations(current_chain + [gear], depth + 1):
                        return True
            return False
        
        return check_combinations([], 0)
    
    def _find_achievable_target(self, available_gears):
        def calculate_ma(chain):
            if len(chain) == 0:
                return 1.0
            ma = 1.0
            for i in range(0, len(chain) - 1, 2):
                if i + 1 < len(chain):
                    ma *= chain[i] / chain[i + 1]
            return ma
        
        possible_targets = []
        for i in range(len(available_gears)):
            for j in range(len(available_gears)):
                if i != j:
                    ma = available_gears[i] / available_gears[j]
                    possible_targets.append(ma)
                    
                    for k in range(len(available_gears)):
                        for l in range(len(available_gears)):
                            if k != l:
                                ma2 = ma * (available_gears[k] / available_gears[l])
                                possible_targets.append(ma2)
        
        return random.choice(possible_targets) if possible_targets else 1.0
    
    def _save_world(self, world_state: Dict[str, Any], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_{seed}"
        else:
            return f"world_{random.randint(10000, 99999)}"