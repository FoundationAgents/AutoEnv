import sys
sys.path.append("/Users/didi/Documents/GitHub/AutoEnv")
from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import random
import yaml
import os
from datetime import datetime
from copy import deepcopy

class ValleyFarmGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Load state template
        with open("./config.yaml", "r") as f:
            full_config = yaml.safe_load(f)
        
        base_state = deepcopy(full_config["state_template"])
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Add termination config
        world_state["termination"] = full_config["termination"]
        
        # Generate world ID and save
        world_id = self._generate_world_id(seed)
        self._save_world(world_state, world_id)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # Step 1: Place static structures
        self._place_static_structures(base_state)
        
        # Step 2: Populate entities
        self._populate_entities(base_state)
        
        return base_state
    
    def _place_static_structures(self, state: Dict[str, Any]):
        used_positions = set()
        
        # Reserve agent starting position
        agent_pos = tuple(state["agent"]["pos"])
        used_positions.add(agent_pos)
        
        # Place 4 crop fields
        crop_fields = []
        for i in range(4):
            pos = self._find_free_position(used_positions)
            crop_fields.append({
                "id": f"field_{i}",
                "pos": list(pos),
                "crop": {
                    "stage": "empty",
                    "waterings": 0
                }
            })
            used_positions.add(pos)
        state["objects"]["crop_fields"] = crop_fields
        
        # Place 3 barns
        barns = []
        for i in range(3):
            pos = self._find_free_position(used_positions)
            barns.append({
                "id": f"barn_{i}",
                "pos": list(pos),
                "animal": None  # Will be populated later
            })
            used_positions.add(pos)
        state["objects"]["barns"] = barns
        
        # Place 3 cottages
        cottages = []
        for i in range(3):
            pos = self._find_free_position(used_positions)
            cottages.append({
                "id": f"cottage_{i}",
                "pos": list(pos)
            })
            used_positions.add(pos)
        state["objects"]["cottages"] = cottages
        
        # Place 1 market
        pos = self._find_free_position(used_positions)
        state["objects"]["market"] = list(pos)
    
    def _populate_entities(self, state: Dict[str, Any]):
        # Assign animals to barns
        animal_species = ["cow", "sheep", "chicken"]
        for i, barn in enumerate(state["objects"]["barns"]):
            barn["animal"] = {
                "species": animal_species[i],
                "hungry": True,
                "timer": 0,
                "product_count": 0
            }
        
        # Create villagers
        villagers = []
        for i in range(3):
            cottage_pos = state["objects"]["cottages"][i]["pos"]
            relationship = random.randint(0, 8) * 5  # Multiple of 5, 0-40
            villagers.append({
                "id": f"villager_{i}",
                "pos": cottage_pos[:],  # Start at home
                "home_pos": cottage_pos[:],
                "relationship": relationship,
                "mood": "neutral"
            })
        state["objects"]["villagers"] = villagers
    
    def _find_free_position(self, used_positions: set) -> tuple:
        while True:
            x = random.randint(0, 14)
            y = random.randint(0, 14)
            pos = (x, y)
            if pos not in used_positions:
                return pos
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"_seed{seed}" if seed is not None else ""
        return f"valley_farm_{timestamp}{seed_str}"
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        os.makedirs("./levels", exist_ok=True)
        filepath = f"./levels/{world_id}.yaml"
        
        with open(filepath, "w") as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id