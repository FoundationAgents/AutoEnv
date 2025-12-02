from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import numpy as np
import yaml
import os
from copy import deepcopy
import time

class BackwardsValleyFarmGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        world_id = self._generate_world_id(seed)
        with open("config.yaml", "r") as f:
            full_config = yaml.safe_load(f)
        base_state = deepcopy(full_config["state_template"])
        world_state = self._execute_pipeline(base_state, seed)
        self._save_world(world_state, world_id)
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)
        
        pipeline = self.config["pipeline"]
        state = base_state
        
        for step in pipeline:
            if step["name"] == "init_from_template":
                continue
            elif step["name"] == "place_zones":
                state = self._place_zones(state, step["args"])
            elif step["name"] == "populate_entities":
                state = self._populate_entities(state, step["args"])
            elif step["name"] == "assign_initial_states":
                state = self._assign_initial_states(state, step["args"])
            elif step["name"] == "place_agent":
                state = self._place_agent(state, step["args"])
        
        return state
    
    def _place_zones(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        map_size = state["tiles"]["size"]
        
        crop_zone = [(x, y) for x in range(2, 6) for y in range(2, 6)]
        pen_zone = [(x, y) for x in range(6, 9) for y in range(2, 5)]
        village_zone = [(x, y) for x in range(2, 5) for y in range(6, 9)]
        
        fences = []
        fence_positions = [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 1), (6, 1), (6, 6),
            (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),
            (9, 2), (9, 3), (9, 4),
            (1, 9), (2, 9), (3, 9), (4, 9), (5, 9)
        ]
        
        for pos in fence_positions:
            if 0 <= pos[0] < map_size[0] and 0 <= pos[1] < map_size[1]:
                fences.append({"pos": [pos[0], pos[1]]})
        
        state["objects"]["fences"] = fences
        state["_crop_zone"] = crop_zone
        state["_pen_zone"] = pen_zone
        state["_village_zone"] = village_zone
        
        return state
    
    def _populate_entities(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        crop_types = args["crop_types"]
        animal_types = args["animal_types"]
        
        fields = []
        available_positions = list(range(len(state["_crop_zone"])))
        np.random.shuffle(available_positions)
        crop_positions = available_positions[:min(12, len(state["_crop_zone"]))]
        for i in crop_positions:
            pos = state["_crop_zone"][i]
            crop_type = crop_types[np.random.randint(len(crop_types))]
            fields.append({
                "pos": [pos[0], pos[1]],
                "crop_type": crop_type,
                "stage": "Seed",
                "base_value": {"wheat": 2, "corn": 3, "rice": 4, "pumpkin": 5}[crop_type],
                "harvested": False
            })
        
        pens = []
        available_pen_positions = list(range(len(state["_pen_zone"])))
        np.random.shuffle(available_pen_positions)
        pen_positions = available_pen_positions[:min(4, len(state["_pen_zone"]))]
        for i in pen_positions:
            pos = state["_pen_zone"][i]
            animal_type = animal_types[np.random.randint(len(animal_types))]
            pens.append({
                "pos": [pos[0], pos[1]],
                "animal_type": animal_type,
                "health_state": "Okay",
                "base_value": {"cow": 1, "sheep": 2, "pig": 3}[animal_type],
                "thriving_achieved": False
            })
        
        villagers = []
        available_house_positions = list(range(len(state["_village_zone"])))
        np.random.shuffle(available_house_positions)
        house_positions = available_house_positions[:min(6, len(state["_village_zone"]))]
        for i in house_positions:
            pos = state["_village_zone"][i]
            villagers.append({
                "pos": [pos[0], pos[1]],
                "mood": "Neutral",
                "base_value": 4,
                "friendly_achieved": False
            })
        
        state["objects"]["fields"] = fields
        state["objects"]["pens"] = pens
        state["objects"]["villagers"] = villagers
        
        return state
    
    def _assign_initial_states(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        crop_stages = args["crop_stages"]
        animal_states = args["animal_states"]
        moods = args["moods"]
        
        for field in state["objects"]["fields"]:
            field["stage"] = crop_stages[np.random.randint(len(crop_stages))]
        
        for pen in state["objects"]["pens"]:
            pen["health_state"] = animal_states[np.random.randint(len(animal_states))]
        
        for villager in state["objects"]["villagers"]:
            villager["mood"] = moods[np.random.randint(len(moods))]
        
        return state
    
    def _place_agent(self, state: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
        map_size = state["tiles"]["size"]
        occupied_positions = set()
        
        for obj_list in state["objects"].values():
            for obj in obj_list:
                if "pos" in obj:
                    occupied_positions.add(tuple(obj["pos"]))
        
        empty_positions = []
        for x in range(map_size[0]):
            for y in range(map_size[1]):
                if (x, y) not in occupied_positions:
                    empty_positions.append([x, y])
        
        if empty_positions:
            agent_pos = empty_positions[np.random.randint(len(empty_positions))]
            state["agent"]["pos"] = agent_pos
        else:
            state["agent"]["pos"] = [0, 0]
        
        state["agent"]["facing"] = ["N", "S", "E", "W"][np.random.randint(4)]
        
        del state["_crop_zone"]
        del state["_pen_zone"] 
        del state["_village_zone"]
        
        return state
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        levels_dir = "./levels"
        os.makedirs(levels_dir, exist_ok=True)
        
        file_path = os.path.join(levels_dir, f"{world_id}.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"world_{seed}_{timestamp}"
        else:
            return f"world_{timestamp}"