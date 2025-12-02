from base.env.base_env import SkinEnv
from env_obs import RadiusObserver
from env_generate import BackwardsValleyFarmGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy
import numpy as np

class BackwardsValleyFarmEnv(SkinEnv):
    def __init__(self, env_id: str):
        obs_policy = RadiusObserver(radius=2)
        super().__init__(env_id, obs_policy)
        self._base_state = None
    
    def _dsl_config(self):
        with open("./config.yaml", 'r') as f:
            self.configs = yaml.safe_load(f)
        self._base_state = deepcopy(self.configs["state_template"])
    
    def reset(self, mode: str = "generate", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        else:
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        
        if "max_steps" in self._state["globals"]:
            self.configs["termination"]["max_steps"] = self._state["globals"]["max_steps"]
        
        self._t = 0
        self._history = []
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        file_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"World file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        required_keys = ["globals", "agent", "tiles", "objects"]
        for key in required_keys:
            if key not in world_state:
                raise ValueError(f"Invalid world file: missing key '{key}'")
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = BackwardsValleyFarmGenerator(self.env_id, self.configs["generator"])
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        if action_name in ["MoveN", "MoveS", "MoveE", "MoveW"]:
            self._handle_movement(action_name)
        elif action_name == "Wait":
            pass
        elif action_name in ["UseWateringCan", "SpreadFertilizer", "Feed", "CleanPen", "Compliment", "Insult"]:
            self._handle_interaction(action_name)
        
        self._global_tick()
        self._handle_automatic_events()
        
        return self._state
    
    def _handle_movement(self, action_name: str):
        current_pos = self._state["agent"]["pos"]
        x, y = current_pos
        
        if action_name == "MoveN":
            new_pos = [x, y - 1]
            facing = "N"
        elif action_name == "MoveS":
            new_pos = [x, y + 1]
            facing = "S"
        elif action_name == "MoveE":
            new_pos = [x + 1, y]
            facing = "E"
        elif action_name == "MoveW":
            new_pos = [x - 1, y]
            facing = "W"
        
        map_size = self._state["tiles"]["size"]
        if (0 <= new_pos[0] < map_size[0] and 0 <= new_pos[1] < map_size[1]):
            blocked = False
            for fence in self._state["objects"]["fences"]:
                if fence["pos"] == new_pos:
                    blocked = True
                    break
            
            if not blocked:
                self._state["agent"]["pos"] = new_pos
        
        self._state["agent"]["facing"] = facing
    
    def _handle_interaction(self, action_name: str):
        front_pos = self._get_front_position()
        
        if action_name in ["UseWateringCan", "SpreadFertilizer"]:
            for field in self._state["objects"]["fields"]:
                if field["pos"] == front_pos:
                    field["stage"] = "Seed"
                    field["interacted"] = True
                    break
        
        elif action_name in ["Feed", "CleanPen"]:
            for pen in self._state["objects"]["pens"]:
                if pen["pos"] == front_pos:
                    pen["health_state"] = "Weak"
                    pen["interacted"] = True
                    break
        
        elif action_name == "Compliment":
            for villager in self._state["objects"]["villagers"]:
                if villager["pos"] == front_pos:
                    mood_order = ["Friendly", "Neutral", "Hostile"]
                    current_idx = mood_order.index(villager["mood"])
                    new_idx = min(current_idx + 1, len(mood_order) - 1)
                    villager["mood"] = mood_order[new_idx]
                    break
        
        elif action_name == "Insult":
            for villager in self._state["objects"]["villagers"]:
                if villager["pos"] == front_pos:
                    mood_order = ["Hostile", "Neutral", "Friendly"]
                    current_idx = mood_order.index(villager["mood"])
                    new_idx = min(current_idx + 1, len(mood_order) - 1)
                    villager["mood"] = mood_order[new_idx]
                    break
    
    def _get_front_position(self):
        pos = self._state["agent"]["pos"]
        facing = self._state["agent"]["facing"]
        
        x, y = pos
        if facing == "N":
            return [x, y - 1]
        elif facing == "S":
            return [x, y + 1]
        elif facing == "E":
            return [x + 1, y]
        elif facing == "W":
            return [x - 1, y]
        
        return pos
    
    def _global_tick(self):
        stage_order = ["Seed", "Sprout", "Young", "HarvestReady"]
        for field in self._state["objects"]["fields"]:
            if not field.get("interacted", False):
                current_idx = stage_order.index(field["stage"])
                if current_idx < len(stage_order) - 1:
                    field["stage"] = stage_order[current_idx + 1]
            field["interacted"] = False
        
        health_order = ["Weak", "Okay", "Thriving"]
        for pen in self._state["objects"]["pens"]:
            if not pen.get("interacted", False):
                current_idx = health_order.index(pen["health_state"])
                if current_idx < len(health_order) - 1:
                    pen["health_state"] = health_order[current_idx + 1]
            pen["interacted"] = False
    
    def _handle_automatic_events(self):
        agent_pos = self._state["agent"]["pos"]
        
        for field in self._state["objects"]["fields"][:]:
            if field["pos"] == agent_pos and field["stage"] == "HarvestReady" and not field.get("harvested", False):
                field["harvested"] = True
                farm_value_increase = field["base_value"]
                self._state["globals"]["farm_value"] += farm_value_increase
                self._state["objects"]["fields"].remove(field)
                break
        
        for pen in self._state["objects"]["pens"]:
            if pen["health_state"] == "Thriving" and not pen.get("thriving_achieved", False):
                pen["thriving_achieved"] = True
                farm_value_increase = pen["base_value"]
                self._state["globals"]["farm_value"] += farm_value_increase
        
        for villager in self._state["objects"]["villagers"]:
            if villager["mood"] == "Friendly" and not villager.get("friendly_achieved", False):
                villager["friendly_achieved"] = True
                farm_value_increase = villager["base_value"]
                self._state["globals"]["farm_value"] += farm_value_increase
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        current_farm_value = self._state["globals"]["farm_value"]
        
        if self._history:
            previous_farm_value = self._history[-1]["globals"]["farm_value"]
        else:
            previous_farm_value = 0
        
        reward_value = current_farm_value - previous_farm_value
        
        if reward_value > 0:
            events.append("farm_value_increased")
        
        reward_info = {
            "farm_value": current_farm_value,
            "reward_delta": reward_value
        }
        
        return float(reward_value), events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        visible_tiles = omega["visible_tiles"]
        agent_info = omega["agent"]
        
        grid = {}
        for tile in visible_tiles:
            dx, dy = tile["dx"], tile["dy"]
            grid[(dx, dy)] = tile
        
        lines = []
        for dy in range(-2, 3):
            line = ""
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    line += "A"
                elif (dx, dy) in grid:
                    tile = grid[(dx, dy)]
                    if tile["type"] == "fence":
                        line += "#"
                    elif tile["type"] == "crop":
                        line += "C"
                    elif tile["type"] == "pen":
                        line += "P"
                    elif tile["type"] == "villager":
                        line += "V"
                    elif tile["type"] == "out_of_bounds":
                        line += "X"
                    else:
                        line += "."
                else:
                    line += "."
            lines.append(line)
        
        tiles_ascii = "\n".join(lines)
        
        template = self.configs["skin"]["template"]
        return template.format(
            t=omega["t"],
            max_steps=self.configs["termination"]["max_steps"],
            farm_value=omega["farm_value"],
            agent_pos=agent_info["pos"],
            agent_facing=agent_info["facing"],
            remaining=omega["remaining"],
            tiles_ascii=tiles_ascii
        )
    
    def done(self, state: Dict[str, Any] = None) -> bool:
        if state is None:
            state = self._state
        
        if self._t >= self.configs["termination"]["max_steps"]:
            return True
        
        if state["globals"]["farm_value"] >= 300:
            return True
        
        return False