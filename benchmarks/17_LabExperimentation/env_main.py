from base.env.base_env import SkinEnv
from env_obs import FullLabObservation
from env_generate import BizarroLabGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os
import random
import math
from copy import deepcopy

class BizarroLabEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = FullLabObservation()
        super().__init__(env_id, obs_policy)
        self.generator = None
        self.inverted_reaction_table = {
            ("Acid", "Base"): {"product": "Xylene", "pH_change": 2.0, "temp_change": -10},
            ("Solvent", "Acid"): {"product": "Bizarrolene", "pH_change": -1.5, "temp_change": 8},
            ("Acid", "Solvent"): {"product": "Bizarrolene", "pH_change": -1.5, "temp_change": 8},
            ("Base", "Solvent"): {"product": "InvertedAcetate", "pH_change": 1.0, "temp_change": -5}
        }
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = BizarroLabGenerator("bizarro_lab_v1", self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id required for load mode")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self._t = 0
        self._history = []
        max_steps = self.configs["termination"]["max_steps"]
        if "termination" in self._state and "max_steps" in self._state["termination"]:
            max_steps = self._state["termination"]["max_steps"]
        self._state["globals"]["step_remaining"] = max_steps
        self._state["globals"]["submitted"] = False
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        if self.generator is None:
            self._dsl_config()
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        prev_state = deepcopy(self._state)
        self._history.append(prev_state)
        
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        self._last_action_result = {"success": False, "message": ""}
        
        if action_name == "AddReagent":
            beaker_id = params.get("beaker_id")
            reagent_name = params.get("reagent_name")
            volume_ml = params.get("volume_ml", 0)
            
            if (beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]) and
                volume_ml > 0 and reagent_name in [r["name"] for r in self._state["reagents_catalog"]]):
                beaker = self._state["beakers"][beaker_id]
                if beaker["volume_ml"] + volume_ml <= beaker["capacity_ml"]:
                    old_volume = beaker["volume_ml"]
                    new_volume = old_volume + volume_ml
                    
                    if old_volume == 0:
                        beaker["composition_pct"] = {reagent_name: 100.0}
                    else:
                        for compound in beaker["composition_pct"]:
                            beaker["composition_pct"][compound] = (beaker["composition_pct"][compound] * old_volume) / new_volume
                        
                        if reagent_name in beaker["composition_pct"]:
                            beaker["composition_pct"][reagent_name] += (volume_ml * 100.0) / new_volume
                        else:
                            beaker["composition_pct"][reagent_name] = (volume_ml * 100.0) / new_volume
                    
                    beaker["volume_ml"] = new_volume
                    self._last_action_result = {"success": True, "message": f"Added {volume_ml}ml of {reagent_name}"}
        
        elif action_name == "ToggleHotPlate":
            beaker_id = params.get("beaker_id")
            if beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]):
                self._state["equipment"]["hot_plates"][beaker_id] = not self._state["equipment"]["hot_plates"][beaker_id]
                self._last_action_result = {"success": True, "message": f"Toggled hot plate for beaker {beaker_id}"}
        
        elif action_name == "ToggleCoolingCoil":
            beaker_id = params.get("beaker_id")
            if beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]):
                self._state["equipment"]["cooling_coils"][beaker_id] = not self._state["equipment"]["cooling_coils"][beaker_id]
                self._last_action_result = {"success": True, "message": f"Toggled cooling coil for beaker {beaker_id}"}
        
        elif action_name == "SetStirSpeed":
            beaker_id = params.get("beaker_id")
            speed = params.get("speed", 0)
            if (beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]) and 
                0 <= speed <= 3):
                self._state["equipment"]["stir_speeds"][beaker_id] = speed
                self._last_action_result = {"success": True, "message": f"Set stir speed to {speed} for beaker {beaker_id}"}
        
        elif action_name == "Transfer":
            from_beaker_id = params.get("from_beaker_id")
            to_beaker_id = params.get("to_beaker_id")
            volume_ml = params.get("volume_ml", 0)
            
            if (from_beaker_id is not None and to_beaker_id is not None and
                0 <= from_beaker_id < len(self._state["beakers"]) and
                0 <= to_beaker_id < len(self._state["beakers"]) and
                from_beaker_id != to_beaker_id and volume_ml > 0):
                
                from_beaker = self._state["beakers"][from_beaker_id]
                to_beaker = self._state["beakers"][to_beaker_id]
                
                if (from_beaker["volume_ml"] >= volume_ml and 
                    to_beaker["volume_ml"] + volume_ml <= to_beaker["capacity_ml"]):
                    
                    from_old_volume = from_beaker["volume_ml"]
                    to_old_volume = to_beaker["volume_ml"]
                    to_new_volume = to_old_volume + volume_ml
                    
                    if to_old_volume == 0:
                        to_beaker["composition_pct"] = deepcopy(from_beaker["composition_pct"])
                    else:
                        for compound in from_beaker["composition_pct"]:
                            transferred_amount = (from_beaker["composition_pct"][compound] * volume_ml) / 100.0
                            if compound in to_beaker["composition_pct"]:
                                existing_amount = (to_beaker["composition_pct"][compound] * to_old_volume) / 100.0
                                to_beaker["composition_pct"][compound] = ((existing_amount + transferred_amount) * 100.0) / to_new_volume
                            else:
                                to_beaker["composition_pct"][compound] = (transferred_amount * 100.0) / to_new_volume
                        
                        for compound in to_beaker["composition_pct"]:
                            if compound not in from_beaker["composition_pct"]:
                                existing_amount = (to_beaker["composition_pct"][compound] * to_old_volume) / 100.0
                                to_beaker["composition_pct"][compound] = (existing_amount * 100.0) / to_new_volume
                    
                    from_beaker["volume_ml"] -= volume_ml
                    to_beaker["volume_ml"] = to_new_volume
                    
                    if from_beaker["volume_ml"] == 0:
                        from_beaker["composition_pct"] = {}
                    
                    self._last_action_result = {"success": True, "message": f"Transferred {volume_ml}ml"}
        
        elif action_name == "Discard":
            beaker_id = params.get("beaker_id")
            if beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]):
                beaker = self._state["beakers"][beaker_id]
                beaker["volume_ml"] = 0
                beaker["composition_pct"] = {}
                beaker["temperature_c"] = self._state["globals"]["ambient_temperature_c"]
                beaker["pH"] = 7.0
                self._last_action_result = {"success": True, "message": f"Discarded contents of beaker {beaker_id}"}
        
        elif action_name == "SubmitForAnalysis":
            beaker_id = params.get("beaker_id")
            if beaker_id is not None and 0 <= beaker_id < len(self._state["beakers"]):
                self._state["globals"]["submitted"] = True
                self._state["globals"]["submitted_beaker_id"] = beaker_id
                self._last_action_result = {"success": True, "message": f"Submitted beaker {beaker_id} for analysis"}
        
        elif action_name == "Wait":
            self._last_action_result = {"success": True, "message": "Waited"}
        
        self._apply_kinetic_evolution()
        self._state["globals"]["step_remaining"] -= 1
        
        return self._state
    
    def _apply_kinetic_evolution(self):
        for i, beaker in enumerate(self._state["beakers"]):
            if beaker["volume_ml"] > 0:
                temp_change = 0
                if self._state["equipment"]["hot_plates"][i]:
                    temp_change -= 5
                if self._state["equipment"]["cooling_coils"][i]:
                    temp_change += 5
                
                temp_change += random.randint(-1, 1)
                beaker["temperature_c"] += temp_change
                
                self._apply_reactions(beaker, i)
    
    def _apply_reactions(self, beaker: Dict[str, Any], beaker_id: int):
        stir_speed = self._state["equipment"]["stir_speeds"][beaker_id]
        reaction_rate = 0.1 * (stir_speed + 1)
        
        compounds = list(beaker["composition_pct"].keys())
        for i in range(len(compounds)):
            for j in range(i + 1, len(compounds)):
                comp1, comp2 = compounds[i], compounds[j]
                reaction_key = tuple(sorted([comp1, comp2]))
                
                if reaction_key in self.inverted_reaction_table:
                    reaction = self.inverted_reaction_table[reaction_key]
                    
                    min_pct = min(beaker["composition_pct"][comp1], beaker["composition_pct"][comp2])
                    reaction_amount = min_pct * reaction_rate
                    
                    if reaction_amount > 0.1:
                        beaker["composition_pct"][comp1] -= reaction_amount
                        beaker["composition_pct"][comp2] -= reaction_amount
                        
                        product = reaction["product"]
                        if product in beaker["composition_pct"]:
                            beaker["composition_pct"][product] += reaction_amount * 1.5
                        else:
                            beaker["composition_pct"][product] = reaction_amount * 1.5
                        
                        beaker["pH"] += reaction["pH_change"] * reaction_rate
                        beaker["temperature_c"] += reaction["temp_change"] * reaction_rate
                        
                        if beaker["composition_pct"][comp1] <= 0:
                            del beaker["composition_pct"][comp1]
                        if beaker["composition_pct"][comp2] <= 0:
                            del beaker["composition_pct"][comp2]
        
        total_pct = sum(beaker["composition_pct"].values())
        if total_pct > 100:
            for compound in beaker["composition_pct"]:
                beaker["composition_pct"][compound] = (beaker["composition_pct"][compound] / total_pct) * 100
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        if len(self._history) == 0:
            prev_max_purity = 0
        else:
            prev_max_purity = self._get_max_target_purity(self._history[-1])
        
        current_max_purity = self._get_max_target_purity(self._state)
        
        dense_reward = max(0, current_max_purity - prev_max_purity)
        if dense_reward > 0:
            events.append("improved_purity")
            reward_info["purity_improvement"] = dense_reward
        
        success_bonus = 0
        if action.get("action") == "SubmitForAnalysis" and self._state["globals"].get("submitted", False):
            beaker_id = self._state["globals"].get("submitted_beaker_id", 0)
            submitted_purity = self._get_beaker_target_purity(self._state["beakers"][beaker_id])
            
            if submitted_purity >= self._state["globals"]["target_purity"]:
                events.append("success")
                steps_used = self.configs["termination"]["max_steps"] - self._state["globals"]["step_remaining"]
                unused_fraction = (self.configs["termination"]["max_steps"] - steps_used) / self.configs["termination"]["max_steps"]
                
                success_bonus = (self.configs["reward"]["success_bonus"] + 
                               self.configs["reward"]["time_efficiency_factor"] * unused_fraction)
                reward_info["success_bonus"] = success_bonus
                reward_info["submitted_purity"] = submitted_purity
        
        total_reward = dense_reward + success_bonus
        reward_info["total_reward"] = total_reward
        
        return total_reward, events, reward_info
    
    def _get_max_target_purity(self, state: Dict[str, Any]) -> float:
        target_compound = state["globals"]["target_compound"]
        max_purity = 0
        
        for beaker in state["beakers"]:
            purity = self._get_beaker_target_purity(beaker, target_compound)
            max_purity = max(max_purity, purity)
        
        return max_purity
    
    def _get_beaker_target_purity(self, beaker: Dict[str, Any], target_compound: str = None) -> float:
        if target_compound is None:
            target_compound = self._state["globals"]["target_compound"]
        
        return beaker["composition_pct"].get(target_compound, 0.0) / 100.0
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        template = self.configs["skin"]["template"]
        
        beakers_str = ""
        for i, beaker in enumerate(omega["beakers"]):
            beakers_str += f"[{i}]: {beaker['volume_ml']}ml, {beaker['composition_pct']}, {beaker['temperature_c']}°C, pH {beaker['pH']:.1f}\n  "
        
        target_purity_percent = omega["globals"]["target_purity"] * 100
        
        rendered = template.format(
            t=omega["t"],
            termination=self.configs["termination"],
            globals=omega["globals"],
            target_purity_percent=target_purity_percent,
            beakers=beakers_str,
            equipment=omega["equipment"]
        )
        
        return rendered
    
    def done(self, state: Dict[str, Any] = None) -> bool:
        if state is None:
            state = self._state
        
        max_steps = self.configs["termination"]["max_steps"]
        if "termination" in state and "max_steps" in state["termination"]:
            max_steps = state["termination"]["max_steps"]
        
        return (self._t >= max_steps or 
                state["globals"].get("submitted", False) or 
                state["globals"]["step_remaining"] <= 0)