from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from base.env.base_observation import ObservationPolicy
from base.env.base_generator import WorldGenerator
from env_obs import GearObservationPolicy
from env_generate import GearWorldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class GearRatioEnv(SkinEnv):
    def __init__(self, env_id: int):
        self.obs_policy = GearObservationPolicy("full", {})
        super().__init__(env_id, self.obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        try:
            with open(config_path, 'r') as f:
                self.configs = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            
        if world_id is None:
            raise ValueError("world_id must be provided for load mode")
            
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
            return world_state
        except FileNotFoundError:
            raise FileNotFoundError(f"World file not found at {world_path}")
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = GearWorldGenerator(str(self.env_id), self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        action_name = action.get("action", "")
        params = action.get("params", {})
        
        self._last_action_result = {"action": action_name, "success": True, "message": ""}
        
        if action_name == "PlaceGear":
            gear_index = params.get("gear_index", -1)
            available_gears = self._state["gear_system"]["available_gears"]
            
            if 0 <= gear_index < len(available_gears):
                gear_teeth = available_gears[gear_index]
                self._state["gear_system"]["gear_chain"].append(gear_teeth)
                self._state["gear_system"]["current_ma"] = self._calculate_mechanical_advantage(
                    self._state["gear_system"]["gear_chain"]
                )
                self._last_action_result["message"] = f"Placed gear with {gear_teeth} teeth"
            else:
                self._last_action_result["success"] = False
                self._last_action_result["message"] = f"Invalid gear index: {gear_index}"
                
        elif action_name == "RemoveLast":
            if len(self._state["gear_system"]["gear_chain"]) > 0:
                removed_gear = self._state["gear_system"]["gear_chain"].pop()
                self._state["gear_system"]["current_ma"] = self._calculate_mechanical_advantage(
                    self._state["gear_system"]["gear_chain"]
                )
                self._last_action_result["message"] = f"Removed gear with {removed_gear} teeth"
            else:
                self._last_action_result["success"] = False
                self._last_action_result["message"] = "No gears to remove"
                
        elif action_name == "Finish":
            self._state["gear_system"]["episode_finished"] = True
            current_ma = self._state["gear_system"]["current_ma"]
            target_ma = self._state["gear_system"]["target_ma"]
            tolerance = self._state["globals"]["tolerance"]
            
            error_ratio = abs(current_ma - target_ma) / target_ma
            if error_ratio <= tolerance:
                self._state["gear_system"]["success"] = True
                self._last_action_result["message"] = f"Success! MA: {current_ma:.4f}, Target: {target_ma:.4f}"
            else:
                self._state["gear_system"]["success"] = False
                self._last_action_result["message"] = f"Failed. MA: {current_ma:.4f}, Target: {target_ma:.4f}, Error: {error_ratio:.1%}"
                
        elif action_name == "Skip":
            self._last_action_result["message"] = "Skipped turn"
        else:
            self._last_action_result["success"] = False
            self._last_action_result["message"] = f"Unknown action: {action_name}"
        
        if self._state["agent"]["remaining_steps"] > 0:
            self._state["agent"]["remaining_steps"] -= 1
            
        return self._state
    
    def _calculate_mechanical_advantage(self, gear_chain: List[int]) -> float:
        if len(gear_chain) == 0:
            return 1.0
            
        ma = 1.0
        for i in range(0, len(gear_chain) - 1, 2):
            if i + 1 < len(gear_chain):
                ma *= gear_chain[i] / gear_chain[i + 1]
        
        return ma
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        action_name = action.get("action", "")
        
        if action_name == "Finish":
            if self._state["gear_system"]["success"]:
                return 1.0, ["finish_success"], {"success": True}
            else:
                return 0.0, ["finish_fail"], {"success": False}
        else:
            return 0.0, ["step"], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        max_steps = self._state["globals"]["max_steps"]
        remaining_steps = omega["remaining_steps"]
        target_ma = omega["target_ma"]
        current_ma = omega["current_ma"]
        tolerance = omega["tolerance"]
        available_gears = omega["available_gears"]
        gear_chain = omega["gear_chain"]
        t = omega["t"]
        
        chain_str = " -> ".join(map(str, gear_chain)) if gear_chain else "Empty"
        
        return f"""Step {t}/{max_steps} | Remaining: {remaining_steps}
Target MA: {target_ma:.4f} (±{tolerance:.1%})
Current MA: {current_ma:.4f}

Available Gears (tooth counts): {available_gears}
Current Chain: {chain_str} -> MA = {current_ma:.4f}

Actions: PlaceGear(i), RemoveLast(), Finish(), Skip()"""
    
    def done(self, state=None) -> bool:
        max_steps = self._state["globals"]["max_steps"]
        return (self._t >= max_steps or 
                self._state["agent"]["remaining_steps"] <= 0 or 
                self._state["gear_system"]["episode_finished"])