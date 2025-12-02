from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import FullObservation
from env_generate import EntropyWorldGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os
import math
from copy import deepcopy

class EntropyReversalEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = FullObservation()
        super().__init__(env_id, obs_policy)
        self._dsl_config()
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            raise ValueError("world_id must be provided for load mode")
            
        self._state = self._load_world(world_id)
        self._t = self._state.get("globals", {}).get("step", 0)
        self._history = []
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = EntropyWorldGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        self._history.append(deepcopy(self._state))
        
        action_name = action["action"]
        params = action.get("params", {})
        
        if action_name == "inject_energy":
            target_domain = params.get("target_domain")
            amount = params.get("amount")
            
            if target_domain in self._state["domains"]:
                self._state["globals"]["entropy_tokens"] += amount
                chaos_increase = math.ceil(amount / 10)
                self._state["domains"][target_domain]["chaos"] += chaos_increase
        
        elif action_name == "reverse_entropy":
            target_domain = params.get("target_domain")
            tokens = params.get("tokens")
            
            if self._state["globals"]["entropy_tokens"] >= tokens and target_domain in self._state["domains"]:
                self._state["globals"]["entropy_tokens"] -= tokens
                self._state["domains"][target_domain]["order"] += tokens
                
                chaos_spillover = math.ceil(tokens / 2)
                for domain_name, domain in self._state["domains"].items():
                    if domain_name != target_domain and not domain.get("locked", False):
                        domain["chaos"] += chaos_spillover
        
        elif action_name == "redistribute_order":
            source_domain = params.get("source_domain")
            target_domain = params.get("target_domain")
            amount = params.get("amount")
            
            if (source_domain in self._state["domains"] and 
                target_domain in self._state["domains"] and
                self._state["domains"][source_domain]["order"] >= amount):
                
                self._state["domains"][source_domain]["order"] -= amount
                self._state["domains"][target_domain]["order"] += amount
        
        elif action_name == "vent_chaos":
            target_domain = params.get("target_domain")
            amount = params.get("amount")
            
            if self._state["globals"]["entropy_tokens"] >= amount:
                self._state["globals"]["entropy_tokens"] -= amount
                if target_domain in self._state["domains"]:
                    self._state["domains"][target_domain]["chaos"] = max(0, 
                        self._state["domains"][target_domain]["chaos"] - amount)
            else:
                if target_domain in self._state["domains"]:
                    self._state["domains"][target_domain]["chaos"] += amount * 2
        
        elif action_name == "lockdown":
            target_domain = params.get("target_domain")
            
            if self._state["globals"]["entropy_tokens"] >= 10 and target_domain in self._state["domains"]:
                self._state["globals"]["entropy_tokens"] -= 10
                self._state["domains"][target_domain]["locked"] = True
        
        for domain_name, domain in self._state["domains"].items():
            if not domain.get("locked", False):
                other_domains = [d for name, d in self._state["domains"].items() if name != domain_name]
                if other_domains:
                    avg_chaos = sum(d["chaos"] for d in other_domains) / len(other_domains)
                    domain["chaos"] += math.ceil(avg_chaos / 4)
        
        for domain in self._state["domains"].values():
            domain["locked"] = False
        
        total_order = sum(domain["order"] for domain in self._state["domains"].values())
        self._state["globals"]["global_order_score"] = total_order - 200
        
        for domain in self._state["domains"].values():
            domain["order"] = max(0, min(100, domain["order"]))
            domain["energy"] = max(0, min(200, domain["energy"]))
            domain["chaos"] = max(0, min(100, domain["chaos"]))
        
        self._state["globals"]["entropy_tokens"] = max(0, self._state["globals"]["entropy_tokens"])
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        if len(self._history) > 0:
            prev_score = self._history[-1]["globals"]["global_order_score"]
            curr_score = self._state["globals"]["global_order_score"]
            order_increase = curr_score - prev_score
            if order_increase > 0:
                order_reward = order_increase * 2.0
                total_reward += order_reward
                events.append("order_increase")
                reward_info["order_increase"] = order_reward
        
        all_stable = all(domain["chaos"] <= 50 for domain in self._state["domains"].values())
        if all_stable:
            stability_reward = 1.0
            total_reward += stability_reward
            events.append("stability_maintained")
            reward_info["stability_maintained"] = stability_reward
        
        if (action["action"] == "reverse_entropy" and 
            all(domain["chaos"] <= 60 for domain in self._state["domains"].values())):
            efficiency_reward = 0.5
            total_reward += efficiency_reward
            events.append("efficient_reversal")
            reward_info["efficient_reversal"] = efficiency_reward
        
        chaos_spike = any(domain["chaos"] > 70 for domain in self._state["domains"].values())
        if chaos_spike:
            chaos_penalty = -3.0
            total_reward += chaos_penalty
            events.append("chaos_spike")
            reward_info["chaos_spike"] = chaos_penalty
        
        domain_collapse = any(domain["chaos"] > 90 for domain in self._state["domains"].values())
        if domain_collapse:
            collapse_penalty = -25.0
            total_reward += collapse_penalty
            events.append("domain_collapse")
            reward_info["domain_collapse"] = collapse_penalty
        
        if self._state["globals"]["global_order_score"] >= 100:
            achievement_reward = 50.0
            total_reward += achievement_reward
            events.append("goal_achieved")
            reward_info["goal_achieved"] = achievement_reward
        
        return total_reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        obs = deepcopy(self._state)
        obs["t"] = self._t
        return obs
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        max_steps = self.configs["termination"]["max_steps"]
        if "globals" in omega and "max_steps" in omega["globals"]:
            max_steps = omega["globals"]["max_steps"]
        
        output = "=== Entropy Reversal Engineering Status ===\n"
        output += f"Step: {omega['t'] + 1}/{max_steps}\n"
        output += f"Global Order Score: {omega['globals']['global_order_score']}/100\n"
        output += f"Entropy Tokens: {omega['globals']['entropy_tokens']}\n\n"
        
        output += "Domain Status:\n"
        
        domains = omega["domains"]
        domain_display = {
            "thermal_grid": "Thermal Grid ",
            "data_archive": "Data Archive ",
            "crystal_lattice": "Crystal Latt.",
            "bio_habitat": "Bio Habitat  "
        }
        
        for domain_key, display_name in domain_display.items():
            if domain_key in domains:
                domain = domains[domain_key]
                locked_status = "[LOCKED]" if domain.get("locked", False) else "       "
                output += f"{display_name} - Order: {domain['order']:3d} | Energy: {domain['energy']:3d} | Chaos: {domain['chaos']:3d} {locked_status}\n"
        
        output += "\nActions: inject_energy(domain,amount), reverse_entropy(domain,tokens),\n"
        output += "         redistribute_order(source,target,amount), vent_chaos(domain,amount), lockdown(domain)"
        
        return output
    
    def done(self, state=None) -> bool:
        if state is None:
            state = self._state
            
        max_steps = self.configs["termination"]["max_steps"]
        if "globals" in state and "max_steps" in state["globals"]:
            max_steps = state["globals"]["max_steps"]
        
        if self._t >= max_steps:
            return True
        
        if state["globals"]["global_order_score"] >= 100:
            return True
        
        if any(domain["chaos"] > 90 for domain in state["domains"].values()):
            return True
        
        return False
