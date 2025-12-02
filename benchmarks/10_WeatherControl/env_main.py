import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from base.env.base_env import SkinEnv
from env_obs import AtmosphereObservationPolicy
from env_generate import AtmosphereGenerator
from copy import deepcopy

class AtmosphereEnv(SkinEnv):
    def __init__(self, env_id: int):
        self.obs_policy = AtmosphereObservationPolicy()
        # Initialize attributes before calling super() so they don't get overridden
        self.generator = None
        self.action_costs = {}
        self.discovery_bonuses = {}
        self.perfect_episode_streak = 0
        super().__init__(env_id, self.obs_policy)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        
        self.action_costs = {}
        for action in self.configs["transition"]["actions"]:
            self.action_costs[action["name"]] = action["cost"]
            
        self.generator = AtmosphereGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            raise ValueError("world_id must be provided for load mode")
            
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self.discovery_bonuses = {}
        self.perfect_episode_streak = 0
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed=seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action.get('action')
        action_cost = self.action_costs.get(action_name, 0)
        
        if self._state['agent']['energy_budget'] < action_cost:
            self._last_action_result = "Insufficient energy for action"
            return self._state
        
        self._state['agent']['energy_budget'] -= action_cost
        self._state['agent']['step_counter'] += 1
        
        self._apply_action_effects(action_name)
        self._process_delayed_effects()
        self._apply_natural_drift()
        self._update_coupled_dynamics()
        self._recalculate_csi()
        
        return self._state
    
    def _apply_action_effects(self, action_name: str):
        atmosphere = self._state['atmosphere']
        effects = {}
        
        if action_name == "inject_cold_ions":
            effects = {'temperature': +15, 'atmospheric_pressure': -0.1, 'storm_energy': -5}
        elif action_name == "release_dry_fog":
            effects = {'humidity': +8, 'cloud_coverage': -10, 'solar_flux': +50}
        elif action_name == "vent_heavy_vapor":
            effects = {'temperature': -20, 'atmospheric_pressure': +0.15, 'cloud_coverage': +15}
        elif action_name == "trigger_pressure_spike":
            effects = {'humidity': -10, 'storm_energy': +8, 'atmospheric_pressure': +0.1}
        elif action_name == "emit_solar_net":
            effects = {'solar_flux': -80, 'temperature': +10, 'cloud_coverage': +12}
        elif action_name == "redirect_jet_stream":
            effects = {'humidity': +12, 'storm_energy': +6, 'atmospheric_pressure': -0.08}
        
        physics = self._state.setdefault('physics', {})
        queue = physics.setdefault('action_effects_queue', [])
        
        queue.append({'effects': effects, 'steps_remaining': 2})
    
    def _process_delayed_effects(self):
        physics = self._state.get('physics', {})
        queue = physics.get('action_effects_queue', [])
        atmosphere = self._state['atmosphere']
        
        remaining_effects = []
        for effect_data in queue:
            if effect_data['steps_remaining'] > 0:
                for var, change in effect_data['effects'].items():
                    if var in atmosphere:
                        atmosphere[var] += change
                effect_data['steps_remaining'] -= 1
                if effect_data['steps_remaining'] > 0:
                    remaining_effects.append(effect_data)
        
        physics['action_effects_queue'] = remaining_effects
    
    def _apply_natural_drift(self):
        atmosphere = self._state['atmosphere']
        physics = self._state.get('physics', {})
        drift_rates = physics.get('drift_rates', {})
        drift_directions = physics.get('drift_directions', {})
        
        for var, rate in drift_rates.items():
            if var in atmosphere:
                direction = drift_directions.get(var, 1)
                drift_amount = atmosphere[var] * rate * direction
                atmosphere[var] += drift_amount
    
    def _update_coupled_dynamics(self):
        atmosphere = self._state['atmosphere']
        
        temp = atmosphere.get('temperature', 300)
        humidity = atmosphere.get('humidity', 50)
        pressure = atmosphere.get('atmospheric_pressure', 1.0)
        clouds = atmosphere.get('cloud_coverage', 50)
        storms = atmosphere.get('storm_energy', 30)
        solar = atmosphere.get('solar_flux', 1000)
        
        atmosphere['atmospheric_pressure'] += (temp - 300) * 0.001
        atmosphere['humidity'] += (temp - 300) * 0.02
        atmosphere['cloud_coverage'] += humidity * 0.1 - 5
        atmosphere['storm_energy'] += humidity * 0.05 - 2.5
        atmosphere['temperature'] += (pressure - 1.0) * 10
        atmosphere['solar_flux'] += (pressure - 1.0) * 50
        atmosphere['solar_flux'] -= clouds * 2
        atmosphere['temperature'] -= clouds * 0.2
        atmosphere['atmospheric_pressure'] += storms * 0.002 - 0.06
        atmosphere['humidity'] += storms * 0.3 - 9
        atmosphere['temperature'] += (solar - 1000) * 0.01
        
        for var in ['temperature', 'humidity', 'atmospheric_pressure', 'cloud_coverage', 'storm_energy', 'solar_flux']:
            if var == 'temperature':
                atmosphere[var] = max(100, min(500, atmosphere[var]))
            elif var == 'humidity':
                atmosphere[var] = max(0, min(100, atmosphere[var]))
            elif var == 'atmospheric_pressure':
                atmosphere[var] = max(0.5, min(2.0, atmosphere[var]))
            elif var == 'cloud_coverage':
                atmosphere[var] = max(0, min(100, atmosphere[var]))
            elif var == 'storm_energy':
                atmosphere[var] = max(0, min(100, atmosphere[var]))
            elif var == 'solar_flux':
                atmosphere[var] = max(500, min(1500, atmosphere[var]))
    
    def _recalculate_csi(self):
        atmosphere = self._state['atmosphere']
        
        temp_norm = (atmosphere.get('temperature', 300) - 200) / 200
        humidity_norm = atmosphere.get('humidity', 50) / 100
        pressure_norm = atmosphere.get('atmospheric_pressure', 1.0)
        cloud_norm = atmosphere.get('cloud_coverage', 50) / 100
        storm_norm = (atmosphere.get('storm_energy', 30) - 10) / 40
        solar_norm = (atmosphere.get('solar_flux', 1000) - 800) / 400
        
        csi = 50 + 10 * (temp_norm - 0.5) + 8 * (humidity_norm - 0.5) + 12 * (pressure_norm - 1.0) + 6 * (cloud_norm - 0.5) + 4 * (storm_norm - 0.5) + 5 * (solar_norm - 0.5)
        atmosphere['climate_stability_index'] = max(0, min(100, csi))
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        csi = self._state['atmosphere']['climate_stability_index']
        prev_csi = None
        if self._history:
            prev_csi = self._history[-1]['atmosphere']['climate_stability_index']
        
        if 45 <= csi <= 55:
            total_reward += 0.5
            events.append("stability_maintenance")
            self.perfect_episode_streak += 1
        else:
            self.perfect_episode_streak = 0
        
        if prev_csi is not None and (prev_csi < 45 or prev_csi > 55) and 45 <= csi <= 55:
            total_reward += 3.0
            events.append("stability_recovery")
        
        action_name = action.get('action')
        if action_name not in self.discovery_bonuses:
            if self.perfect_episode_streak >= 3:
                total_reward += 2.0
                events.append("discovery_bonus")
                self.discovery_bonuses[action_name] = True
        
        if self._t >= self.configs["termination"]["max_steps"] or self.done():
            if self.perfect_episode_streak == self._t:
                total_reward += 20.0
                events.append("perfect_episode")
            
            remaining_energy = self._state['agent']['energy_budget']
            energy_bonus = remaining_energy * 0.1
            total_reward += energy_bonus
            events.append("energy_efficiency")
            reward_info['remaining_energy_bonus'] = energy_bonus
        
        reward_info['csi'] = csi
        reward_info['perfect_streak'] = self.perfect_episode_streak
        
        return total_reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        template = self.configs["skin"]["template"]
        # Display steps as 1-based for player view
        omega_mod = dict(omega)
        try:
            omega_mod['t'] = (omega.get('t', 0) or 0) + 1
        except Exception:
            omega_mod['t'] = 1
        return template.format(**omega_mod)
    
    def done(self, state=None) -> bool:
        state = state if state is not None else self._state
        
        if self._t >= self.configs["termination"]["max_steps"]:
            return True
        
        csi = state['atmosphere']['climate_stability_index']
        if csi < 15 or csi > 85:
            return True
        
        if state['agent']['energy_budget'] <= 0:
            return True
        
        return False
