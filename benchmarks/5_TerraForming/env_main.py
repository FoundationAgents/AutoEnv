from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import TerraformingObservation
from env_generate import TerraformingGenerator
import yaml
import os
import math
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class TerraformingEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = TerraformingObservation()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = TerraformingGenerator(str(self.env_id), self.configs)
        world_id = generator.generate(seed)
        return world_id
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load" and world_id:
            self._state = self._load_world(world_id)
        else:
            raise ValueError("Invalid reset mode or missing parameters")
        
        if 'globals' in self._state and 'max_steps' in self._state['globals']:
            self.configs['termination']['max_steps'] = self._state['globals']['max_steps']
        
        return self.observe_semantic()
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        action_name = action.get('action', '')
        params = action.get('params', {})
        
        if action_name == "DEPLOY_ATMOSPHERIC_PROCESSOR":
            self._deploy_atmospheric_processor()
        elif action_name == "RELEASE_WATER_CATALYSTS":
            self._release_water_catalysts()
        elif action_name == "SEED_MICROBIAL_LIFE":
            self._seed_microbial_life()
        elif action_name == "STABILIZE_TECTONICS":
            self._stabilize_tectonics()
        elif action_name == "CONSTRUCT_UPGRADE_STATION":
            self._construct_upgrade_station()
        elif action_name == "DIVERT_ENERGY_TO_SHIELDS":
            self._divert_energy_to_shields()
        elif action_name == "PASSIVE_OBSERVATION":
            self._passive_observation()
        
        self._update_habitability_index()
        self._apply_cascading_effects()
        self._last_action_result = f"Executed {action_name}"
        
        return self._state
    
    def _deploy_atmospheric_processor(self):
        if self._state['infrastructure']['energy_reserves'] < 100:
            return
        
        self._state['infrastructure']['energy_reserves'] -= 100
        
        o2_increase = 3.0 + (self._state['infrastructure']['station_upgrade_level'] * 0.5)
        co2_decrease = 4.0 + (self._state['infrastructure']['station_upgrade_level'] * 0.5)
        
        self._state['atmosphere']['oxygen_pct'] = min(25.0, 
            self._state['atmosphere']['oxygen_pct'] + o2_increase)
        self._state['atmosphere']['co2_pct'] = max(0.0,
            self._state['atmosphere']['co2_pct'] - co2_decrease)
        
        temp_change = -8.0 * (co2_decrease / 100.0)
        self._state['atmosphere']['temperature'] += temp_change
        
        pressure_change = -0.05 * (co2_decrease / 100.0)
        self._state['atmosphere']['pressure'] = max(0.1,
            self._state['atmosphere']['pressure'] + pressure_change)
        
        self._state['global_metrics']['instability_index'] += 2.0
    
    def _release_water_catalysts(self):
        if self._state['infrastructure']['energy_reserves'] < 80:
            return
        
        self._state['infrastructure']['energy_reserves'] -= 80
        
        ice_conversion = min(15.0, self._state['hydrosphere']['subsurface_ice_pct'])
        water_gain = ice_conversion * 0.8
        
        self._state['hydrosphere']['subsurface_ice_pct'] -= ice_conversion
        self._state['hydrosphere']['surface_water_pct'] = min(70.0,
            self._state['hydrosphere']['surface_water_pct'] + water_gain)
        
        ph_improvement = 0.3
        self._state['hydrosphere']['ph_level'] = min(7.0,
            self._state['hydrosphere']['ph_level'] + ph_improvement)
        
        self._state['atmosphere']['pressure'] += 0.02
        self._state['global_metrics']['instability_index'] += 1.5
    
    def _seed_microbial_life(self):
        if self._state['infrastructure']['energy_reserves'] < 60:
            return
        
        o2_suitable = 5.0 <= self._state['atmosphere']['oxygen_pct'] <= 25.0
        water_suitable = self._state['hydrosphere']['surface_water_pct'] >= 10.0
        ph_suitable = self._state['hydrosphere']['ph_level'] >= 3.0
        
        if not (o2_suitable and water_suitable):
            self._state['global_metrics']['instability_index'] += 3.0
            return
        
        self._state['infrastructure']['energy_reserves'] -= 60
        
        microbe_activation = min(50.0, self._state['biosphere_seeds']['dormant_microbes'])
        self._state['biosphere_seeds']['dormant_microbes'] -= microbe_activation
        
        flora_activation = 0  # Initialize to 0
        if ph_suitable:
            flora_activation = min(20.0, self._state['biosphere_seeds']['dormant_flora'])
            self._state['biosphere_seeds']['dormant_flora'] -= flora_activation
        
        fertility_gain = (microbe_activation * 0.4) + (flora_activation * 0.6)
        self._state['lithosphere']['soil_fertility'] = min(100.0,
            self._state['lithosphere']['soil_fertility'] + fertility_gain)
        
        self._state['atmosphere']['oxygen_pct'] = min(25.0,
            self._state['atmosphere']['oxygen_pct'] + (microbe_activation * 0.1))
        
        self._state['global_metrics']['instability_index'] += 1.0
    
    def _stabilize_tectonics(self):
        energy_needed = max(50, int(self._state['lithosphere']['tectonic_stress'] * 2))
        if self._state['infrastructure']['energy_reserves'] < energy_needed:
            return
        
        self._state['infrastructure']['energy_reserves'] -= energy_needed
        
        stress_reduction = 15.0 + (self._state['infrastructure']['station_upgrade_level'] * 2.0)
        self._state['lithosphere']['tectonic_stress'] = max(0.0,
            self._state['lithosphere']['tectonic_stress'] - stress_reduction)
        
        self._state['global_metrics']['instability_index'] = max(0.0,
            self._state['global_metrics']['instability_index'] - 2.0)
    
    def _construct_upgrade_station(self):
        if self._state['infrastructure']['terraforming_stations'] < 3:
            energy_cost = 200
            if self._state['infrastructure']['energy_reserves'] < energy_cost:
                return
            
            self._state['infrastructure']['energy_reserves'] -= energy_cost
            self._state['infrastructure']['terraforming_stations'] += 1
            self._state['global_metrics']['instability_index'] += 5.0
        else:
            if self._state['infrastructure']['station_upgrade_level'] >= 3:
                return
            
            energy_cost = 150 * (self._state['infrastructure']['station_upgrade_level'] + 1)
            if self._state['infrastructure']['energy_reserves'] < energy_cost:
                return
            
            self._state['infrastructure']['energy_reserves'] -= energy_cost
            self._state['infrastructure']['station_upgrade_level'] += 1
            self._state['global_metrics']['instability_index'] += 3.0
    
    def _divert_energy_to_shields(self):
        if self._state['infrastructure']['energy_reserves'] < 50:
            return
        
        energy_used = min(200, self._state['infrastructure']['energy_reserves'])
        self._state['infrastructure']['energy_reserves'] -= energy_used
        
        stability_reduction = energy_used * 0.1
        self._state['global_metrics']['instability_index'] = max(0.0,
            self._state['global_metrics']['instability_index'] - stability_reduction)
    
    def _passive_observation(self):
        natural_evolution = 0.2
        self._state['atmosphere']['oxygen_pct'] = max(0.0,
            self._state['atmosphere']['oxygen_pct'] - natural_evolution)
        self._state['hydrosphere']['surface_water_pct'] = max(0.0,
            self._state['hydrosphere']['surface_water_pct'] - (natural_evolution * 0.5))
        self._state['global_metrics']['instability_index'] = max(0.0,
            self._state['global_metrics']['instability_index'] - 0.5)
    
    def _update_habitability_index(self):
        atmosphere_score = 0
        if 15.0 <= self._state['atmosphere']['oxygen_pct'] <= 25.0:
            atmosphere_score += 40
        else:
            atmosphere_score += max(0, 40 - abs(20.0 - self._state['atmosphere']['oxygen_pct']) * 2)
        
        if self._state['atmosphere']['co2_pct'] <= 10.0:
            atmosphere_score += 35
        else:
            atmosphere_score += max(0, 35 - (self._state['atmosphere']['co2_pct'] - 10.0) * 2)
        
        if -10.0 <= self._state['atmosphere']['temperature'] <= 30.0:
            atmosphere_score += 25
        else:
            temp_penalty = max(abs(self._state['atmosphere']['temperature'] + 10), 
                             abs(self._state['atmosphere']['temperature'] - 30))
            atmosphere_score += max(0, 25 - temp_penalty)
        
        atmosphere_component = min(30.0, atmosphere_score * 0.3)
        
        water_score = min(100, self._state['hydrosphere']['surface_water_pct'] * 1.5)
        if self._state['hydrosphere']['ph_level'] >= 6.0:
            water_score *= 1.2
        elif self._state['hydrosphere']['ph_level'] < 4.0:
            water_score *= 0.7
        water_component = min(25.0, water_score * 0.25)
        
        biology_score = 0
        active_microbes = 100.0 - self._state['biosphere_seeds']['dormant_microbes']
        active_flora = 50.0 - self._state['biosphere_seeds']['dormant_flora']
        biology_score = (active_microbes * 0.6) + (active_flora * 0.8) + self._state['lithosphere']['soil_fertility']
        biology_component = min(25.0, biology_score * 0.125)
        
        stability_score = max(0, 100 - self._state['global_metrics']['instability_index'])
        stability_component = min(20.0, stability_score * 0.2)
        
        total_habitability = atmosphere_component + water_component + biology_component + stability_component
        self._state['global_metrics']['habitability_index'] = min(100.0, total_habitability)
    
    def _apply_cascading_effects(self):
        if self._state['infrastructure']['energy_reserves'] < 200:
            self._state['global_metrics']['instability_index'] += 1.0
        
        if self._state['atmosphere']['oxygen_pct'] > 25.0:
            self._state['global_metrics']['instability_index'] += 2.0
        
        if self._state['hydrosphere']['surface_water_pct'] > 70.0:
            self._state['lithosphere']['tectonic_stress'] += 1.0
            self._state['global_metrics']['instability_index'] += 1.0
        
        if self._state['lithosphere']['tectonic_stress'] > 80.0:
            self._state['global_metrics']['instability_index'] += 2.0
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_breakdown = {}
        
        prev_state = self._history[-1] if self._history else {}
        
        if self._state['global_metrics']['instability_index'] < 50.0:
            stability_reward = 0.1
            total_reward += stability_reward
            reward_breakdown['stability_maintained'] = stability_reward
            events.append('stability_maintained')
        
        if prev_state:
            o2_change = self._state['atmosphere']['oxygen_pct'] - prev_state['atmosphere']['oxygen_pct']
            if o2_change > 0 and 15.0 <= self._state['atmosphere']['oxygen_pct'] <= 25.0:
                atm_reward = o2_change * 0.05
                total_reward += atm_reward
                reward_breakdown['atmospheric_progress'] = atm_reward
                events.append('atmospheric_progress')
            
            water_change = self._state['hydrosphere']['surface_water_pct'] - prev_state['hydrosphere']['surface_water_pct']
            if water_change > 0 and 30.0 <= self._state['hydrosphere']['surface_water_pct'] <= 70.0:
                hydro_reward = water_change * 0.1
                total_reward += hydro_reward
                reward_breakdown['hydrological_progress'] = hydro_reward
                events.append('hydrological_progress')
            
            hab_change = self._state['global_metrics']['habitability_index'] - prev_state['global_metrics']['habitability_index']
            if hab_change > 0:
                hab_reward = hab_change * 0.2
                total_reward += hab_reward
                reward_breakdown['habitability_increase'] = hab_reward
                events.append('habitability_increase')
        
        if self._state['global_metrics']['habitability_index'] >= 100.0:
            completion_reward = 20.0
            total_reward += completion_reward
            reward_breakdown['mission_completed'] = completion_reward
            events.append('mission_completed')
        
        if self._state['global_metrics']['instability_index'] > 70.0:
            excess_instability = self._state['global_metrics']['instability_index'] - 70.0
            instability_penalty = -0.2 * excess_instability
            total_reward += instability_penalty
            reward_breakdown['instability_penalty'] = instability_penalty
            events.append('instability_penalty')
        
        if self._state['global_metrics']['instability_index'] >= 100.0:
            failure_penalty = -40.0
            total_reward += failure_penalty
            reward_breakdown['catastrophic_failure'] = failure_penalty
            events.append('catastrophic_failure')
        
        return total_reward, events, reward_breakdown
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t + 1)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        template = """=== TERRAFORMING MISSION CONTROL - STEP {t}/{max_steps} ===

ATMOSPHERIC STATUS:
O₂: {oxygen_pct:.1f}% | CO₂: {co2_pct:.1f}% 
Pressure: {pressure:.2f} atm | Temp: {temperature:.1f}°C

HYDROSPHERE STATUS:
Surface Water: {surface_water_pct:.1f}% | Ice Reserves: {subsurface_ice_pct:.1f}%
pH Level: {ph_level:.1f}

GEOLOGICAL & BIOLOGICAL STATUS:
Soil Fertility: {soil_fertility:.1f} | Tectonic Stress: {tectonic_stress:.1f}
Dormant Microbes: {dormant_microbes:.0f} | Dormant Flora: {dormant_flora:.0f}

INFRASTRUCTURE:
Stations: {terraforming_stations} (Level {station_upgrade_level})
Energy Reserves: {energy_reserves:.0f}

=== MISSION METRICS ===
Habitability Index: {habitability_index:.1f}% / 100%
Instability Index: {instability_index:.1f}% (CRITICAL at 100%)

AVAILABLE ACTIONS:
1. DEPLOY_ATMOSPHERIC_PROCESSOR - Modify atmospheric composition
2. RELEASE_WATER_CATALYSTS - Activate water systems  
3. SEED_MICROBIAL_LIFE - Introduce biological processes
4. STABILIZE_TECTONICS - Reduce geological instability
5. CONSTRUCT_UPGRADE_STATION - Build/upgrade terraforming infrastructure
6. DIVERT_ENERGY_TO_SHIELDS - Emergency instability reduction
7. PASSIVE_OBSERVATION - Monitor natural system evolution"""
        
        return template.format(**omega)
    
    def done(self, state=None) -> bool:
        if state is None:
            state = self._state
        
        max_steps = self.configs.get("termination", {}).get("max_steps", 40)
        if hasattr(state, 'get') and state.get('globals', {}).get('max_steps'):
            max_steps = state['globals']['max_steps']
        
        return (self._t >= max_steps or 
                state['global_metrics']['habitability_index'] >= 100.0 or
                state['global_metrics']['instability_index'] >= 100.0)