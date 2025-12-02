from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import ArchitectureObservationPolicy
from env_generate import CityGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class SentientArchitectureEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = ArchitectureObservationPolicy()
        super().__init__(env_id, obs_policy)
    
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = CityGenerator(str(self.env_id), self.configs)
        world_id = generator.generate(seed)
        return world_id
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        action_name = action.get('action')
        params = action.get('params', {})
        
        if action_name == "Negotiate":
            self._negotiate(params.get('building_id'))
        elif action_name == "AllocateEnergy":
            self._allocate_energy(params.get('building_id'), params.get('amount'))
        elif action_name == "SupplyBioMaterial":
            self._supply_bio_material(params.get('building_id'), params.get('amount'))
        elif action_name == "Mediate":
            self._mediate(params.get('building_id_1'), params.get('building_id_2'))
        elif action_name == "StimulateGrowth":
            self._stimulate_growth(params.get('building_id'))
        elif action_name == "Repair":
            self._repair(params.get('building_id'), params.get('resource_amount'))
        elif action_name == "CityFestival":
            self._city_festival()
        
        self._autonomous_behavior()
        return self._state
    
    def _negotiate(self, building_id: str):
        building = self._get_building(building_id)
        if building:
            trust_change = random.randint(-5, 15)
            building['trust'] = max(0, min(100, building['trust'] + trust_change))
            
            if trust_change > 0:
                mood_options = ["Calm", "Contemplative", "Energetic"]
                if building['mood'] not in mood_options:
                    if random.random() < 0.3:
                        building['mood'] = random.choice(mood_options)
            
            self._last_action_result = f"Negotiated with {building_id}: Trust {'+' if trust_change >= 0 else ''}{trust_change}"
    
    def _allocate_energy(self, building_id: str, amount: int):
        building = self._get_building(building_id)
        if building and amount > 0:
            max_transfer = min(amount, self._state['city']['energy_grid_capacity'])
            actual_transfer = min(max_transfer, 50 - building['energy_reserves'])
            
            building['energy_reserves'] += actual_transfer
            self._state['city']['energy_grid_capacity'] -= actual_transfer
            
            self._last_action_result = f"Allocated {actual_transfer} energy to {building_id}"
    
    def _supply_bio_material(self, building_id: str, amount: int):
        building = self._get_building(building_id)
        if building and amount > 0:
            max_supply = min(amount, self._state['city']['bio_material_stock'])
            building['bio_materials'] = building.get('bio_materials', 0) + max_supply
            self._state['city']['bio_material_stock'] -= max_supply
            
            self._last_action_result = f"Supplied {max_supply} bio-materials to {building_id}"
    
    def _mediate(self, building_id_1: str, building_id_2: str):
        conflict = None
        for c in self._state['conflicts']:
            if (c['building_id_1'] == building_id_1 and c['building_id_2'] == building_id_2) or \
               (c['building_id_1'] == building_id_2 and c['building_id_2'] == building_id_1):
                conflict = c
                break
        
        if conflict:
            conflict['intensity'] = max(0, conflict['intensity'] - random.randint(10, 25))
            if conflict['intensity'] <= 0:
                self._state['conflicts'].remove(conflict)
            
            self._state['city']['harmony_index'] = min(100, self._state['city']['harmony_index'] + 5)
            self._last_action_result = f"Mediated conflict between {building_id_1} and {building_id_2}"
    
    def _stimulate_growth(self, building_id: str):
        building = self._get_building(building_id)
        if building:
            current_stage = building['growth_stage']
            if current_stage == "Seedling" and building.get('bio_materials', 0) >= 10:
                building['growth_stage'] = "Mature"
                building['bio_materials'] -= 10
                building['integrity'] = max(30, building['integrity'] - 15)
                self._last_action_result = f"{building_id} grew to Mature stage"
            elif current_stage == "Mature" and building.get('bio_materials', 0) >= 20:
                building['growth_stage'] = "Monumental"
                building['bio_materials'] -= 20
                building['integrity'] = max(30, building['integrity'] - 20)
                self._last_action_result = f"{building_id} grew to Monumental stage"
    
    def _repair(self, building_id: str, resource_amount: int):
        building = self._get_building(building_id)
        if building and resource_amount > 0:
            max_repair = min(resource_amount, self._state['city']['bio_material_stock'])
            integrity_gain = max_repair * 2
            building['integrity'] = min(100, building['integrity'] + integrity_gain)
            self._state['city']['bio_material_stock'] -= max_repair
            
            self._last_action_result = f"Repaired {building_id}: +{integrity_gain} integrity"
    
    def _city_festival(self):
        cost_bio = 15
        cost_energy = 20
        
        if self._state['city']['bio_material_stock'] >= cost_bio and \
           self._state['city']['energy_grid_capacity'] >= cost_energy:
            
            self._state['city']['bio_material_stock'] -= cost_bio
            self._state['city']['energy_grid_capacity'] -= cost_energy
            
            for building in self._state['buildings']:
                building['trust'] = min(100, building['trust'] + random.randint(5, 15))
                building['mood'] = random.choice(["Calm", "Contemplative", "Energetic"])
            
            self._state['city']['harmony_index'] = min(100, self._state['city']['harmony_index'] + 10)
            self._last_action_result = "City Festival boosted morale across all buildings"
    
    def _autonomous_behavior(self):
        for building in self._state['buildings']:
            building['energy_reserves'] = min(50, building['energy_reserves'] + random.randint(1, 3))
            building['bio_materials'] = building.get('bio_materials', 0) + random.randint(0, 2)
        
        self._state['city']['energy_grid_capacity'] = min(200, self._state['city']['energy_grid_capacity'] + random.randint(2, 5))
        self._state['city']['bio_material_stock'] = min(100, self._state['city']['bio_material_stock'] + random.randint(1, 3))
        
        self._update_synergy()
    
    def _update_synergy(self):
        high_trust_buildings = [b for b in self._state['buildings'] if b['trust'] >= 80]
        if len(high_trust_buildings) >= 3:
            synergy_gain = len(high_trust_buildings) * 5
            self._state['city']['synergy_score'] = min(100, self._state['city']['synergy_score'] + synergy_gain)
    
    def _get_building(self, building_id: str):
        for building in self._state['buildings']:
            if building['building_id'] == building_id:
                return building
        return None
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        if len(self._history) == 0:
            return 0.0, [], {}
        
        prev_state = self._history[-1]
        
        # Trust increase rewards
        prev_total_trust = sum(b['trust'] for b in prev_state['buildings'])
        curr_total_trust = sum(b['trust'] for b in self._state['buildings'])
        trust_gain = curr_total_trust - prev_total_trust
        if trust_gain > 0:
            trust_reward = trust_gain * 0.01
            total_reward += trust_reward
            events.append("trust_increase")
            reward_info["trust_gain"] = trust_gain
        
        # Integrity maintenance
        all_above_70 = all(b['integrity'] >= 70 for b in self._state['buildings'])
        if all_above_70:
            total_reward += 1.0
            events.append("integrity_maintenance")
        
        # Growth completion
        for i, building in enumerate(self._state['buildings']):
            if i < len(prev_state['buildings']):
                prev_stage = prev_state['buildings'][i]['growth_stage']
                curr_stage = building['growth_stage']
                if prev_stage != curr_stage:
                    total_reward += 3.0
                    events.append("successful_growth")
                    reward_info["growth_building"] = building['building_id']
        
        # Conflict resolution
        prev_conflicts = len(prev_state['conflicts'])
        curr_conflicts = len(self._state['conflicts'])
        if curr_conflicts < prev_conflicts:
            total_reward += 2.0
            events.append("conflict_resolution")
        
        # Synergy cascade
        high_trust_count = sum(1 for b in self._state['buildings'] if b['trust'] >= 80)
        if high_trust_count >= 3:
            total_reward += 5.0
            events.append("synergy_cascade")
            reward_info["high_trust_buildings"] = high_trust_count
        
        # Objective completion
        if self._state['city']['synergy_score'] >= self._state['globals']['target_synergy']:
            prev_synergy = prev_state['city']['synergy_score']
            if prev_synergy < self._state['globals']['target_synergy']:
                total_reward += 100.0
                events.append("objective_completion")
        
        return total_reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        city = omega['city']
        buildings = omega['buildings']
        conflicts = omega['conflicts']
        
        output = f"=== Sentient City Management Console ===\n"
        output += f"Step {omega['t'] + 1}/{omega['max_steps']} | City Synergy: {city['synergy_score']}/{omega['target_synergy']} | Harmony: {city['harmony_index']}\n\n"
        
        output += f"City Resources:\n"
        output += f"- Bio-Materials: {city['bio_material_stock']} units\n"
        output += f"- Energy Grid: {city['energy_grid_capacity']} units\n\n"
        
        output += f"Building Status:\n"
        for building in buildings:
            output += f"  {building['building_id']}: "
            output += f"Integrity={building['integrity']}, Energy={building['energy_reserves']}, "
            output += f"Trust={building['trust']}, Stage={building['growth_stage']}, "
            output += f"Mood={building['mood']}, Bio-Materials={building.get('bio_materials', 0)}\n"
        
        output += f"\nActive Conflicts: {len(conflicts)}\n"
        for conflict in conflicts:
            output += f"  {conflict['building_id_1']} vs {conflict['building_id_2']} (intensity: {conflict['intensity']})\n"
        
        output += f"\nAvailable Actions: Negotiate, AllocateEnergy, SupplyBioMaterial, Mediate, StimulateGrowth, Repair, CityFestival"
        
        return output
    
    def done(self, state: Dict[str, Any] = None) -> bool:
        state = state if state is not None else getattr(self, "_state", None)
        if state is None:
            return False
        
        # Check failure conditions
        for building in state['buildings']:
            if building['integrity'] <= 0 or building['trust'] <= 0:
                return True
        
        # Check success condition
        if state['city']['synergy_score'] >= state['globals']['target_synergy']:
            return True
        
        # Check max steps
        max_steps = state['globals'].get('max_steps', self.configs['termination']['max_steps'])
        return self._t >= max_steps
