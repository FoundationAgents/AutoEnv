from base.env.base_env import SkinEnv
from env_obs import ColonyObservationPolicy
from env_generate import ColonyWorldGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os
import random
import copy

class AlienColonyEnv(SkinEnv):
    def __init__(self, env_id: int):
        self.observation_policy = ColonyObservationPolicy()
        super().__init__(env_id, self.observation_policy)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None and mode == "load":
            # Find the most recent world file if no world_id specified
            levels_dir = "./levels/"
            if os.path.exists(levels_dir):
                world_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
                if world_files:
                    world_id = max(world_files).replace('.yaml', '')
        
        if world_id:
            self._state = self._load_world(world_id)
        else:
            # Fallback to generating a new world
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        raw_obs = self.observe_semantic()
        return self.render_skin(raw_obs)
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = ColonyWorldGenerator(str(self.env_id), self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(copy.deepcopy(self._state))
        action_name = action.get('action')
        params = action.get('params', {})
        
        result_message = ""
        
        if action_name == "gather_resource":
            result_message = self._handle_gather_resource(params)
        elif action_name == "build_structure":
            result_message = self._handle_build_structure(params)
        elif action_name == "allocate_resource":
            result_message = self._handle_allocate_resource(params)
        elif action_name == "manage_environment":
            result_message = self._handle_manage_environment(params)
        elif action_name == "explore_area":
            result_message = self._handle_explore_area(params)
        
        self._last_action_result = result_message
        return self._state
    
    def _handle_gather_resource(self, params: Dict[str, Any]) -> str:
        resource_type = params.get('resource_type')
        amount = params.get('amount', 1)
        
        if resource_type not in self._state['resources']:
            return f"Unknown resource type: {resource_type}"
        
        # Apply environment modifiers
        season_effect = self._state['_hidden']['environment_effects']['seasons'][self._state['environment']['season']]
        weather_effect = self._state['_hidden']['environment_effects']['weather'][self._state['environment']['weather']]
        multiplier = season_effect['resource_multiplier'] * weather_effect['resource_multiplier']
        
        actual_amount = int(amount * multiplier)
        self._state['resources'][resource_type] += actual_amount
        
        return f"Gathered {actual_amount} units of {resource_type}"
    
    def _handle_build_structure(self, params: Dict[str, Any]) -> str:
        building_type = params.get('building_type')
        x = params.get('x')
        y = params.get('y')
        
        if not building_type or x is None or y is None:
            return "Invalid building parameters"
        
        # Check if position is valid and not occupied
        if x < 0 or x >= self._state['grid']['size'][0] or y < 0 or y >= self._state['grid']['size'][1]:
            return "Position outside grid bounds"
        
        if [x, y] in self._state['grid']['occupied']:
            return "Position already occupied"
        
        # Add building
        new_building = {
            'type': building_type,
            'location': [x, y],
            'operational': True
        }
        self._state['buildings'].append(new_building)
        self._state['grid']['occupied'].append([x, y])
        
        return f"Built {building_type} at position ({x}, {y})"
    
    def _handle_allocate_resource(self, params: Dict[str, Any]) -> str:
        resource_type = params.get('resource_type')
        allocation_amount = params.get('allocation_amount', 1)
        target_system = params.get('target_system', 'colony')
        
        if resource_type not in self._state['resources']:
            return f"Unknown resource type: {resource_type}"

        usage_tracker = self._state['_hidden']['discovery_tracking']['resource_usage']
        if resource_type not in usage_tracker:
            usage_tracker[resource_type] = 0

        if self._state['resources'][resource_type] < allocation_amount:
            return f"Not enough {resource_type} (have {self._state['resources'][resource_type]}, need {allocation_amount})"
        
        # Use semantic mapping to determine true effect
        true_effect = self._state['_hidden']['resource_mappings'].get(resource_type)
        
        # Deduct resource
        self._state['resources'][resource_type] -= allocation_amount
        
        # Apply true effect (opposite of what name suggests)
        if true_effect == 'nutrition_boost':
            self._state['colony']['population'] += allocation_amount
            self._state['colony']['happiness'] += allocation_amount // 2
        elif true_effect == 'health_boost':
            self._state['colony']['happiness'] += allocation_amount * 2
        elif true_effect == 'happiness_boost':
            self._state['colony']['happiness'] += allocation_amount * 3
        elif true_effect == 'efficiency_boost':
            # Boost all building efficiency temporarily
            self._state['environment']['effectiveness_modifiers']['building_efficiency'] += 0.1 * allocation_amount
        
        # Track usage for discovery
        usage_tracker[resource_type] += allocation_amount
        
        return f"Allocated {allocation_amount} {resource_type} to {target_system}"
    
    def _handle_manage_environment(self, params: Dict[str, Any]) -> str:
        intervention_type = params.get('intervention_type', 'weather_change')
        intensity = params.get('intensity', 1)
        
        if intervention_type == 'weather_change':
            weather_options = ['corrosive_weather', 'calm_weather', 'toxic_weather']
            new_weather = random.choice(weather_options)
            self._state['environment']['weather'] = new_weather
            return f"Changed weather to {new_weather}"
        elif intervention_type == 'season_change':
            season_options = ['harsh_season', 'mild_season', 'storm_season']
            new_season = random.choice(season_options)
            self._state['environment']['season'] = new_season
            return f"Changed season to {new_season}"
        
        return f"Applied {intervention_type} intervention with intensity {intensity}"
    
    def _handle_explore_area(self, params: Dict[str, Any]) -> str:
        direction = params.get('direction', 'north')
        investment_level = params.get('investment_level', 1)
        
        self._state['colony']['area_exploration'] += investment_level
        
        # Check if new resources are unlocked
        unlocked_resources = []
        for resource, info in self._state['_hidden']['explorable_resources'].items():
            if (self._state['colony']['area_exploration'] >= info['required_exploration_level'] and 
                resource not in self._state['resources']):
                self._state['resources'][resource] = info['initial_amount']
                usage_tracker = self._state['_hidden']['discovery_tracking']['resource_usage']
                if resource not in usage_tracker:
                    usage_tracker[resource] = 0
                unlocked_resources.append(resource)
        
        result = f"Explored {direction}, area exploration level now {self._state['colony']['area_exploration']}"
        if unlocked_resources:
            result += f". Discovered: {', '.join(unlocked_resources)}"
        
        return result
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        # Check for resource discoveries
        resource_discoveries = self._check_resource_discoveries()
        for resource in resource_discoveries:
            if resource not in self._state['discovery']['resource_effects_found']:
                self._state['discovery']['resource_effects_found'].append(resource)
                total_reward += 0.1
                events.append(f"discovered_resource_{resource}")
        
        # Check for building discoveries
        building_discoveries = self._check_building_discoveries()
        for building in building_discoveries:
            if building not in self._state['discovery']['building_effects_found']:
                self._state['discovery']['building_effects_found'].append(building)
                total_reward += 0.1
                events.append(f"discovered_building_{building}")
        
        # Check for goal achievement
        if self._state['colony']['population'] >= self._state['globals']['target_population']:
            total_reward += 0.7
            events.append("goal_achieved")
        
        reward_info = {
            'resource_discoveries': resource_discoveries,
            'building_discoveries': building_discoveries,
            'population': self._state['colony']['population'],
            'happiness': self._state['colony']['happiness']
        }
        
        return total_reward, events, reward_info
    
    def _check_resource_discoveries(self) -> List[str]:
        discoveries = []
        for resource, usage_count in self._state['_hidden']['discovery_tracking']['resource_usage'].items():
            if usage_count >= 3:  # Threshold for discovery
                # Check if population or happiness increased after usage
                if len(self._history) > 0:
                    prev_pop = self._history[-1]['colony']['population']
                    prev_happy = self._history[-1]['colony']['happiness']
                    if (self._state['colony']['population'] > prev_pop or 
                        self._state['colony']['happiness'] > prev_happy):
                        discoveries.append(resource)
        return discoveries
    
    def _check_building_discoveries(self) -> List[str]:
        discoveries = []
        # Simple discovery mechanism - if buildings exist and colony is growing
        for building in self._state['buildings']:
            building_type = building['type']
            if building_type not in self._state['discovery']['building_effects_found']:
                if self._state['colony']['population'] > 15:  # Some growth threshold
                    discoveries.append(building_type)
        return discoveries
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        resources_display = "\n".join([f"  {name}: {amount}" for name, amount in omega['resources'].items()])
        buildings_display = "\n".join([f"  {b['type']} at ({b['location'][0]}, {b['location'][1]})" for b in omega['buildings']])
        resource_discoveries = ", ".join(omega['resource_effects_found']) if omega['resource_effects_found'] else "None"
        building_discoveries = ", ".join(omega['building_effects_found']) if omega['building_effects_found'] else "None"
        
        return f"""=== ALIEN COLONY STATUS - Step {omega['t']}/{omega['max_steps']} ===
Population: {omega['population']} (Target: {omega['target_population']})
Happiness: {omega['happiness']}/100
Area Exploration Level: {omega['area_exploration']}

Resources:
{resources_display}

Buildings:
{buildings_display}

Environment: {omega['season']} with {omega['weather']}

Discovered Effects:
Resources: {resource_discoveries}
Buildings: {building_discoveries}

Available Actions:
- gather_resource(resource_type, amount): Collect 1-10 units of a resource
- build_structure(building_type, x, y): Construct building at grid position
- allocate_resource(resource_type, amount, target): Distribute resources to colony systems
- manage_environment(intervention_type, intensity): Influence weather/season (intensity 1-5)
- explore_area(direction, investment): Expand territory in cardinal directions"""
    
    def done(self, state=None) -> bool:
        # Check max steps from loaded level config if available
        max_steps = self._state.get('globals', {}).get('max_steps', self.configs["termination"]["max_steps"])
        
        return (self._t >= max_steps or 
                self._state['colony']['population'] >= self._state['globals']['target_population'] or
                self._state['colony']['population'] < self._state['globals']['min_population'])
