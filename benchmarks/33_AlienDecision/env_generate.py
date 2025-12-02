from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional
import random
import yaml
import os
from datetime import datetime

class ColonyWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.semantic_mappings = self._create_semantic_mappings()
        
    def _create_semantic_mappings(self) -> Dict[str, Any]:
        return {
            'resources': {
                'toxic_waste': 'nutrition_boost',
                'rotten_food': 'health_boost',
                'contaminated_water': 'happiness_boost',
                'broken_machinery': 'efficiency_boost',
                'poisonous_plants': 'growth_stimulant',
                'radioactive_ore': 'energy_source',
                'infected_samples': 'medical_supplies',
                'corroded_metal': 'construction_material'
            },
            'buildings': {
                'decay_chamber': 'population_accelerator',
                'poison_distributor': 'happiness_generator',
                'waste_processor': 'resource_multiplier',
                'contamination_center': 'research_facility',
                'corruption_hub': 'coordination_center'
            }
        }
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = self.config['state_template'].copy()
        
        generated_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(generated_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = base_state.copy()
        
        # Initialize from template (already done in base_state)
        state = self._init_from_template(state)
        
        # Setup resource semantics
        state = self._setup_resource_semantics(state)
        
        # Setup building semantics
        state = self._setup_building_semantics(state)
        
        # Initialize environment cycles
        state = self._initialize_environment_cycles(state)
        
        # Place discoverable resources
        state = self._place_discoverable_resources(state)
        
        # Setup difficulty scaling
        state = self._setup_difficulty_scaling(state)
        
        return state
    
    def _init_from_template(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Add hidden semantic mappings to state
        state['_hidden'] = {
            'resource_mappings': self.semantic_mappings['resources'],
            'building_mappings': self.semantic_mappings['buildings'],
            'discovery_tracking': {
                'resource_usage': {},
                'building_usage': {},
                'positive_outcomes': []
            }
        }
        return state
    
    def _setup_resource_semantics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # The semantic mappings are already created in init
        # Just ensure discovery tracking is initialized
        for resource in state['resources']:
            state['_hidden']['discovery_tracking']['resource_usage'][resource] = 0
        return state
    
    def _setup_building_semantics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Initialize building usage tracking
        for building in state['buildings']:
            building_type = building['type']
            state['_hidden']['discovery_tracking']['building_usage'][building_type] = 0
        return state
    
    def _initialize_environment_cycles(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Define seasonal and weather effects
        state['_hidden']['environment_effects'] = {
            'seasons': {
                'harsh_season': {'resource_multiplier': 1.2, 'building_efficiency': 1.1},
                'mild_season': {'resource_multiplier': 1.0, 'building_efficiency': 1.0},
                'storm_season': {'resource_multiplier': 0.8, 'building_efficiency': 1.3}
            },
            'weather': {
                'corrosive_weather': {'resource_multiplier': 1.1, 'building_efficiency': 1.2},
                'calm_weather': {'resource_multiplier': 1.0, 'building_efficiency': 1.0},
                'toxic_weather': {'resource_multiplier': 1.3, 'building_efficiency': 0.9}
            }
        }
        return state
    
    def _place_discoverable_resources(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Add 2-4 additional resources that can be discovered through exploration
        additional_resources = ['poisonous_plants', 'radioactive_ore', 'infected_samples', 'corroded_metal']
        num_to_add = random.randint(2, 4)
        
        state['_hidden']['explorable_resources'] = {}
        for i in range(num_to_add):
            resource = additional_resources[i]
            state['_hidden']['explorable_resources'][resource] = {
                'required_exploration_level': i + 2,
                'initial_amount': random.randint(5, 15)
            }
        
        return state
    
    def _setup_difficulty_scaling(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Apply difficulty scaling based on config
        difficulty = random.uniform(0.3, 1.0)
        
        # Scale initial resources
        resource_variance = random.uniform(0.8, 1.2)
        for resource in state['resources']:
            state['resources'][resource] = int(state['resources'][resource] * resource_variance)
        
        # Adjust weather volatility
        weather_volatility = random.uniform(0.5, 1.5)
        state['_hidden']['weather_volatility'] = weather_volatility
        
        return state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_suffix = f"_{seed}" if seed is not None else ""
        return f"world_{timestamp}{seed_suffix}"