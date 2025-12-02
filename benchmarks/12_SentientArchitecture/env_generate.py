from base.env.base_generator import WorldGenerator
import yaml
import random
import os
from typing import Dict, Any, Optional
import time
from copy import deepcopy

class CityGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = deepcopy(self.config['state_template'])
        world_state = self._execute_pipeline(base_state, seed)
        
        world_id = self._generate_world_id(seed)
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = deepcopy(base_state)
        
        generator_config = self.config['generator']
        
        # Step 1: init_from_template (already done by passing base_state)
        
        # Step 2: generate_buildings
        building_count = random.randint(4, 8)
        mood_options = ["Calm", "Restless", "Ambitious", "Contemplative", "Energetic"]
        
        buildings = []
        for i in range(building_count):
            building = {
                'building_id': f"building_{i}",
                'integrity': random.randint(60, 80),
                'energy_reserves': random.randint(20, 35),
                'trust': random.randint(50, 70),
                'growth_stage': "Seedling",
                'mood': random.choice(mood_options),
                'bio_materials': 0
            }
            buildings.append(building)
        
        world_state['buildings'] = buildings
        
        # Step 3: randomize_city_resources
        world_state['city']['bio_material_stock'] = random.randint(40, 60)
        world_state['city']['energy_grid_capacity'] = random.randint(80, 120)
        
        # Step 4: initialize_relationships
        conflicts = []
        mood_incompatible = {
            "Calm": ["Restless", "Ambitious"],
            "Restless": ["Calm", "Contemplative"],
            "Ambitious": ["Calm", "Contemplative"],
            "Contemplative": ["Restless", "Ambitious"],
            "Energetic": []
        }
        
        for i, building1 in enumerate(buildings):
            for j, building2 in enumerate(buildings[i+1:], i+1):
                if building2['mood'] in mood_incompatible.get(building1['mood'], []):
                    if random.random() < 0.3:
                        conflicts.append({
                            'building_id_1': building1['building_id'],
                            'building_id_2': building2['building_id'],
                            'intensity': random.randint(10, 30)
                        })
        
        world_state['conflicts'] = conflicts
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time() * 1000)
        if seed is not None:
            return f"world_seed_{seed}_{timestamp}"
        else:
            return f"world_random_{timestamp}"
