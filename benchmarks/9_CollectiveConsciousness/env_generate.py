from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import uuid
from copy import deepcopy

class HiveMindGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = deepcopy(self.config['state_template'])
        
        world_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = deepcopy(base_state)
        
        gen_config = self.config.get('generator', {})
        pipeline = gen_config.get('pipeline', [])
        
        for step in pipeline:
            if step['name'] == 'init_from_template':
                continue
            elif step['name'] == 'randomize_sub_streams':
                self._randomize_sub_streams(world_state, step.get('args', {}))
            elif step['name'] == 'validate_initial_conditions':
                self._validate_initial_conditions(world_state, step.get('args', {}))
        
        return world_state
    
    def _randomize_sub_streams(self, world_state: Dict[str, Any], args: Dict[str, Any]):
        coherence_range = args.get('coherence_range', [50, 60])
        novelty_range = args.get('novelty_range', [40, 50])
        size_variation = args.get('size_variation', 5.0)
        
        total_size = 100.0
        size_per_stream = total_size / len(world_state['sub_streams'])
        
        for i, stream in enumerate(world_state['sub_streams']):
            stream['coherence'] = random.uniform(coherence_range[0], coherence_range[1])
            stream['novelty'] = random.uniform(novelty_range[0], novelty_range[1])
            
            size_modifier = random.uniform(-size_variation, size_variation)
            stream['size'] = max(10.0, min(40.0, size_per_stream + size_modifier))
        
        actual_total = sum(stream['size'] for stream in world_state['sub_streams'])
        scale_factor = total_size / actual_total
        for stream in world_state['sub_streams']:
            stream['size'] *= scale_factor
    
    def _validate_initial_conditions(self, world_state: Dict[str, Any], args: Dict[str, Any]):
        min_unity = args.get('min_unity', 70.0)
        min_diversity = args.get('min_diversity', 50.0)
        
        unity = sum(stream['coherence'] * stream['size'] for stream in world_state['sub_streams']) / 100.0
        
        novelties = [stream['novelty'] for stream in world_state['sub_streams']]
        novelty_variance = sum((n - sum(novelties)/len(novelties))**2 for n in novelties) / len(novelties)
        diversity = 100 - novelty_variance
        
        if unity < min_unity:
            adjustment = (min_unity - unity) / len(world_state['sub_streams'])
            for stream in world_state['sub_streams']:
                stream['coherence'] = min(100.0, stream['coherence'] + adjustment)
        
        if diversity < min_diversity:
            target_variance = 100 - min_diversity
            avg_novelty = sum(novelties) / len(novelties)
            spread = (target_variance * len(novelties))**0.5
            
            for i, stream in enumerate(world_state['sub_streams']):
                offset = (i - len(novelties)/2 + 0.5) * spread / len(novelties)
                stream['novelty'] = max(0.0, min(100.0, avg_novelty + offset))
        
        world_state['globals']['unity'] = unity
        world_state['globals']['diversity'] = diversity
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"hive_world_seed_{seed}"
        else:
            return f"hive_world_{str(uuid.uuid4())[:8]}"