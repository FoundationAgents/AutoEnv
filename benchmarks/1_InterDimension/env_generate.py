from base.env.base_generator import WorldGenerator
import numpy as np
import yaml
import os
import uuid
from typing import Dict, Any, Optional
import copy

class MarketWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            np.random.seed(seed)
            
        world_id = self._generate_world_id(seed)
        
        # Load state template
        base_state = copy.deepcopy(self.config['state_template'])
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Save world
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
            
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = copy.deepcopy(base_state)
        
        # Pipeline step 1: init_from_template (already done)
        
        # Pipeline step 2: randomize_inventory
        self._randomize_inventory(state)
        
        # Pipeline step 3: initialize_ledgers
        self._initialize_ledgers(state)
        
        # Pipeline step 4: generate_exchange_patterns
        self._generate_exchange_patterns(state)
        
        # Pipeline step 5: set_embargo_risks
        self._set_embargo_risks(state)
        
        return state
    
    def _randomize_inventory(self, state: Dict[str, Any]):
        pipeline_config = self.config['generator']['pipeline'][1]['args']
        min_categories = pipeline_config['min_categories']
        max_categories = pipeline_config['max_categories']
        min_units = pipeline_config['min_units']
        max_units = pipeline_config['max_units']
        
        # Select random number of categories
        num_categories = np.random.randint(min_categories, max_categories + 1)
        categories = np.random.choice(
            state['globals']['item_categories'], 
            num_categories, 
            replace=False
        )
        
        # Assign random quantities
        for category in categories:
            quantity = np.random.randint(min_units, max_units + 1)
            state['agent']['inventory'][category] = quantity
    
    def _initialize_ledgers(self, state: Dict[str, Any]):
        pipeline_config = self.config['generator']['pipeline'][2]['args']
        min_balance = pipeline_config['min_balance']
        max_balance = pipeline_config['max_balance']
        
        for dimension in state['globals']['dimensions']:
            balance = np.random.uniform(min_balance, max_balance)
            state['agent']['ledgers'][dimension] = balance
    
    def _generate_exchange_patterns(self, state: Dict[str, Any]):
        pipeline_config = self.config['generator']['pipeline'][3]['args']
        
        rate_pairs = [
            'mass_entropy', 'mass_historical', 'entropy_mass', 
            'entropy_historical', 'historical_mass', 'historical_entropy'
        ]
        
        for rate_pair in rate_pairs:
            # Generate sinusoidal parameters [amplitude, frequency, phase]
            base_rate = np.random.uniform(*pipeline_config['base_rate_range'])
            amplitude = np.random.uniform(*pipeline_config['amplitude_range'])
            frequency = np.random.uniform(*pipeline_config['frequency_range'])
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Store drift parameters
            state['market']['exchange_matrices']['drift_params'][rate_pair] = [
                amplitude, frequency, phase
            ]
            
            # Set initial true rate
            state['market']['exchange_matrices']['true_rates'][rate_pair] = base_rate
            
            # Set initial observed rate (will be updated with noise in observation)
            state['market']['exchange_matrices']['observed_rates'][rate_pair] = base_rate
    
    def _set_embargo_risks(self, state: Dict[str, Any]):
        pipeline_config = self.config['generator']['pipeline'][4]['args']
        min_risk = pipeline_config['min_risk']
        max_risk = pipeline_config['max_risk']
        
        for dimension in state['globals']['dimensions']:
            risk = np.random.uniform(min_risk, max_risk)
            state['market']['embargo_risks'][dimension] = risk
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"market_world_{seed}_{uuid.uuid4().hex[:8]}"
        else:
            return f"market_world_{uuid.uuid4().hex[:12]}"