from base.env.base_observation import ObservationPolicy
import numpy as np
from typing import Dict, Any
import copy

class MarketObservationPolicy(ObservationPolicy):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {}
        
        # Agent inventory (full visibility)
        obs['inventory'] = copy.deepcopy(env_state['agent']['inventory'])
        
        # Agent ledgers (exact current balances)
        obs['ledgers'] = copy.deepcopy(env_state['agent']['ledgers'])
        
        # Total profit
        obs['total_profit'] = env_state['agent']['total_profit']
        
        # Exchange rates with noise
        true_rates = env_state['market']['exchange_matrices']['true_rates']
        noise_reduction = env_state['market']['exchange_matrices']['noise_reduction']
        
        observed_rates = {}
        for rate_pair, true_rate in true_rates.items():
            noise_std = self.config.get('noise_level', 0.05) * noise_reduction[rate_pair]
            noise = np.random.normal(0, noise_std)
            observed_rates[rate_pair] = max(0.1, true_rate + noise)
            
        obs['exchange_rates'] = observed_rates
        
        # Embargo risks (exact values)
        obs['embargo_risks'] = copy.deepcopy(env_state['market']['embargo_risks'])
        
        # Time information
        obs['timestep'] = env_state['timestep'] + 1
        obs['max_steps'] = env_state['globals']['max_steps']
        obs['remaining_steps'] = env_state['globals']['max_steps'] - env_state['timestep']
        
        # Available dimensions and items for action validation
        obs['dimensions'] = env_state['globals']['dimensions']
        obs['item_categories'] = env_state['globals']['item_categories']
        
        return obs