from base.env.base_env import SkinEnv
from env_obs import MarketObservationPolicy
from env_generate import MarketWorldGenerator
import yaml
import numpy as np
import copy
import random
from typing import Dict, Any, Optional, Tuple, List

class InterdimensionalMarketEnv(SkinEnv):
    def __init__(self, env_id: int):
        # Initialize observation policy first
        obs_policy = MarketObservationPolicy({'noise_level': 0.05})
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        with open("./config.yaml", 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = MarketWorldGenerator(str(self.env_id), self.configs)
        
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided in load mode")
            self._state = self._load_world(world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
        
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        with open(f"./levels/{world_id}.yaml", 'r') as f:
            return yaml.safe_load(f)
            
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(copy.deepcopy(self._state))
        
        action_name = action.get('action')
        params = action.get('params', {})
        
        # Update exchange rates based on drift patterns
        self._update_exchange_rates()
        
        # Process action
        if action_name == "PROPOSE_TRADE":
            self._process_trade(params)
        elif action_name == "CONVERT_CREDITS":
            self._process_conversion(params)
        elif action_name == "HEDGE":
            self._process_hedge(params)
        elif action_name == "RESEARCH":
            self._process_research(params)
        elif action_name == "DONATE":
            self._process_donation(params)
        else:
            self._last_action_result = f"Unknown action: {action_name}"
            
        # Update hedge counters
        self._update_hedge_status()
        
        # Update timestep
        self._state['timestep'] = self._t + 1
        
        return self._state
    
    def _update_exchange_rates(self):
        for rate_pair, drift_params in self._state['market']['exchange_matrices']['drift_params'].items():
            amplitude, frequency, phase = drift_params
            
            # Check if this dimension is hedged
            dimension = rate_pair.split('_')[0]
            if self._state['market']['hedge_status'][dimension] > 0:
                continue  # Skip update if hedged
                
            # Calculate new rate using sinusoidal pattern
            base_rate = 1.0  # Could be made configurable
            new_rate = base_rate + amplitude * np.sin(frequency * self._t + phase)
            new_rate = max(0.1, new_rate)  # Prevent negative rates
            
            self._state['market']['exchange_matrices']['true_rates'][rate_pair] = new_rate
    
    def _process_trade(self, params):
        item_category = params.get('item_category')
        source_dimension = params.get('source_dimension')
        target_dimension = params.get('target_dimension')
        quantity = params.get('quantity', 0)
        
        # Validate parameters
        if (item_category not in self._state['agent']['inventory'] or
            source_dimension not in self._state['globals']['dimensions'] or
            target_dimension not in self._state['globals']['dimensions'] or
            quantity <= 0 or
            self._state['agent']['inventory'][item_category] < quantity):
            self._last_action_result = "Invalid trade parameters"
            return
            
        # Calculate item value (simplified - could be made more complex)
        item_value = quantity * 10  # Base value per item
        
        # Get exchange rate from source to target
        rate_key = f"{source_dimension}_{target_dimension}"
        if rate_key not in self._state['market']['exchange_matrices']['true_rates']:
            self._last_action_result = "Invalid currency pair"
            return
            
        exchange_rate = self._state['market']['exchange_matrices']['true_rates'][rate_key]
        
        # Execute trade
        self._state['agent']['inventory'][item_category] -= quantity
        self._state['agent']['ledgers'][source_dimension] += item_value
        self._state['agent']['ledgers'][target_dimension] -= item_value * exchange_rate
        
        # Update embargo risk based on trade fairness
        trade_advantage = abs(exchange_rate - 1.0)
        if exchange_rate > 1.0:  # Source dimension benefits
            self._state['market']['embargo_risks'][target_dimension] += trade_advantage * 5
        else:  # Target dimension benefits
            self._state['market']['embargo_risks'][source_dimension] += trade_advantage * 5
            
        self._last_action_result = f"Trade executed: {quantity} {item_category}"
    
    def _process_conversion(self, params):
        source_currency = params.get('source_currency')
        target_currency = params.get('target_currency')
        amount = params.get('amount', 0)
        
        if (source_currency not in self._state['globals']['dimensions'] or
            target_currency not in self._state['globals']['dimensions'] or
            amount <= 0 or
            self._state['agent']['ledgers'][source_currency] < amount):
            self._last_action_result = "Invalid conversion parameters"
            return
            
        rate_key = f"{source_currency}_{target_currency}"
        if rate_key not in self._state['market']['exchange_matrices']['true_rates']:
            self._last_action_result = "Invalid currency pair"
            return
            
        exchange_rate = self._state['market']['exchange_matrices']['true_rates'][rate_key]
        
        # Execute conversion
        self._state['agent']['ledgers'][source_currency] -= amount
        self._state['agent']['ledgers'][target_currency] += amount * exchange_rate
        
        self._last_action_result = f"Converted {amount} {source_currency} to {target_currency}"
    
    def _process_hedge(self, params):
        dimension = params.get('dimension')
        hedge_fee = self.configs.get('misc', {}).get('hedge_fee', 5.0)
        
        if (dimension not in self._state['globals']['dimensions'] or
            self._state['agent']['ledgers'][dimension] < hedge_fee):
            self._last_action_result = "Cannot hedge: insufficient funds"
            return
            
        # Deduct fee and activate hedge
        self._state['agent']['ledgers'][dimension] -= hedge_fee
        self._state['market']['hedge_status'][dimension] = self.configs.get('misc', {}).get('hedge_duration', 5)
        
        self._last_action_result = f"Hedged {dimension} for 5 steps"
    
    def _process_research(self, params):
        # Randomly select a rate pair to improve
        rate_pairs = list(self._state['market']['exchange_matrices']['noise_reduction'].keys())
        selected_pair = random.choice(rate_pairs)
        
        # Reduce noise for this pair
        current_reduction = self._state['market']['exchange_matrices']['noise_reduction'][selected_pair]
        improvement = 0.3  # 30% improvement
        new_reduction = max(0.1, current_reduction - improvement)
        self._state['market']['exchange_matrices']['noise_reduction'][selected_pair] = new_reduction
        
        self._last_action_result = f"Research improved accuracy for {selected_pair}"
    
    def _process_donation(self, params):
        item_category = params.get('item_category')
        target_dimension = params.get('target_dimension')
        quantity = params.get('quantity', 0)
        
        if (item_category not in self._state['agent']['inventory'] or
            target_dimension not in self._state['globals']['dimensions'] or
            quantity <= 0 or
            self._state['agent']['inventory'][item_category] < quantity):
            self._last_action_result = "Invalid donation parameters"
            return
            
        # Execute donation
        self._state['agent']['inventory'][item_category] -= quantity
        
        # Reduce embargo risk
        risk_reduction = quantity * 2  # 2 points per donated item
        self._state['market']['embargo_risks'][target_dimension] = max(0, 
            self._state['market']['embargo_risks'][target_dimension] - risk_reduction)
            
        self._last_action_result = f"Donated {quantity} {item_category} to {target_dimension}"
    
    def _update_hedge_status(self):
        for dimension in self._state['market']['hedge_status']:
            if self._state['market']['hedge_status'][dimension] > 0:
                self._state['market']['hedge_status'][dimension] -= 1
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        # Calculate current profit
        current_total = sum(self._state['agent']['ledgers'].values())
        if len(self._history) > 0:
            previous_total = sum(self._history[-1]['agent']['ledgers'].values())
            profit_gain = current_total - previous_total
            if profit_gain > 0:
                profit_reward = profit_gain * 1.0
                total_reward += profit_reward
                events.append("profit_gained")
                reward_info['profit_rewards'] = profit_reward
        
        # Update total profit
        self._state['agent']['total_profit'] = current_total - sum(self.configs['state_template']['agent']['ledgers'].values())
        
        # Stability bonus
        if all(risk < 80 for risk in self._state['market']['embargo_risks'].values()):
            stability_reward = 0.2
            total_reward += stability_reward
            events.append("stability_bonus")
            reward_info['stability_rewards'] = stability_reward
        
        # Fairness bonus
        if all(balance >= 0 for balance in self._state['agent']['ledgers'].values()):
            fairness_reward = 5.0
            total_reward += fairness_reward
            events.append("fairness_bonus")
            reward_info['fairness_rewards'] = fairness_reward
        
        # Research discovery bonus
        if action.get('action') == 'RESEARCH' and 'improved accuracy' in str(self._last_action_result):
            research_reward = 3.0
            total_reward += research_reward
            events.append("research_discovery")
            reward_info['research_rewards'] = research_reward
        
        # Goal achievement bonus
        if (self._state['agent']['total_profit'] >= 100.0 and
            (len(self._history) == 0 or self._history[-1]['agent']['total_profit'] < 100.0)):
            goal_reward = 50.0
            total_reward += goal_reward
            events.append("goal_achieved")
            reward_info['goal_rewards'] = goal_reward
        
        # Update accumulated rewards
        self._state['agent']['accumulated_rewards'] = self._state['agent'].get('accumulated_rewards', 0) + total_reward
        
        return total_reward, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        inventory_display = ", ".join([f"{k}:{v}" for k, v in omega['inventory'].items() if v > 0])
        if not inventory_display:
            inventory_display = "Empty"
            
        exchange_display = "\n    ".join([f"{k}: {v:.3f}" for k, v in omega['exchange_rates'].items()])
        
        return f"""=== Interdimensional Trading Terminal - Step {omega['timestep']}/{omega['max_steps']} ===

AGENT STATUS:
Inventory: {inventory_display}
Ledgers: Mass={omega['ledgers']['mass']:.2f} | Entropy={omega['ledgers']['entropy']:.2f} | Historical={omega['ledgers']['historical']:.2f}
Total Profit: {omega['total_profit']:.2f} credits

MARKET CONDITIONS:
Exchange Rates (Observed):
    {exchange_display}

DIPLOMATIC STATUS:
Embargo Risks: Mass={omega['embargo_risks']['mass']:.1f}% | Entropy={omega['embargo_risks']['entropy']:.1f}% | Historical={omega['embargo_risks']['historical']:.1f}%

AVAILABLE ACTIONS:
- PROPOSE_TRADE(item_category, source_dimension, target_dimension, quantity)
- CONVERT_CREDITS(source_currency, target_currency, amount)
- HEDGE(dimension) - Fee: 5 credits
- RESEARCH() - Improve rate accuracy
- DONATE(item_category, target_dimension, quantity) - Reduce embargo risk"""
    
    def done(self, s_next: Dict[str, Any] = None) -> bool:
        if s_next is None:
            s_next = self._state
            
        # Check max steps
        if self._t >= self.configs['termination']['max_steps']:
            return True
            
        # Check embargo threshold
        if any(risk >= 100.0 for risk in s_next['market']['embargo_risks'].values()):
            return True
            
        # Check negative inventory or ledgers
        if any(quantity < 0 for quantity in s_next['agent']['inventory'].values()):
            return True
            
        if any(balance < 0 for balance in s_next['agent']['ledgers'].values()):
            return True
            
        return False