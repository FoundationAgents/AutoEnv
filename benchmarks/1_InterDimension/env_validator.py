import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
import copy

class InterdimensionalMarketValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profit_target = config.get('misc', {}).get('profit_target', 100.0)
        self.max_steps = config.get('termination', {}).get('max_steps', 40)
        self.hedge_fee = config.get('misc', {}).get('hedge_fee', 5.0)
        self.embargo_threshold = config.get('misc', {}).get('embargo_threshold', 100.0)
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Main validation function that checks level solvability and reward structure"""
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. BASIC STATE CONSISTENCY
        consistency_issues = self._check_state_consistency(world_state)
        issues.extend(consistency_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(world_state)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(world_state)
        issues.extend(reachability_issues)
        
        # COMMON IMPOSSIBLE PATTERNS
        pattern_issues = self._check_impossible_patterns(world_state)
        issues.extend(pattern_issues)
        
        return issues
    
    def _analyze_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Understand environment's fundamental limitations"""
        issues = []
        
        # Check if agent has any inventory to trade
        inventory = world_state['agent']['inventory']
        total_items = sum(inventory.values())
        if total_items == 0:
            issues.append("CRITICAL: No inventory items available - impossible to generate profit through trading")
        
        # Check if agent has any initial credits for conversions/hedging
        ledgers = world_state['agent']['ledgers']
        total_credits = sum(ledgers.values())
        if total_credits <= 0:
            issues.append("CRITICAL: No initial credits - agent cannot perform conversions or hedging")
        
        # Check if hedge fees are affordable
        max_single_ledger = max(ledgers.values()) if ledgers.values() else 0
        if max_single_ledger < self.hedge_fee:
            issues.append(f"WARNING: No dimension has enough credits ({max_single_ledger:.2f}) to afford hedge fee ({self.hedge_fee})")
        
        # Validate exchange rate drift parameters
        drift_params = world_state['market']['exchange_matrices']['drift_params']
        for rate_pair, params in drift_params.items():
            if len(params) != 3:
                issues.append(f"CRITICAL: Invalid drift parameters for {rate_pair}: expected 3 values, got {len(params)}")
            else:
                amplitude, frequency, phase = params
                if amplitude <= 0:
                    issues.append(f"WARNING: Zero amplitude for {rate_pair} - no exchange rate variation possible")
        
        return issues
    
    def _check_target_reachability(self, world_state: Dict[str, Any]) -> List[str]:
        """Verify target state is actually achievable"""
        issues = []
        
        # Calculate maximum possible profit through perfect arbitrage
        max_profit = self._estimate_maximum_profit(world_state)
        
        if max_profit < self.profit_target:
            issues.append(f"CRITICAL: Maximum possible profit ({max_profit:.2f}) is less than target ({self.profit_target})")
        
        # Check if profit target is reachable within step limits
        min_steps_needed = self._estimate_minimum_steps_needed(world_state)
        if min_steps_needed > self.max_steps:
            issues.append(f"CRITICAL: Minimum steps needed ({min_steps_needed}) exceeds max steps ({self.max_steps})")
        
        # Check embargo risk constraints
        embargo_issues = self._check_embargo_constraints(world_state)
        issues.extend(embargo_issues)
        
        return issues
    
    def _estimate_maximum_profit(self, world_state: Dict[str, Any]) -> float:
        """Calculate theoretical maximum profit possible"""
        inventory = world_state['agent']['inventory']
        drift_params = world_state['market']['exchange_matrices']['drift_params']
        
        # Calculate maximum value from inventory arbitrage
        # Assume base item value of 10 credits per item (from environment code)
        total_items = sum(inventory.values())
        base_inventory_value = total_items * 10
        
        # Find maximum exchange rate amplitude for arbitrage potential
        max_amplitude = 0
        for rate_pair, params in drift_params.items():
            amplitude = params[0]
            max_amplitude = max(max_amplitude, amplitude)
        
        # Theoretical maximum profit: base value * maximum rate variation
        # Plus initial credit advantage through optimal conversions
        initial_credits = sum(world_state['agent']['ledgers'].values())
        max_arbitrage_gain = base_inventory_value * max_amplitude
        max_conversion_gain = initial_credits * max_amplitude
        
        return max_arbitrage_gain + max_conversion_gain
    
    def _estimate_minimum_steps_needed(self, world_state: Dict[str, Any]) -> int:
        """Estimate minimum steps needed for optimal solution"""
        inventory = world_state['agent']['inventory']
        
        # Count non-zero inventory categories
        active_categories = sum(1 for qty in inventory.values() if qty > 0)
        
        # Minimum steps: 1 research + 1 trade per category + conversions
        # This is a conservative estimate
        min_research_steps = 1  # At least one research to understand rates
        min_trading_steps = active_categories  # One trade per item category
        min_conversion_steps = 2  # At least some conversions needed
        
        return min_research_steps + min_trading_steps + min_conversion_steps
    
    def _check_embargo_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if embargo risks allow for profitable trading"""
        issues = []
        
        embargo_risks = world_state['market']['embargo_risks']
        
        # Check if any dimension is too close to embargo threshold
        for dimension, risk in embargo_risks.items():
            if risk >= 90:
                issues.append(f"CRITICAL: {dimension} dimension starts with risk {risk:.1f}% - too close to embargo threshold")
            elif risk >= 80:
                issues.append(f"WARNING: {dimension} dimension starts with high risk {risk:.1f}%")
        
        # Check if agent has donation capability to manage risks
        inventory = world_state['agent']['inventory']
        total_donation_capacity = sum(inventory.values())
        
        if total_donation_capacity == 0:
            for dimension, risk in embargo_risks.items():
                if risk > 60:
                    issues.append(f"WARNING: High risk {dimension} ({risk:.1f}%) with no donation items available")
        
        return issues
    
    def _check_impossible_patterns(self, world_state: Dict[str, Any]) -> List[str]:
        """Check for common impossible patterns"""
        issues = []
        
        # Pattern 1: Circular dependency in exchange rates
        if self._has_circular_dependencies(world_state):
            issues.append("WARNING: Potential circular dependencies in exchange rate structure")
        
        # Pattern 2: Insufficient resources for required operations
        if self._insufficient_resources(world_state):
            issues.append("CRITICAL: Insufficient resources to perform necessary operations")
        
        # Pattern 3: Exchange rate parameters that create impossible scenarios
        if self._invalid_exchange_parameters(world_state):
            issues.append("CRITICAL: Exchange rate parameters create impossible trading scenarios")
        
        return issues
    
    def _has_circular_dependencies(self, world_state: Dict[str, Any]) -> bool:
        """Check for circular dependencies in trading requirements"""
        # In this environment, circular dependencies are less likely due to the nature of arbitrage
        # But we check if exchange rates are set up in a way that prevents profitable cycles
        
        drift_params = world_state['market']['exchange_matrices']['drift_params']
        
        # Check if all drift parameters are effectively zero (no variation)
        all_zero_amplitude = all(params[0] == 0 for params in drift_params.values())
        
        return all_zero_amplitude and sum(world_state['agent']['ledgers'].values()) < self.profit_target
    
    def _insufficient_resources(self, world_state: Dict[str, Any]) -> bool:
        """Check if there are sufficient resources for operations"""
        inventory = world_state['agent']['inventory']
        ledgers = world_state['agent']['ledgers']
        
        # Need either inventory items or credits to operate
        has_items = sum(inventory.values()) > 0
        has_credits = sum(ledgers.values()) > 0
        
        return not (has_items or has_credits)
    
    def _invalid_exchange_parameters(self, world_state: Dict[str, Any]) -> bool:
        """Check if exchange parameters create impossible scenarios"""
        drift_params = world_state['market']['exchange_matrices']['drift_params']
        
        for rate_pair, params in drift_params.items():
            amplitude, frequency, phase = params
            
            # Check for invalid numerical values
            if not all(np.isfinite([amplitude, frequency, phase])):
                return True
            
            # Check for negative frequency (could cause issues)
            if frequency < 0:
                return True
        
        return False
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Check goal-oriented rewards vs action grinding
        reward_issues = self._check_reward_alignment()
        issues.extend(reward_issues)
        
        # Check for exploitable reward loops
        exploit_issues = self._check_reward_exploits(world_state)
        issues.extend(exploit_issues)
        
        return issues
    
    def _check_reward_alignment(self) -> List[str]:
        """Check if rewards prioritize problem-solving over action usage"""
        issues = []
        
        reward_config = self.config.get('reward', {})
        
        # Goal achievement should be highest reward
        goal_reward = reward_config.get('goal_rewards', {}).get('target_reached', 0)
        profit_per_credit = reward_config.get('profit_rewards', {}).get('per_credit', 0)
        stability_per_step = reward_config.get('stability_rewards', {}).get('per_step', 0)
        
        if goal_reward <= 0:
            issues.append("CRITICAL: No goal achievement reward - agents won't prioritize main objective")
        
        # Goal reward should be significantly higher than per-step rewards
        max_step_rewards = stability_per_step * self.max_steps
        if goal_reward <= max_step_rewards:
            issues.append(f"WARNING: Goal reward ({goal_reward}) not significantly higher than max step rewards ({max_step_rewards:.1f})")
        
        # Profit rewards should be meaningful but not excessive
        if profit_per_credit <= 0:
            issues.append("WARNING: No profit rewards - agents won't be incentivized to optimize profit")
        elif profit_per_credit > 5:
            issues.append("WARNING: Profit rewards too high - may encourage risky behavior")
        
        return issues
    
    def _check_reward_exploits(self, world_state: Dict[str, Any]) -> List[str]:
        """Check for exploitable reward patterns"""
        issues = []
        
        reward_config = self.config.get('reward', {})
        
        # Check if research can be spammed for rewards
        research_reward = reward_config.get('research_rewards', {}).get('discovery', 0)
        if research_reward > 5:
            issues.append("WARNING: Research rewards too high - agents may spam research actions")
        
        # Check if donation rewards could be exploited
        # (No direct donation rewards, but check if inventory allows sustainable donations)
        inventory = world_state['agent']['inventory']
        total_items = sum(inventory.values())
        
        if total_items > 100:  # Arbitrary threshold for "too many items"
            issues.append("WARNING: Very high inventory count - may allow excessive donation exploitation")
        
        # Check fairness bonus sustainability
        fairness_reward = reward_config.get('fairness_rewards', {}).get('all_positive', 0)
        if fairness_reward > 10:
            issues.append("WARNING: Fairness bonus too high - may discourage profitable but risky strategies")
        
        return issues
    
    def _check_state_consistency(self, world_state: Dict[str, Any]) -> List[str]:
        """Basic state consistency checks"""
        issues = []
        
        # Check all required fields exist
        required_fields = [
            'agent', 'market', 'globals', 'timestep', 'episode_complete'
        ]
        
        for field in required_fields:
            if field not in world_state:
                issues.append(f"CRITICAL: Missing required field: {field}")
        
        # Check agent state
        if 'agent' in world_state:
            agent_issues = self._check_agent_state(world_state['agent'])
            issues.extend(agent_issues)
        
        # Check market state
        if 'market' in world_state:
            market_issues = self._check_market_state(world_state['market'])
            issues.extend(market_issues)
        
        return issues
    
    def _check_agent_state(self, agent_state: Dict[str, Any]) -> List[str]:
        """Validate agent state consistency"""
        issues = []
        
        # Check inventory
        if 'inventory' not in agent_state:
            issues.append("CRITICAL: Missing agent inventory")
        else:
            for item, qty in agent_state['inventory'].items():
                if qty < 0:
                    issues.append(f"CRITICAL: Negative inventory for {item}: {qty}")
        
        # Check ledgers
        if 'ledgers' not in agent_state:
            issues.append("CRITICAL: Missing agent ledgers")
        else:
            for dimension, balance in agent_state['ledgers'].items():
                if not np.isfinite(balance):
                    issues.append(f"CRITICAL: Invalid ledger balance for {dimension}: {balance}")
        
        return issues
    
    def _check_market_state(self, market_state: Dict[str, Any]) -> List[str]:
        """Validate market state consistency"""
        issues = []
        
        # Check exchange matrices
        if 'exchange_matrices' not in market_state:
            issues.append("CRITICAL: Missing exchange matrices")
        else:
            exchange_issues = self._check_exchange_matrices(market_state['exchange_matrices'])
            issues.extend(exchange_issues)
        
        # Check embargo risks
        if 'embargo_risks' not in market_state:
            issues.append("CRITICAL: Missing embargo risks")
        else:
            for dimension, risk in market_state['embargo_risks'].items():
                if not (0 <= risk <= 100):
                    issues.append(f"WARNING: Embargo risk for {dimension} out of range: {risk}")
        
        return issues
    
    def _check_exchange_matrices(self, exchange_matrices: Dict[str, Any]) -> List[str]:
        """Validate exchange matrix structure"""
        issues = []
        
        required_matrices = ['true_rates', 'observed_rates', 'drift_params', 'noise_reduction']
        
        for matrix in required_matrices:
            if matrix not in exchange_matrices:
                issues.append(f"CRITICAL: Missing exchange matrix: {matrix}")
        
        # Check rate consistency
        if 'true_rates' in exchange_matrices and 'drift_params' in exchange_matrices:
            true_rates = exchange_matrices['true_rates']
            drift_params = exchange_matrices['drift_params']
            
            if set(true_rates.keys()) != set(drift_params.keys()):
                issues.append("CRITICAL: Mismatch between true_rates and drift_params keys")
        
        return issues

def validate_market_level(world_file_path: str, config_file_path: str) -> Tuple[bool, List[str]]:
    """Convenience function to validate a market level file"""
    
    # Load world state
    with open(world_file_path, 'r') as f:
        world_state = yaml.safe_load(f)
    
    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create validator and check
    validator = InterdimensionalMarketValidator(config)
    return validator.validate_level(world_state)

# Example usage function
def validate_generated_levels(levels_dir: str, config_path: str) -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all generated levels in a directory"""
    import os
    import glob
    
    results = {}
    
    for level_file in glob.glob(os.path.join(levels_dir, "*.yaml")):
        level_name = os.path.basename(level_file)
        try:
            is_valid, issues = validate_market_level(level_file, config_path)
            results[level_name] = (is_valid, issues)
        except Exception as e:
            results[level_name] = (False, [f"CRITICAL: Validation failed with exception: {str(e)}"])
    
    return results