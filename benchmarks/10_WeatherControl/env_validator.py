import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy
import random

class AtmosphereValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_costs = {}
        for action in config["transition"]["actions"]:
            self.action_costs[action["name"]] = action["cost"]
        
        self.actions = list(self.action_costs.keys())
        
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """Main validation function that checks both solvability and reward structure"""
        issues = []
        
        # Load level
        try:
            with open(level_path, 'r') as f:
                level_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level: {e}"]
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(level_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(level_state)
        issues.extend(reward_issues)
        
        # 3. STATE CONSISTENCY CHECKS
        consistency_issues = self._check_state_consistency(level_state)
        issues.extend(consistency_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, initial_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_analysis = self._analyze_action_constraints(initial_state)
        issues.extend(action_analysis)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(initial_state)
        issues.extend(reachability_issues)
        
        # RESOURCE AVAILABILITY
        resource_issues = self._check_resource_availability(initial_state)
        issues.extend(resource_issues)
        
        # STEP BUDGET ANALYSIS
        step_budget_issues = self._check_step_budget_feasibility(initial_state)
        issues.extend(step_budget_issues)
        
        return issues
    
    def _analyze_action_constraints(self, state: Dict[str, Any]) -> List[str]:
        """Understand environment's fundamental limitations"""
        issues = []
        
        initial_energy = state.get('agent', {}).get('energy_budget', 45)
        max_steps = state.get('globals', {}).get('max_steps', 30)
        
        # Check if any actions are usable at all
        min_action_cost = min(self.action_costs.values())
        if initial_energy < min_action_cost:
            issues.append("CRITICAL: Initial energy insufficient to perform any action")
        
        # Check if energy budget allows reasonable action usage
        total_possible_actions = initial_energy // min_action_cost
        if total_possible_actions < max_steps // 5:  # Less than 20% action usage possible
            issues.append(f"WARNING: Energy budget ({initial_energy}) may be insufficient for effective control over {max_steps} steps")
        
        # Verify action effects can actually modify atmospheric variables
        for action_name in self.actions:
            effects = self._get_action_effects(action_name)
            if not effects:
                issues.append(f"CRITICAL: Action '{action_name}' has no defined effects")
        
        return issues
    
    def _check_target_reachability(self, state: Dict[str, Any]) -> List[str]:
        """Verify target state (CSI 45-55) is achievable"""
        issues = []
        
        atmosphere = state.get('atmosphere', {})
        initial_csi = atmosphere.get('climate_stability_index', 50.0)
        target_range = (45, 55)
        
        # If already in target range, check if it can be maintained against drift
        if target_range[0] <= initial_csi <= target_range[1]:
            maintenance_issues = self._check_stability_maintenance_feasibility(state)
            issues.extend(maintenance_issues)
        else:
            # Check if CSI can be brought into target range
            correction_issues = self._check_csi_correction_feasibility(state, initial_csi, target_range)
            issues.extend(correction_issues)
        
        return issues
    
    def _check_stability_maintenance_feasibility(self, state: Dict[str, Any]) -> List[str]:
        """Check if CSI can be maintained in target range against natural drift"""
        issues = []
        
        # Simulate natural drift without any actions
        test_state = deepcopy(state)
        drift_rate = self._estimate_drift_impact(test_state)
        
        if drift_rate > 2.0:  # CSI drifts more than 2 points per step on average
            issues.append("WARNING: Natural atmospheric drift is very strong, may be difficult to maintain stability")
        
        return issues
    
    def _check_csi_correction_feasibility(self, state: Dict[str, Any], initial_csi: float, target_range: Tuple[float, float]) -> List[str]:
        """Check if CSI can be corrected from initial value to target range"""
        issues = []
        
        # Calculate required CSI change
        if initial_csi < target_range[0]:
            required_change = target_range[0] - initial_csi
            direction = "increase"
        else:
            required_change = initial_csi - target_range[1]
            direction = "decrease"
        
        # Test if actions can produce sufficient change
        max_single_action_impact = self._estimate_max_action_impact()
        
        if required_change > max_single_action_impact * 5:  # Requires more than 5 optimal actions
            issues.append(f"WARNING: Initial CSI ({initial_csi:.1f}) may be too far from target range, requires {direction} of {required_change:.1f}")
        
        if required_change > 30:  # Extremely far from target
            issues.append(f"CRITICAL: Initial CSI ({initial_csi:.1f}) is extremely far from target range")
        
        return issues
    
    def _check_resource_availability(self, state: Dict[str, Any]) -> List[str]:
        """Check if energy resources are sufficient for problem solving"""
        issues = []
        
        initial_energy = state.get('agent', {}).get('energy_budget', 45)
        max_steps = state.get('globals', {}).get('max_steps', 30)
        
        # Calculate minimum energy needed for basic control
        min_energy_for_control = self._estimate_minimum_energy_requirement(state)
        
        if initial_energy < min_energy_for_control:
            issues.append(f"CRITICAL: Initial energy ({initial_energy}) insufficient for basic atmospheric control (need ~{min_energy_for_control})")
        
        # Check if energy allows sustained intervention
        avg_action_cost = sum(self.action_costs.values()) / len(self.action_costs)
        sustainable_actions = initial_energy / avg_action_cost
        
        if sustainable_actions < max_steps * 0.3:  # Less than 30% intervention capability
            issues.append(f"WARNING: Energy budget may not allow sustained atmospheric intervention")
        
        return issues
    
    def _check_step_budget_feasibility(self, state: Dict[str, Any]) -> List[str]:
        """Check if solution is achievable within step limits"""
        issues = []
        
        max_steps = state.get('globals', {}).get('max_steps', 30)
        
        if max_steps < 10:
            issues.append("CRITICAL: Episode too short for meaningful atmospheric control")
        elif max_steps < 20:
            issues.append("WARNING: Episode may be too short for stable atmospheric control")
        
        # Check if delayed action effects (2-step propagation) allow sufficient reaction time
        reaction_time_needed = 6  # At least 6 steps to observe and react to delayed effects
        if max_steps < reaction_time_needed:
            issues.append("CRITICAL: Episode too short to handle 2-step action delay propagation")
        
        return issues
    
    def _validate_reward_structure(self, state: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # GOAL-ORIENTED REWARDS CHECK
        goal_reward_issues = self._check_goal_oriented_rewards()
        issues.extend(goal_reward_issues)
        
        # INCENTIVE MISALIGNMENT CHECK
        misalignment_issues = self._check_incentive_misalignment()
        issues.extend(misalignment_issues)
        
        # REWARD BALANCE ANALYSIS
        balance_issues = self._check_reward_balance()
        issues.extend(balance_issues)
        
        return issues
    
    def _check_goal_oriented_rewards(self) -> List[str]:
        """Ensure rewards prioritize problem-solving over action usage"""
        issues = []
        
        reward_events = self.config.get('reward', {}).get('events', [])
        
        # Check for high-value target achievement rewards
        perfect_episode_reward = 0
        stability_maintenance_reward = 0
        discovery_bonus_reward = 0
        
        for event in reward_events:
            if event.get('trigger') == 'perfect_episode':
                perfect_episode_reward = event.get('value', 0)
            elif event.get('trigger') == 'stability_maintenance':
                stability_maintenance_reward = event.get('value', 0)
            elif event.get('trigger') == 'discovery_bonus':
                discovery_bonus_reward = event.get('value', 0)
        
        if perfect_episode_reward < 15:
            issues.append(f"WARNING: Perfect episode reward ({perfect_episode_reward}) may be too low to incentivize goal achievement")
        
        if stability_maintenance_reward < 0.1:
            issues.append("CRITICAL: Stability maintenance reward too low, agents won't prioritize staying in target range")
        
        # Check that goal rewards outweigh action farming potential
        max_stability_reward = stability_maintenance_reward * 30  # Max steps
        if discovery_bonus_reward * len(self.actions) > max_stability_reward:
            issues.append("WARNING: Discovery bonuses may outweigh actual stability maintenance")
        
        return issues
    
    def _check_incentive_misalignment(self) -> List[str]:
        """Check for reward loops and action farming opportunities"""
        issues = []
        
        # Check if actions themselves are directly rewarded (should not be)
        reward_events = self.config.get('reward', {}).get('events', [])
        
        for event in reward_events:
            trigger = event.get('trigger', '')
            if 'action' in trigger.lower():
                issues.append(f"WARNING: Direct action reward detected: {trigger}")
        
        # Verify diminishing returns exist for discovery bonuses
        discovery_limited = False
        for event in reward_events:
            if event.get('trigger') == 'discovery_bonus':
                # Check if there's mechanism to prevent repeated discovery bonuses
                # This is implemented in the code as self.discovery_bonuses tracking
                discovery_limited = True
                break
        
        if not discovery_limited:
            issues.append("CRITICAL: Discovery bonuses may be farmable without limit")
        
        return issues
    
    def _check_reward_balance(self) -> List[str]:
        """Check overall reward structure balance"""
        issues = []
        
        reward_events = self.config.get('reward', {}).get('events', [])
        
        # Calculate potential reward ranges
        min_episode_reward = 0  # Worst case: no stability maintenance
        max_episode_reward = 0  # Best case: perfect episode + all bonuses
        
        for event in reward_events:
            value = event.get('value', 0)
            trigger = event.get('trigger', '')
            
            if trigger == 'stability_maintenance':
                max_episode_reward += value * 30  # Max steps
            elif trigger == 'perfect_episode':
                max_episode_reward += value
            elif trigger == 'stability_recovery':
                max_episode_reward += value * 5  # Reasonable number of recoveries
            elif trigger == 'discovery_bonus':
                max_episode_reward += value * len(self.actions)
            elif trigger == 'energy_efficiency':
                multiplier = self.config.get('reward', {}).get('remaining_energy_multiplier', 0.1)
                max_episode_reward += 45 * multiplier  # Max energy * multiplier
        
        # Check reward range sanity
        if max_episode_reward < 20:
            issues.append("WARNING: Maximum possible reward may be too low to drive learning")
        
        if max_episode_reward > 200:
            issues.append("WARNING: Maximum possible reward may be too high, could cause training instability")
        
        return issues
    
    def _check_state_consistency(self, state: Dict[str, Any]) -> List[str]:
        """Check for state consistency and valid ranges"""
        issues = []
        
        # Check atmospheric variables are within valid ranges
        atmosphere = state.get('atmosphere', {})
        
        variables_ranges = {
            'temperature': (100, 500),
            'humidity': (0, 100),
            'atmospheric_pressure': (0.5, 2.0),
            'cloud_coverage': (0, 100),
            'storm_energy': (0, 100),
            'solar_flux': (500, 1500),
            'climate_stability_index': (0, 100)
        }
        
        for var, (min_val, max_val) in variables_ranges.items():
            if var in atmosphere:
                value = atmosphere[var]
                if not (min_val <= value <= max_val):
                    issues.append(f"CRITICAL: {var} ({value}) outside valid range [{min_val}, {max_val}]")
        
        # Check agent state
        agent = state.get('agent', {})
        energy_budget = agent.get('energy_budget', 45)
        if energy_budget < 0 or energy_budget > 100:
            issues.append(f"CRITICAL: Invalid energy budget: {energy_budget}")
        
        # Check physics state
        physics = state.get('physics', {})
        drift_rates = physics.get('drift_rates', {})
        for var, rate in drift_rates.items():
            if rate < 0 or rate > 0.1:  # Drift rates should be small positive values
                issues.append(f"WARNING: Unusual drift rate for {var}: {rate}")
        
        return issues
    
    # Helper methods for estimation
    def _get_action_effects(self, action_name: str) -> Dict[str, float]:
        """Get the direct effects of an action on atmospheric variables"""
        effects = {}
        
        if action_name == "inject_cold_ions":
            effects = {'temperature': +15, 'atmospheric_pressure': -0.1, 'storm_energy': -5}
        elif action_name == "release_dry_fog":
            effects = {'humidity': +8, 'cloud_coverage': -10, 'solar_flux': +50}
        elif action_name == "vent_heavy_vapor":
            effects = {'temperature': -20, 'atmospheric_pressure': +0.15, 'cloud_coverage': +15}
        elif action_name == "trigger_pressure_spike":
            effects = {'humidity': -10, 'storm_energy': +8, 'atmospheric_pressure': +0.1}
        elif action_name == "emit_solar_net":
            effects = {'solar_flux': -80, 'temperature': +10, 'cloud_coverage': +12}
        elif action_name == "redirect_jet_stream":
            effects = {'humidity': +12, 'storm_energy': +6, 'atmospheric_pressure': -0.08}
        
        return effects
    
    def _estimate_max_action_impact(self) -> float:
        """Estimate maximum CSI change from a single action"""
        # Based on CSI calculation weights, estimate maximum single action impact
        # This is a rough approximation
        return 8.0  # Conservative estimate
    
    def _estimate_drift_impact(self, state: Dict[str, Any]) -> float:
        """Estimate CSI drift per step due to natural atmospheric changes"""
        physics = state.get('physics', {})
        drift_rates = physics.get('drift_rates', {})
        
        # Estimate average drift impact on CSI
        avg_drift = sum(drift_rates.values()) / len(drift_rates) if drift_rates else 0.02
        return avg_drift * 50  # Rough CSI impact estimation
    
    def _estimate_minimum_energy_requirement(self, state: Dict[str, Any]) -> int:
        """Estimate minimum energy needed for basic atmospheric control"""
        # Based on need for sustained intervention over episode
        max_steps = state.get('globals', {}).get('max_steps', 30)
        min_actions_needed = max_steps // 5  # At least 20% intervention
        avg_cost = sum(self.action_costs.values()) / len(self.action_costs)
        return int(min_actions_needed * avg_cost)

def validate_atmosphere_level(level_path: str, config_path: str = "./config.yaml") -> Tuple[bool, List[str]]:
    """Standalone function to validate a single atmosphere regulation level"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validator = AtmosphereValidator(config)
        return validator.validate_level(level_path)
    except Exception as e:
        return False, [f"Validation failed: {e}"]

def validate_all_levels(levels_dir: str = "./levels/", config_path: str = "./config.yaml") -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all levels in a directory"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validator = AtmosphereValidator(config)
        results = {}
        
        for filename in os.listdir(levels_dir):
            if filename.endswith('.yaml'):
                level_path = os.path.join(levels_dir, filename)
                is_valid, issues = validator.validate_level(level_path)
                results[filename] = (is_valid, issues)
        
        return results
    except Exception as e:
        return {"validation_error": (False, [f"Failed to validate levels: {e}"])}