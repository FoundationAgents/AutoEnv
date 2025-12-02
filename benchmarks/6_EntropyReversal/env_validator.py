"""
Entropy Reversal Engineering Environment Level Validator

This validator ensures generated levels are solvable and have properly aligned reward structures
to prevent impossible puzzles and incentive misalignment issues.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import copy
import yaml
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    solvability_score: float
    reward_alignment_score: float

class EntropyReversalValidator:
    
    def __init__(self):
        self.max_steps = 40
        self.target_global_order = 100
        self.chaos_danger_threshold = 70
        self.chaos_collapse_threshold = 90
        self.baseline_order = 200
        
    def validate_level(self, world_state: Dict[str, Any]) -> ValidationResult:
        """
        Main validation function that checks level solvability and reward alignment
        """
        issues = []
        warnings = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE DESIGN VALIDATION
        reward_issues = self._check_reward_alignment(world_state)
        issues.extend(reward_issues)
        
        # 3. STATE VALIDITY CHECKS
        state_issues = self._check_state_validity(world_state)
        issues.extend(state_issues)
        
        # 4. CONSTRAINT VALIDATION
        constraint_warnings = self._check_constraints(world_state)
        warnings.extend(constraint_warnings)
        
        is_valid = len(issues) == 0
        solvability_score = self._calculate_solvability_score(world_state, solvability_issues)
        reward_score = self._calculate_reward_alignment_score(world_state, reward_issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            solvability_score=solvability_score,
            reward_alignment_score=reward_score
        )
    
    def _check_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for impossible puzzles by analyzing action constraints and reachability
        """
        issues = []
        
        # Extract initial state
        initial_global_order = world_state["globals"]["global_order_score"]
        initial_entropy_tokens = world_state["globals"]["entropy_tokens"]
        domains = world_state["domains"]
        
        # Calculate order gap to target
        order_gap = self.target_global_order - initial_global_order
        
        # 1. RESOURCE AVAILABILITY CHECK
        if initial_entropy_tokens <= 0:
            issues.append("CRITICAL: No initial entropy tokens available - impossible to perform any entropy operations")
        
        # 2. MINIMUM RESOURCE REQUIREMENT ANALYSIS
        min_tokens_needed = self._calculate_minimum_tokens_required(order_gap, domains)
        max_tokens_obtainable = self._calculate_maximum_tokens_obtainable(domains, initial_entropy_tokens)
        
        if min_tokens_needed > max_tokens_obtainable:
            issues.append(f"SOLVABILITY: Insufficient total resources - need {min_tokens_needed} tokens but can only obtain {max_tokens_obtainable}")
        
        # 3. CHAOS CONSTRAINT ANALYSIS
        chaos_issues = self._check_chaos_constraints(world_state, order_gap)
        issues.extend(chaos_issues)
        
        # 4. STEP BUDGET ANALYSIS
        min_steps_needed = self._estimate_minimum_steps(order_gap, domains)
        if min_steps_needed > self.max_steps:
            issues.append(f"SOLVABILITY: Insufficient step budget - need at least {min_steps_needed} steps but only have {self.max_steps}")
        
        # 5. ACTION CONSTRAINT ANALYSIS
        action_issues = self._check_action_constraints(world_state)
        issues.extend(action_issues)
        
        return issues
    
    def _calculate_minimum_tokens_required(self, order_gap: int, domains: Dict[str, Any]) -> int:
        """
        Calculate minimum entropy tokens needed to achieve target order
        """
        if order_gap <= 0:
            return 0
        
        # Best case: use reverse_entropy efficiently (1 token = 1 order)
        # But we need to account for chaos management costs
        base_tokens_for_order = order_gap
        
        # Estimate chaos management overhead
        # Each reverse_entropy operation creates spillover chaos
        num_operations = order_gap // 15 + (1 if order_gap % 15 > 0 else 0)  # Assuming max 15 tokens per operation
        chaos_spillover_per_op = 8  # ceil(15/2) worst case
        total_spillover = num_operations * chaos_spillover_per_op
        
        # Estimate chaos venting costs (tokens needed to manage spillover)
        chaos_venting_cost = total_spillover // 2  # Optimistic estimate
        
        return base_tokens_for_order + chaos_venting_cost
    
    def _calculate_maximum_tokens_obtainable(self, domains: Dict[str, Any], initial_tokens: int) -> int:
        """
        Calculate maximum entropy tokens obtainable through energy injection
        """
        max_obtainable = initial_tokens
        
        # Energy injection: can add tokens but creates chaos
        # Theoretical maximum limited by chaos constraints and domain energy
        for domain_name, domain in domains.items():
            domain_energy = domain.get("energy", 0)
            # Conservative estimate: can inject up to domain energy without excessive chaos
            max_injectable = min(domain_energy, 300)  # Cap to prevent excessive calculations
            max_obtainable += max_injectable
        
        return max_obtainable
    
    def _check_chaos_constraints(self, world_state: Dict[str, Any], order_gap: int) -> List[str]:
        """
        Check if chaos constraints make the level unsolvable
        """
        issues = []
        domains = world_state["domains"]
        
        # Check if any domain starts too close to danger threshold
        for domain_name, domain in domains.items():
            initial_chaos = domain.get("chaos", 0)
            if initial_chaos >= self.chaos_danger_threshold:
                issues.append(f"CHAOS CONSTRAINT: Domain {domain_name} starts at chaos {initial_chaos} >= danger threshold {self.chaos_danger_threshold}")
            
            if initial_chaos >= self.chaos_collapse_threshold:
                issues.append(f"CRITICAL: Domain {domain_name} starts at collapse threshold - immediate failure")
        
        # Estimate chaos accumulation during solution
        estimated_operations = max(1, order_gap // 10)  # Rough estimate
        chaos_per_operation = 4  # Average spillover per reverse_entropy
        total_estimated_chaos = estimated_operations * chaos_per_operation
        
        # Check if any domain would exceed limits
        for domain_name, domain in domains.items():
            initial_chaos = domain.get("chaos", 0)
            projected_chaos = initial_chaos + total_estimated_chaos
            if projected_chaos > self.chaos_collapse_threshold:
                issues.append(f"CHAOS PROJECTION: Domain {domain_name} likely to exceed collapse threshold (projected: {projected_chaos})")
        
        return issues
    
    def _estimate_minimum_steps(self, order_gap: int, domains: Dict[str, Any]) -> int:
        """
        Estimate minimum steps needed to solve the level
        """
        if order_gap <= 0:
            return 1
        
        # Optimistic estimate: reverse_entropy with max tokens (15) each step
        max_order_per_step = 15
        min_steps_for_order = (order_gap + max_order_per_step - 1) // max_order_per_step
        
        # Add overhead for chaos management and energy injection
        overhead_steps = max(2, min_steps_for_order // 3)
        
        return min_steps_for_order + overhead_steps
    
    def _check_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Check if action preconditions can be satisfied
        """
        issues = []
        domains = world_state["domains"]
        
        # Check domain names validity
        expected_domains = {"thermal_grid", "data_archive", "crystal_lattice", "bio_habitat"}
        actual_domains = set(domains.keys())
        
        if actual_domains != expected_domains:
            issues.append(f"DOMAIN STRUCTURE: Expected domains {expected_domains}, got {actual_domains}")
        
        # Check if redistribute_order has valid sources
        total_redistributable_order = 0
        for domain in domains.values():
            order = domain.get("order", 0)
            # Domains need to keep some order for stability
            redistributable = max(0, order - 10)
            total_redistributable_order += redistributable
        
        if total_redistributable_order < 20:  # Minimum useful redistribution
            issues.append("ACTION CONSTRAINT: Insufficient order available for redistribution operations")
        
        return issues
    
    def _check_reward_alignment(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Check reward structure for proper incentive alignment
        """
        issues = []
        
        # Based on the environment design, check reward ratios
        # Order Creation: 2 points per unit (PRIMARY GOAL)
        # Stability: 1 point per step (SECONDARY)
        # Efficiency: 0.5 points (BONUS)
        # Chaos Spike: -3 points (PENALTY)
        # Collapse: -25 points (SEVERE PENALTY)
        # Goal Achievement: 50 points (SUCCESS BONUS)
        
        order_gap = self.target_global_order - world_state["globals"]["global_order_score"]
        
        # 1. Check if goal achievement is properly incentivized
        max_order_reward = order_gap * 2  # 2 points per order unit
        goal_bonus = 50
        max_stability_reward = self.max_steps * 1  # 1 point per step if always stable
        
        total_goal_focused_reward = max_order_reward + goal_bonus
        
        # Goal rewards should dominate over just action grinding
        if max_stability_reward >= total_goal_focused_reward * 0.8:
            issues.append("REWARD MISALIGNMENT: Stability rewards too high relative to goal achievement - agents might focus on staying stable rather than solving")
        
        # 2. Check penalty proportions
        collapse_penalty = -25
        max_positive_from_goal = total_goal_focused_reward
        
        # Collapse penalty should be significant but not overwhelmingly harsh
        if abs(collapse_penalty) > max_positive_from_goal:
            issues.append("REWARD BALANCE: Collapse penalty too harsh - might prevent necessary exploration")
        
        # 3. Check for reward loops
        # In this environment, stability rewards could be farmed by doing nothing
        # But this is partially mitigated by the step limit and order requirement
        if order_gap <= 10:
            # If very close to goal, stability farming becomes viable
            issues.append("REWARD LOOP RISK: Low order gap makes stability farming potentially viable")
        
        return issues
    
    def _check_state_validity(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Check basic state validity and structure
        """
        issues = []
        
        # Check required structure
        if "globals" not in world_state:
            issues.append("STRUCTURE: Missing 'globals' section")
            return issues
        
        if "domains" not in world_state:
            issues.append("STRUCTURE: Missing 'domains' section")
            return issues
        
        globals_section = world_state["globals"]
        
        # Check required global fields
        required_global_fields = ["step", "global_order_score", "entropy_tokens"]
        for field in required_global_fields:
            if field not in globals_section:
                issues.append(f"STRUCTURE: Missing required global field '{field}'")
        
        # Check global field ranges
        if "entropy_tokens" in globals_section:
            tokens = globals_section["entropy_tokens"]
            if not isinstance(tokens, int) or tokens < 0:
                issues.append(f"STATE VALIDITY: entropy_tokens must be non-negative integer, got {tokens}")
        
        # Check domain structure
        domains = world_state["domains"]
        required_domain_fields = ["order", "energy", "chaos", "locked"]
        
        for domain_name, domain in domains.items():
            for field in required_domain_fields:
                if field not in domain:
                    issues.append(f"STRUCTURE: Domain {domain_name} missing required field '{field}'")
            
            # Check domain field ranges
            if "order" in domain:
                order = domain["order"]
                if not isinstance(order, int) or order < 0 or order > 100:
                    issues.append(f"STATE VALIDITY: Domain {domain_name} order must be 0-100, got {order}")
            
            if "energy" in domain:
                energy = domain["energy"]
                if not isinstance(energy, int) or energy < 0 or energy > 200:
                    issues.append(f"STATE VALIDITY: Domain {domain_name} energy must be 0-200, got {energy}")
            
            if "chaos" in domain:
                chaos = domain["chaos"]
                if not isinstance(chaos, int) or chaos < 0 or chaos > 100:
                    issues.append(f"STATE VALIDITY: Domain {domain_name} chaos must be 0-100, got {chaos}")
            
            if "locked" in domain:
                locked = domain["locked"]
                if not isinstance(locked, bool):
                    issues.append(f"STATE VALIDITY: Domain {domain_name} locked must be boolean, got {locked}")
        
        return issues
    
    def _check_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Check constraint violations that might make the level too easy/hard
        """
        warnings = []
        
        order_gap = self.target_global_order - world_state["globals"]["global_order_score"]
        
        # Check if level is too easy
        if order_gap <= 10:
            warnings.append("DIFFICULTY: Very low order gap - level might be too easy")
        
        # Check if level is too hard
        if order_gap >= 80:
            warnings.append("DIFFICULTY: Very high order gap - level might be too difficult")
        
        # Check initial chaos distribution
        domains = world_state["domains"]
        chaos_values = [domain.get("chaos", 0) for domain in domains.values()]
        avg_chaos = sum(chaos_values) / len(chaos_values) if chaos_values else 0
        
        if avg_chaos > 50:
            warnings.append("CHAOS DISTRIBUTION: High average initial chaos - might limit solution space")
        
        if avg_chaos < 15:
            warnings.append("CHAOS DISTRIBUTION: Very low initial chaos - might make level too easy")
        
        # Check token economy
        initial_tokens = world_state["globals"]["entropy_tokens"]
        if initial_tokens > 500:
            warnings.append("ECONOMY: Very high initial tokens - might trivialize resource management")
        
        if initial_tokens < 50:
            warnings.append("ECONOMY: Very low initial tokens - might make level too constrained")
        
        return warnings
    
    def _calculate_solvability_score(self, world_state: Dict[str, Any], issues: List[str]) -> float:
        """
        Calculate a solvability score (0.0 to 1.0)
        """
        if any("CRITICAL" in issue or "SOLVABILITY" in issue for issue in issues):
            return 0.0
        
        # Base score
        score = 1.0
        
        # Deduct for each issue type
        for issue in issues:
            if "CHAOS CONSTRAINT" in issue:
                score -= 0.2
            elif "ACTION CONSTRAINT" in issue:
                score -= 0.3
            elif "STRUCTURE" in issue:
                score -= 0.4
            elif "STATE VALIDITY" in issue:
                score -= 0.1
        
        return max(0.0, score)
    
    def _calculate_reward_alignment_score(self, world_state: Dict[str, Any], issues: List[str]) -> float:
        """
        Calculate reward alignment score (0.0 to 1.0)
        """
        if any("REWARD MISALIGNMENT" in issue for issue in issues):
            return 0.0
        
        score = 1.0
        
        for issue in issues:
            if "REWARD" in issue:
                score -= 0.3
            elif "BALANCE" in issue:
                score -= 0.2
            elif "LOOP RISK" in issue:
                score -= 0.1
        
        return max(0.0, score)

def validate_entropy_level(world_file_path: str) -> ValidationResult:
    """
    Convenience function to validate a world file
    """
    with open(world_file_path, 'r') as f:
        world_state = yaml.safe_load(f)
    
    validator = EntropyReversalValidator()
    return validator.validate_level(world_state)

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample world state
    test_world = {
        "globals": {
            "step": 0,
            "global_order_score": 0,
            "entropy_tokens": 200
        },
        "domains": {
            "thermal_grid": {"order": 50, "energy": 100, "chaos": 30, "locked": False},
            "data_archive": {"order": 50, "energy": 100, "chaos": 30, "locked": False},
            "crystal_lattice": {"order": 50, "energy": 100, "chaos": 30, "locked": False},
            "bio_habitat": {"order": 50, "energy": 100, "chaos": 30, "locked": False}
        }
    }
    
    validator = EntropyReversalValidator()
    result = validator.validate_level(test_world)
    
    print(f"Validation Result: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Solvability Score: {result.solvability_score:.2f}")
    print(f"Reward Alignment Score: {result.reward_alignment_score:.2f}")
    
    if result.issues:
        print("\nIssues Found:")
        for issue in result.issues:
            print(f"  - {issue}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")