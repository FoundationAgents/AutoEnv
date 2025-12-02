import yaml
import os
import random
from typing import Dict, Any, List, Tuple, Optional, Set
from itertools import combinations_with_replacement
import math

class GearRatioValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tolerance = config.get("state_template", {}).get("globals", {}).get("tolerance", 0.02)
        self.max_steps = config.get("state_template", {}).get("globals", {}).get("max_steps", 30)
        self.max_chain_length = 10  # Reasonable limit for gear chain length
        
    def validate_level(self, world_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except FileNotFoundError:
            return False, [f"World file not found: {world_path}"]
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML format: {e}"]
        
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. STATE CONSISTENCY CHECKS
        consistency_issues = self._check_state_consistency(world_state)
        issues.extend(consistency_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        try:
            gear_system = world_state["gear_system"]
            available_gears = gear_system["available_gears"]
            target_ma = gear_system["target_ma"]
        except KeyError as e:
            return [f"Missing required field in gear_system: {e}"]
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(available_gears)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(available_gears, target_ma)
        issues.extend(reachability_issues)
        
        # RESOURCE AVAILABILITY
        resource_issues = self._check_resource_availability(available_gears, target_ma)
        issues.extend(resource_issues)
        
        # STEP BUDGET FEASIBILITY
        step_issues = self._check_step_feasibility(available_gears, target_ma)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, available_gears: List[int]) -> List[str]:
        """Understand fundamental limitations of available actions"""
        issues = []
        
        # Check if gear library is valid
        if not available_gears or len(available_gears) == 0:
            issues.append("Empty gear library - no gears available for assembly")
        
        # Check gear tooth count validity
        for i, gear in enumerate(available_gears):
            if not isinstance(gear, int) or gear < 6 or gear > 60:
                issues.append(f"Invalid gear tooth count at index {i}: {gear} (must be 6-60)")
        
        # Check if gears provide sufficient diversity for meaningful ratios
        if len(set(available_gears)) == 1:
            issues.append("All gears have identical tooth counts - no meaningful ratios possible")
        
        return issues
    
    def _check_target_reachability(self, available_gears: List[int], target_ma: float) -> List[str]:
        """Verify target state is actually achievable through available actions"""
        issues = []
        
        if target_ma <= 0:
            issues.append(f"Invalid target mechanical advantage: {target_ma} (must be positive)")
            return issues
        
        # Use efficient breadth-first search with pruning
        is_reachable, closest_ma = self._find_closest_achievable_ma(available_gears, target_ma)
        
        if not is_reachable:
            error_ratio = abs(closest_ma - target_ma) / target_ma
            issues.append(
                f"Target MA {target_ma:.4f} is not achievable within tolerance {self.tolerance:.1%}. "
                f"Closest achievable MA: {closest_ma:.4f} (error: {error_ratio:.1%})"
            )
        
        return issues
    
    def _check_resource_availability(self, available_gears: List[int], target_ma: float) -> List[str]:
        """Check if required resources are available"""
        issues = []
        
        # Calculate theoretical range of achievable MAs
        min_possible_ma, max_possible_ma = self._calculate_ma_bounds(available_gears)
        
        if target_ma < min_possible_ma * (1 - self.tolerance):
            issues.append(
                f"Target MA {target_ma:.4f} is below minimum achievable range "
                f"[{min_possible_ma * (1 - self.tolerance):.4f}, {max_possible_ma * (1 + self.tolerance):.4f}]"
            )
        
        if target_ma > max_possible_ma * (1 + self.tolerance):
            issues.append(
                f"Target MA {target_ma:.4f} is above maximum achievable range "
                f"[{min_possible_ma * (1 - self.tolerance):.4f}, {max_possible_ma * (1 + self.tolerance):.4f}]"
            )
        
        return issues
    
    def _check_step_feasibility(self, available_gears: List[int], target_ma: float) -> List[str]:
        """Check if target is reachable within step limits"""
        issues = []
        
        # Find minimum steps required to achieve target
        min_steps = self._find_minimum_solution_steps(available_gears, target_ma)
        
        if min_steps > self.max_steps:
            issues.append(
                f"Minimum solution requires {min_steps} steps, but limit is {self.max_steps}"
            )
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate reward structure design"""
        issues = []
        
        # Check that success is only achieved through proper goal completion
        gear_system = world_state.get("gear_system", {})
        
        # Verify initial state doesn't accidentally satisfy win condition
        current_ma = gear_system.get("current_ma", 1.0)
        target_ma = gear_system.get("target_ma", 1.0)
        
        if current_ma != 1.0:
            issues.append("Initial mechanical advantage should be 1.0 (empty chain)")
        
        initial_error = abs(current_ma - target_ma) / target_ma if target_ma != 0 else 0
        if initial_error <= self.tolerance:
            issues.append(
                f"Target MA {target_ma:.4f} is too close to initial MA {current_ma:.4f} - "
                f"agent can win immediately with Finish action"
            )
        
        # Verify that intermediate states don't provide exploitable rewards
        # (This is handled by the binary reward structure in the environment)
        
        return issues
    
    def _check_state_consistency(self, world_state: Dict[str, Any]) -> List[str]:
        """Check for state consistency and proper initialization"""
        issues = []
        
        # Check required top-level keys
        required_keys = ["globals", "agent", "gear_system"]
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required top-level key: {key}")
        
        # Check globals
        globals_section = world_state.get("globals", {})
        if globals_section.get("max_steps") != self.max_steps:
            issues.append(f"Inconsistent max_steps: expected {self.max_steps}, got {globals_section.get('max_steps')}")
        
        if globals_section.get("tolerance") != self.tolerance:
            issues.append(f"Inconsistent tolerance: expected {self.tolerance}, got {globals_section.get('tolerance')}")
        
        # Check agent state
        agent_section = world_state.get("agent", {})
        if agent_section.get("remaining_steps") != self.max_steps:
            issues.append("Initial remaining_steps should equal max_steps")
        
        # Check gear_system initialization
        gear_system = world_state.get("gear_system", {})
        if gear_system.get("gear_chain", []) != []:
            issues.append("Initial gear_chain should be empty")
        
        if gear_system.get("episode_finished", False) != False:
            issues.append("Initial episode_finished should be False")
        
        if gear_system.get("success", False) != False:
            issues.append("Initial success should be False")
        
        return issues
    
    def _calculate_mechanical_advantage(self, gear_chain: List[int]) -> float:
        """Calculate mechanical advantage for a gear chain"""
        if len(gear_chain) == 0:
            return 1.0
        
        ma = 1.0
        for i in range(0, len(gear_chain) - 1, 2):
            if i + 1 < len(gear_chain):
                ma *= gear_chain[i] / gear_chain[i + 1]
        
        return ma
    
    def _find_closest_achievable_ma(self, available_gears: List[int], target_ma: float) -> Tuple[bool, float]:
        """Find closest achievable MA to target using BFS with pruning"""
        visited_mas = set()
        closest_ma = 1.0
        min_error = abs(target_ma - 1.0) / target_ma
        
        # BFS queue: (gear_chain, current_ma)
        queue = [([], 1.0)]
        
        for depth in range(self.max_chain_length):
            if not queue:
                break
                
            next_queue = []
            
            for chain, current_ma in queue:
                # Check if current MA is within tolerance
                error_ratio = abs(current_ma - target_ma) / target_ma
                if error_ratio <= self.tolerance:
                    return True, current_ma
                
                # Update closest MA
                if error_ratio < min_error:
                    min_error = error_ratio
                    closest_ma = current_ma
                
                # Prune if we've seen this MA before (with some tolerance for floating point)
                ma_key = round(current_ma, 8)
                if ma_key in visited_mas:
                    continue
                visited_mas.add(ma_key)
                
                # Generate next states
                if len(chain) < self.max_chain_length:
                    for gear in available_gears:
                        new_chain = chain + [gear]
                        new_ma = self._calculate_mechanical_advantage(new_chain)
                        
                        # Prune obviously bad branches
                        if new_ma > 0 and not math.isinf(new_ma):
                            next_queue.append((new_chain, new_ma))
            
            queue = next_queue
            
            # Limit queue size to prevent explosion
            if len(queue) > 1000:
                queue = sorted(queue, key=lambda x: abs(x[1] - target_ma))[:1000]
        
        return False, closest_ma
    
    def _calculate_ma_bounds(self, available_gears: List[int]) -> Tuple[float, float]:
        """Calculate theoretical minimum and maximum achievable MAs"""
        if not available_gears:
            return 1.0, 1.0
        
        min_gear = min(available_gears)
        max_gear = max(available_gears)
        
        # For maximum MA: use largest gears as drivers, smallest as driven
        # For minimum MA: use smallest gears as drivers, largest as driven
        
        # With chain length up to max_chain_length, theoretical bounds are:
        ratio_pairs = min(self.max_chain_length // 2, 5)  # Reasonable limit
        
        max_single_ratio = max_gear / min_gear
        min_single_ratio = min_gear / max_gear
        
        theoretical_max = max_single_ratio ** ratio_pairs
        theoretical_min = min_single_ratio ** ratio_pairs
        
        return theoretical_min, theoretical_max
    
    def _find_minimum_solution_steps(self, available_gears: List[int], target_ma: float) -> int:
        """Find minimum steps required to achieve target MA"""
        # BFS to find shortest solution
        queue = [([], 1.0, 0)]  # (chain, ma, steps)
        visited = set()
        
        for _ in range(1000):  # Limit iterations
            if not queue:
                break
                
            chain, current_ma, steps = queue.pop(0)
            
            # Check if target achieved
            error_ratio = abs(current_ma - target_ma) / target_ma
            if error_ratio <= self.tolerance:
                return steps + 1  # +1 for Finish action
            
            # Avoid revisiting similar states
            ma_key = round(current_ma, 6)
            if ma_key in visited:
                continue
            visited.add(ma_key)
            
            # Add gear placement actions
            if len(chain) < self.max_chain_length and steps < self.max_steps - 1:
                for gear in available_gears:
                    new_chain = chain + [gear]
                    new_ma = self._calculate_mechanical_advantage(new_chain)
                    if new_ma > 0 and not math.isinf(new_ma):
                        queue.append((new_chain, new_ma, steps + 1))
        
        return self.max_steps + 1  # Indicate no solution found within limit
    
    def validate_batch(self, levels_dir: str) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all levels in a directory"""
        results = {}
        
        if not os.path.exists(levels_dir):
            return {"ERROR": (False, [f"Directory not found: {levels_dir}"])}
        
        for filename in os.listdir(levels_dir):
            if filename.endswith('.yaml'):
                level_path = os.path.join(levels_dir, filename)
                level_id = filename[:-5]  # Remove .yaml extension
                results[level_id] = self.validate_level(level_path)
        
        return results
    
    def generate_validation_report(self, validation_results: Dict[str, Tuple[bool, List[str]]]) -> str:
        """Generate a human-readable validation report"""
        total_levels = len(validation_results)
        valid_levels = sum(1 for is_valid, _ in validation_results.values() if is_valid)
        
        report = f"GEAR RATIO OPTIMIZATION - LEVEL VALIDATION REPORT\n"
        report += f"{'='*60}\n\n"
        report += f"Total Levels: {total_levels}\n"
        report += f"Valid Levels: {valid_levels}\n"
        report += f"Invalid Levels: {total_levels - valid_levels}\n"
        report += f"Success Rate: {valid_levels/total_levels*100:.1f}%\n\n"
        
        if valid_levels < total_levels:
            report += "DETAILED ISSUES:\n"
            report += "-" * 40 + "\n"
            
            for level_id, (is_valid, issues) in validation_results.items():
                if not is_valid:
                    report += f"\nLevel: {level_id}\n"
                    for issue in issues:
                        report += f"  ❌ {issue}\n"
        
        return report