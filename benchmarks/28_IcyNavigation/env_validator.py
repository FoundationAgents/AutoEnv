import yaml
import os
from collections import deque
from typing import Dict, Any, List, Tuple, Set, Optional
import random

class ReverseLakeValidator:
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def validate_level(self, world_id: str, level_path: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a generated level for solvability and proper reward structure.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Load level data
        if level_path is None:
            level_path = f"./levels/{world_id}.yaml"
            
        if not os.path.exists(level_path):
            issues.append(f"Level file not found: {level_path}")
            return False, issues
            
        try:
            with open(level_path, 'r') as f:
                level_data = yaml.safe_load(f)
        except Exception as e:
            issues.append(f"Failed to load level data: {str(e)}")
            return False, issues
        
        # Run validation checks
        solvability_valid, solvability_issues = self._check_level_solvability(level_data)
        reward_valid, reward_issues = self._validate_reward_structure(level_data)
        
        issues.extend(solvability_issues)
        issues.extend(reward_issues)
        
        is_valid = solvability_valid and reward_valid and len(issues) == 0
        
        return is_valid, issues
    
    def _check_level_solvability(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Critical check for impossible puzzles - validates that the level is actually solvable.
        """
        issues = []
        
        try:
            # Extract key components
            agent_start = tuple(level_data['agent']['pos'])
            goal_pos = tuple(level_data['objects']['goal_flag']['pos'])
            grid_size = level_data['globals']['grid_size']
            max_steps = level_data['globals']['max_steps']
            ice_positions = {tuple(ice['pos']) for ice in level_data['objects']['ice_tiles']}
            
            # 1. ACTION CONSTRAINT ANALYSIS
            # Validate that movement actions can actually modify agent position appropriately
            if not self._validate_action_constraints(grid_size, ice_positions):
                issues.append("Action constraints fail: insufficient safe movement options")
            
            # 2. TARGET REACHABILITY
            # Check if goal is reachable from start position
            is_reachable, min_steps = self._check_reachability(agent_start, goal_pos, grid_size, ice_positions)
            
            if not is_reachable:
                issues.append("CRITICAL: Goal is not reachable from starting position")
            elif min_steps > max_steps:
                issues.append(f"CRITICAL: Minimum steps to goal ({min_steps}) exceeds max_steps limit ({max_steps})")
            
            # 3. RESOURCE AVAILABILITY CHECK
            # Ensure sufficient safe tiles exist for navigation
            total_tiles = grid_size[0] * grid_size[1]
            safe_tiles = total_tiles - len(ice_positions)
            min_required_safe = 2  # At least start and goal positions
            
            if safe_tiles < min_required_safe:
                issues.append(f"Insufficient safe tiles: {safe_tiles} available, minimum {min_required_safe} required")
            
            # 4. COMMON IMPOSSIBLE PATTERNS CHECK
            impossible_patterns = self._detect_impossible_patterns(level_data)
            issues.extend(impossible_patterns)
            
            # 5. STEP BUDGET VALIDATION
            if max_steps < 1:
                issues.append("Invalid step budget: max_steps must be positive")
            elif max_steps < min_steps and is_reachable:
                issues.append(f"Step budget too restrictive: needs {min_steps}, only has {max_steps}")
            
        except Exception as e:
            issues.append(f"Solvability analysis failed: {str(e)}")
        
        return len(issues) == 0, issues
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Critical check for incentive alignment - ensures rewards prioritize problem-solving.
        """
        issues = []
        
        try:
            reward_config = self.config.get('reward', {})
            goal_values = reward_config.get('goal_values', {})
            
            success_reward = goal_values.get('success', 0)
            failure_reward = goal_values.get('failure', 0)
            timeout_reward = goal_values.get('timeout', 0)
            
            # 1. GOAL-ORIENTED REWARDS CHECK
            if success_reward <= 0:
                issues.append("CRITICAL: Success reward must be positive to incentivize goal achievement")
            
            if success_reward <= max(failure_reward, timeout_reward):
                issues.append("CRITICAL: Success reward must be higher than failure/timeout rewards")
            
            # 2. AVOID INCENTIVE MISALIGNMENT
            # Check that failure states don't provide positive rewards (prevents action grinding)
            if failure_reward > 0:
                issues.append("Reward misalignment: Failure should not provide positive reward")
            
            if timeout_reward > 0:
                issues.append("Reward misalignment: Timeout should not provide positive reward")
            
            # 3. REWARD DESIGN PRINCIPLES
            # Ensure binary reward structure is maintained (sparse rewards)
            expected_rewards = [success_reward, failure_reward, timeout_reward]
            if any(r < 0 for r in expected_rewards):
                issues.append("Negative rewards detected - may cause exploration issues")
            
            # 4. EFFICIENCY INCENTIVE CHECK
            # With binary rewards, efficiency is naturally incentivized by episode termination
            # No additional checks needed for this simple reward structure
            
        except Exception as e:
            issues.append(f"Reward structure validation failed: {str(e)}")
        
        return len(issues) == 0, issues
    
    def _validate_action_constraints(self, grid_size: List[int], ice_positions: Set[Tuple[int, int]]) -> bool:
        """
        Analyze if movement actions have sufficient power to enable navigation.
        """
        # Check if there are enough safe positions to allow meaningful movement
        total_positions = grid_size[0] * grid_size[1]
        safe_positions = total_positions - len(ice_positions)
        
        # Need at least 25% safe positions for reasonable navigation
        min_safe_ratio = 0.25
        return (safe_positions / total_positions) >= min_safe_ratio
    
    def _check_reachability(self, start: Tuple[int, int], goal: Tuple[int, int], 
                           grid_size: List[int], ice_positions: Set[Tuple[int, int]]) -> Tuple[bool, int]:
        """
        BFS to verify goal reachability and calculate minimum steps required.
        """
        if start == goal:
            return True, 0
        
        queue = deque([(start, 0)])
        visited = {start}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            (x, y), steps = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or nx >= grid_size[0] or ny < 0 or ny >= grid_size[1]:
                    continue
                
                # Skip visited positions
                if (nx, ny) in visited:
                    continue
                
                # Skip ice tiles (dangerous)
                if (nx, ny) in ice_positions:
                    continue
                
                # Check if reached goal
                if (nx, ny) == goal:
                    return True, steps + 1
                
                visited.add((nx, ny))
                queue.append(((nx, ny), steps + 1))
        
        return False, -1
    
    def _detect_impossible_patterns(self, level_data: Dict[str, Any]) -> List[str]:
        """
        Detect common patterns that make levels impossible to solve.
        """
        issues = []
        
        agent_start = tuple(level_data['agent']['pos'])
        goal_pos = tuple(level_data['objects']['goal_flag']['pos'])
        grid_size = level_data['globals']['grid_size']
        ice_positions = {tuple(ice['pos']) for ice in level_data['objects']['ice_tiles']}
        
        # Pattern 1: Goal or start position on ice (violates environment invariants)
        if agent_start in ice_positions:
            issues.append("Impossible pattern: Agent starts on ice tile")
        
        if goal_pos in ice_positions:
            issues.append("Impossible pattern: Goal placed on ice tile")
        
        # Pattern 2: Goal completely surrounded by ice or boundaries
        if self._is_position_trapped(goal_pos, grid_size, ice_positions):
            issues.append("Impossible pattern: Goal is completely surrounded by ice/boundaries")
        
        # Pattern 3: Start position completely surrounded
        if self._is_position_trapped(agent_start, grid_size, ice_positions):
            issues.append("Impossible pattern: Start position is completely surrounded")
        
        # Pattern 4: Excessive ice density making navigation impossible
        total_tiles = grid_size[0] * grid_size[1]
        ice_ratio = len(ice_positions) / total_tiles
        if ice_ratio > 0.8:  # More than 80% ice
            issues.append("Impossible pattern: Excessive ice density (>80%) makes navigation nearly impossible")
        
        return issues
    
    def _is_position_trapped(self, pos: Tuple[int, int], grid_size: List[int], 
                            ice_positions: Set[Tuple[int, int]]) -> bool:
        """
        Check if a position is completely surrounded by ice or boundaries.
        """
        x, y = pos
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if this adjacent position is accessible
            if (0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and 
                (nx, ny) not in ice_positions):
                return False  # Found at least one safe adjacent position
        
        return True  # All adjacent positions are blocked
    
    def validate_batch(self, level_directory: str = "./levels/") -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all levels in a directory.
        
        Returns:
            Dict mapping world_id to (is_valid, issues_list)
        """
        results = {}
        
        if not os.path.exists(level_directory):
            return results
        
        for filename in os.listdir(level_directory):
            if filename.endswith('.yaml'):
                world_id = filename[:-5]  # Remove .yaml extension
                level_path = os.path.join(level_directory, filename)
                
                try:
                    is_valid, issues = self.validate_level(world_id, level_path)
                    results[world_id] = (is_valid, issues)
                except Exception as e:
                    results[world_id] = (False, [f"Validation error: {str(e)}"])
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Tuple[bool, List[str]]]) -> str:
        """
        Generate a human-readable validation report.
        """
        report = "=== REVERSE LAKE NAVIGATION LEVEL VALIDATION REPORT ===\n\n"
        
        total_levels = len(results)
        valid_levels = sum(1 for valid, _ in results.values() if valid)
        
        report += f"Total Levels: {total_levels}\n"
        report += f"Valid Levels: {valid_levels}\n"
        report += f"Invalid Levels: {total_levels - valid_levels}\n"
        report += f"Success Rate: {valid_levels/total_levels*100:.1f}%\n\n"
        
        if total_levels > 0:
            report += "=== DETAILED RESULTS ===\n"
            
            for world_id, (is_valid, issues) in results.items():
                status = "✓ VALID" if is_valid else "✗ INVALID"
                report += f"\n{world_id}: {status}\n"
                
                if issues:
                    for issue in issues:
                        report += f"  - {issue}\n"
        
        return report

# Usage example and testing functions
def validate_single_level(world_id: str) -> None:
    """Helper function to validate a single level and print results."""
    validator = ReverseLakeValidator()
    is_valid, issues = validator.validate_level(world_id)
    
    print(f"Level {world_id}: {'VALID' if is_valid else 'INVALID'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")

def validate_all_levels() -> None:
    """Helper function to validate all levels and generate a report."""
    validator = ReverseLakeValidator()
    results = validator.validate_batch()
    report = validator.generate_validation_report(results)
    print(report)