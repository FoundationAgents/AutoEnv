import yaml
import os
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
from collections import deque
import random

class SquadReconLevelValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensor_radius = 3
        
    def validate_level(self, world_path: str) -> Tuple[bool, List[str]]:
        """Validate a generated level for solvability and reward structure."""
        with open(world_path, 'r') as f:
            world_state = yaml.load(f, Loader=yaml.FullLoader)
        
        issues = []
        
        # 1. Basic structure validation
        basic_issues = self._validate_basic_structure(world_state)
        issues.extend(basic_issues)
        
        # 2. Level solvability analysis
        solvability_issues = self._validate_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 4. Terrain and placement validation
        terrain_issues = self._validate_terrain_placement(world_state)
        issues.extend(terrain_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_basic_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate basic level structure requirements."""
        issues = []
        
        # Check required fields exist
        required_fields = ["globals", "squads", "enemy_camps", "terrain", "visibility_map"]
        for field in required_fields:
            if field not in state:
                issues.append(f"Missing required field: {field}")
        
        if "globals" in state:
            if state["globals"]["total_enemy_camps"] != 2:
                issues.append("Must have exactly 2 enemy camps")
            
            if state["globals"]["max_steps"] != 40:
                issues.append("Max steps must be 40")
        
        if "squads" in state:
            active_squads = [s for s in state["squads"] if s["active"]]
            if len(active_squads) != 3:
                issues.append("Must have exactly 3 active squads")
        
        return issues
    
    def _validate_solvability(self, state: Dict[str, Any]) -> List[str]:
        """Critical solvability analysis to prevent impossible levels."""
        issues = []
        
        # 1. ACTION CONSTRAINT ANALYSIS
        constraint_issues = self._analyze_action_constraints(state)
        issues.extend(constraint_issues)
        
        # 2. TARGET REACHABILITY
        reachability_issues = self._analyze_target_reachability(state)
        issues.extend(reachability_issues)
        
        # 3. RESOURCE AVAILABILITY
        resource_issues = self._analyze_resource_availability(state)
        issues.extend(resource_issues)
        
        # 4. STEP BUDGET ANALYSIS
        step_issues = self._analyze_step_budget(state)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, state: Dict[str, Any]) -> List[str]:
        """Analyze fundamental action limitations."""
        issues = []
        
        grid_size = state["globals"]["grid_size"]
        walls = state["terrain"]["walls"]
        forests = state["terrain"]["forests"]
        
        # Check if terrain blocks create impossible navigation
        blocked_positions = set(tuple(pos) for pos in walls + forests)
        
        # Verify all positions are theoretically reachable from spawn area
        spawn_positions = [(x, y) for x in range(4) for y in range(4) 
                          if (x, y) not in blocked_positions]
        
        if not spawn_positions:
            issues.append("CRITICAL: No valid spawn positions available")
            return issues
        
        # Check connectivity using flood fill from spawn area
        reachable = self._flood_fill_reachable(spawn_positions[0], grid_size, blocked_positions)
        total_open_cells = grid_size[0] * grid_size[1] - len(blocked_positions)
        
        if len(reachable) < total_open_cells * 0.8:  # At least 80% should be reachable
            issues.append("CRITICAL: Large portions of map unreachable from spawn area")
        
        return issues
    
    def _analyze_target_reachability(self, state: Dict[str, Any]) -> List[str]:
        """Verify enemy camps are reachable and discoverable."""
        issues = []
        
        grid_size = state["globals"]["grid_size"]
        walls = state["terrain"]["walls"]
        forests = state["terrain"]["forests"]
        blocked_positions = set(tuple(pos) for pos in walls + forests)
        
        squad_start_positions = [tuple(squad["pos"]) for squad in state["squads"] if squad["active"]]
        
        for camp in state["enemy_camps"]:
            camp_pos = tuple(camp["pos"])
            
            # 1. Check if camp position is valid
            if camp_pos in blocked_positions:
                issues.append(f"CRITICAL: Enemy camp at {camp_pos} is placed on impassable terrain")
                continue
            
            # 2. Check if camp is reachable from any squad starting position
            reachable_from_any_squad = False
            for squad_pos in squad_start_positions:
                if self._is_path_exists(squad_pos, camp_pos, grid_size, blocked_positions):
                    reachable_from_any_squad = True
                    break
            
            if not reachable_from_any_squad:
                issues.append(f"CRITICAL: Enemy camp at {camp_pos} is unreachable from all squad starting positions")
            
            # 3. Check if camp can be discovered (adjacent position reachable for attack)
            adjacent_positions = self._get_adjacent_positions(camp_pos, grid_size)
            valid_attack_positions = [pos for pos in adjacent_positions 
                                   if pos not in blocked_positions]
            
            if not valid_attack_positions:
                issues.append(f"CRITICAL: Enemy camp at {camp_pos} has no valid adjacent attack positions")
            else:
                # Check if at least one attack position is reachable
                attack_position_reachable = False
                for attack_pos in valid_attack_positions:
                    for squad_pos in squad_start_positions:
                        if self._is_path_exists(squad_pos, attack_pos, grid_size, blocked_positions):
                            attack_position_reachable = True
                            break
                    if attack_position_reachable:
                        break
                
                if not attack_position_reachable:
                    issues.append(f"CRITICAL: No valid attack positions reachable for enemy camp at {camp_pos}")
        
        return issues
    
    def _analyze_resource_availability(self, state: Dict[str, Any]) -> List[str]:
        """Analyze if squads have sufficient combat strength to defeat camps."""
        issues = []
        
        total_friendly_strength = sum(squad["strength"] for squad in state["squads"] if squad["active"])
        enemy_camps = state["enemy_camps"]
        
        # 1. Check if total friendly strength can defeat all camps
        min_required_strength = sum(camp["strength"] + 1 for camp in enemy_camps)  # +1 because attack must exceed defense
        
        if total_friendly_strength < min_required_strength:
            issues.append(f"CRITICAL: Total friendly strength ({total_friendly_strength}) insufficient to defeat all enemy camps (need at least {min_required_strength})")
        
        # 2. Check if individual camps can be defeated
        max_single_camp_strength = max(camp["strength"] for camp in enemy_camps)
        if total_friendly_strength <= max_single_camp_strength:
            issues.append(f"CRITICAL: Strongest enemy camp ({max_single_camp_strength}) cannot be defeated by all squads combined ({total_friendly_strength})")
        
        # 3. Check for reasonable strength distribution
        weakest_camp = min(camp["strength"] for camp in enemy_camps)
        strongest_squad = max(squad["strength"] for squad in state["squads"] if squad["active"])
        
        if strongest_squad * 3 < weakest_camp:  # Even with all squads, might be too weak
            issues.append(f"WARNING: Very high enemy strength relative to squad strength may make level extremely difficult")
        
        # 4. Validate strength ranges
        for camp in enemy_camps:
            if camp["strength"] < 2 or camp["strength"] > 6:
                issues.append(f"Invalid enemy camp strength: {camp['strength']} (must be 2-6)")
        
        for squad in state["squads"]:
            if squad["strength"] < 1 or squad["strength"] > 4:
                issues.append(f"Invalid squad strength: {squad['strength']} (must be 1-4)")
        
        return issues
    
    def _analyze_step_budget(self, state: Dict[str, Any]) -> List[str]:
        """Analyze if level is solvable within step limit."""
        issues = []
        
        max_steps = state["globals"]["max_steps"]
        grid_size = state["globals"]["grid_size"]
        
        # Estimate minimum steps required
        squad_positions = [squad["pos"] for squad in state["squads"] if squad["active"]]
        camp_positions = [camp["pos"] for camp in state["enemy_camps"]]
        
        # Calculate exploration requirements
        total_cells = grid_size[0] * grid_size[1]
        walls = len(state["terrain"]["walls"])
        forests = len(state["terrain"]["forests"])
        explorable_cells = total_cells - walls - forests
        
        # Rough estimate: need to explore enough area to find both camps
        # With 3 squads and sensor radius 3, each step can explore ~21 new cells maximum
        sensor_coverage_per_step = 3 * (2 * self.sensor_radius + 1) ** 2  # 3 squads * 7x7
        min_exploration_steps = max(10, explorable_cells // (sensor_coverage_per_step // 2))  # Conservative estimate
        
        # Add movement and combat steps
        max_distance_to_camps = max(
            min(self._manhattan_distance(squad_pos, camp_pos) 
                for squad_pos in squad_positions)
            for camp_pos in camp_positions
        )
        
        estimated_min_steps = min_exploration_steps + max_distance_to_camps + 5  # +5 for coordination
        
        if estimated_min_steps > max_steps * 0.9:  # Should be solvable with 10% buffer
            issues.append(f"WARNING: Level may be too tight on step budget. Estimated minimum steps: {estimated_min_steps}, Available: {max_steps}")
        
        return issues
    
    def _validate_reward_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate reward structure design for proper incentive alignment."""
        issues = []
        
        # Check that reward values are appropriate
        expected_reward_per_camp = 0.5
        expected_total_reward = 1.0
        
        # Verify reward structure matches configuration
        if "reward" in self.config:
            camp_rewards = self.config["reward"].get("camp_rewards", {})
            first_camp_reward = camp_rewards.get("first_camp", 0)
            second_camp_reward = camp_rewards.get("second_camp", 0)
            
            if first_camp_reward != expected_reward_per_camp:
                issues.append(f"First camp reward should be {expected_reward_per_camp}, got {first_camp_reward}")
            
            if second_camp_reward != expected_reward_per_camp:
                issues.append(f"Second camp reward should be {expected_reward_per_camp}, got {second_camp_reward}")
        
        # Check for reward design principles
        total_enemy_camps = state["globals"]["total_enemy_camps"]
        if total_enemy_camps != 2:
            issues.append(f"Reward structure designed for 2 camps, but level has {total_enemy_camps}")
        
        # Validate no action grinding opportunities exist
        # (This is structural - the environment only rewards camp elimination)
        
        return issues
    
    def _validate_terrain_placement(self, state: Dict[str, Any]) -> List[str]:
        """Validate terrain placement doesn't create impossible scenarios."""
        issues = []
        
        grid_size = state["globals"]["grid_size"]
        walls = state["terrain"]["walls"]
        forests = state["terrain"]["forests"]
        
        # Check terrain density
        total_cells = grid_size[0] * grid_size[1]
        blocked_cells = len(walls) + len(forests)
        density = blocked_cells / total_cells
        
        if density > 0.4:
            issues.append(f"Terrain density too high: {density:.2f} (recommend < 0.4)")
        
        # Check spawn area is clear
        spawn_area = [(x, y) for x in range(4) for y in range(4)]
        blocked_positions = set(tuple(pos) for pos in walls + forests)
        
        spawn_blocked = [pos for pos in spawn_area if pos in blocked_positions]
        if spawn_blocked:
            issues.append(f"Spawn area has blocked positions: {spawn_blocked}")
        
        # Check for completely enclosed areas
        if self._has_isolated_areas(grid_size, blocked_positions):
            issues.append("Map has isolated areas that cannot be reached")
        
        return issues
    
    def _flood_fill_reachable(self, start: Tuple[int, int], grid_size: List[int], 
                            blocked: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Find all positions reachable from start position."""
        reachable = set()
        queue = deque([start])
        
        while queue:
            pos = queue.popleft()
            if pos in reachable or pos in blocked:
                continue
            
            x, y = pos
            if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
                continue
            
            reachable.add(pos)
            
            # Add neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in reachable:
                    queue.append(neighbor)
        
        return reachable
    
    def _is_path_exists(self, start: Tuple[int, int], end: Tuple[int, int], 
                      grid_size: List[int], blocked: Set[Tuple[int, int]]) -> bool:
        """Check if path exists between two positions using BFS."""
        if start == end:
            return True
        
        visited = set()
        queue = deque([start])
        
        while queue:
            pos = queue.popleft()
            if pos == end:
                return True
            
            if pos in visited or pos in blocked:
                continue
            
            x, y = pos
            if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
                continue
            
            visited.add(pos)
            
            # Add neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return False
    
    def _get_adjacent_positions(self, pos: Tuple[int, int], grid_size: List[int]) -> List[Tuple[int, int]]:
        """Get valid adjacent positions for a given position."""
        x, y = pos
        adjacent = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                adjacent.append((nx, ny))
        
        return adjacent
    
    def _manhattan_distance(self, pos1: List[int], pos2: List[int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _has_isolated_areas(self, grid_size: List[int], blocked: Set[Tuple[int, int]]) -> bool:
        """Check if there are isolated areas in the map."""
        # Find all open positions
        all_positions = set((x, y) for x in range(grid_size[0]) for y in range(grid_size[1]))
        open_positions = all_positions - blocked
        
        if not open_positions:
            return True
        
        # Start flood fill from any open position
        start_pos = next(iter(open_positions))
        reachable = self._flood_fill_reachable(start_pos, grid_size, blocked)
        
        # If not all open positions are reachable, there are isolated areas
        return len(reachable) < len(open_positions)

def validate_squad_recon_level(world_path: str, config_path: str = "./config.yaml") -> Tuple[bool, List[str]]:
    """Convenience function to validate a single level."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    validator = SquadReconLevelValidator(config)
    return validator.validate_level(world_path)

def validate_all_levels(levels_dir: str = "./levels/", config_path: str = "./config.yaml") -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all levels in a directory."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    validator = SquadReconLevelValidator(config)
    results = {}
    
    for filename in os.listdir(levels_dir):
        if filename.endswith(".yaml"):
            world_path = os.path.join(levels_dir, filename)
            world_id = filename[:-5]  # Remove .yaml extension
            is_valid, issues = validator.validate_level(world_path)
            results[world_id] = (is_valid, issues)
    
    return results

if __name__ == "__main__":
    # Example usage
    results = validate_all_levels()
    
    for world_id, (is_valid, issues) in results.items():
        print(f"\n{world_id}: {'VALID' if is_valid else 'INVALID'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")