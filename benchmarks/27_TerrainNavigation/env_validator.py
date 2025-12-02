import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import random

class IceLakeValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_template = config.get("state_template", {})
        self.max_steps = self.state_template.get("globals", {}).get("max_steps", 40)
        
    def validate_level(self, world_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated ice lake level for solvability and reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load world file: {str(e)}"]
        
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
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(world_state)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(world_state)
        issues.extend(reachability_issues)
        
        # STEP BUDGET VALIDATION
        step_issues = self._validate_step_budget(world_state)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Understand environment's fundamental limitations"""
        issues = []
        
        # Check if basic movement actions can modify agent position
        grid_size = world_state.get("tiles", {}).get("size", [8, 8])
        if not isinstance(grid_size, list) or len(grid_size) != 2:
            issues.append("Invalid grid size format")
            return issues
        
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            issues.append("Grid size must be positive")
        
        # Verify layout exists and has correct dimensions
        layout = world_state.get("tiles", {}).get("layout", [])
        if len(layout) != grid_size[0]:
            issues.append(f"Layout height {len(layout)} doesn't match grid size {grid_size[0]}")
        
        for i, row in enumerate(layout):
            if len(row) != grid_size[1]:
                issues.append(f"Layout row {i} width {len(row)} doesn't match grid size {grid_size[1]}")
        
        return issues
    
    def _check_target_reachability(self, world_state: Dict[str, Any]) -> List[str]:
        """Verify target state is actually achievable"""
        issues = []
        
        start_pos = world_state.get("start_pos", [4, 0])
        goal_pos = world_state.get("goal_pos", [4, 7])
        layout = world_state.get("tiles", {}).get("layout", [])
        grid_size = world_state.get("tiles", {}).get("size", [8, 8])
        
        # Validate positions are within bounds
        if not self._is_valid_position(start_pos, grid_size):
            issues.append(f"Start position {start_pos} is out of bounds for grid {grid_size}")
        
        if not self._is_valid_position(goal_pos, grid_size):
            issues.append(f"Goal position {goal_pos} is out of bounds for grid {grid_size}")
        
        if len(issues) > 0:
            return issues
        
        # Check start and goal positions are safe
        if layout[start_pos[0]][start_pos[1]] == "water":
            issues.append("Start position is on water - impossible to begin")
        
        if layout[goal_pos[0]][goal_pos[1]] == "water":
            issues.append("Goal position is on water - impossible to reach")
        
        # Path existence check using BFS
        if not self._has_valid_path(start_pos, goal_pos, layout, grid_size):
            issues.append("No valid path exists from start to goal - level is unsolvable")
        
        return issues
    
    def _validate_step_budget(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if solution is achievable within step limits"""
        issues = []
        
        start_pos = world_state.get("start_pos", [4, 0])
        goal_pos = world_state.get("goal_pos", [4, 7])
        layout = world_state.get("tiles", {}).get("layout", [])
        grid_size = world_state.get("tiles", {}).get("size", [8, 8])
        
        # Find shortest path length
        shortest_path_length = self._find_shortest_path_length(start_pos, goal_pos, layout, grid_size)
        
        if shortest_path_length is None:
            issues.append("Cannot find any path from start to goal")
        elif shortest_path_length > self.max_steps:
            issues.append(f"Shortest path requires {shortest_path_length} steps but only {self.max_steps} steps available")
        elif shortest_path_length == self.max_steps:
            issues.append("Shortest path uses all available steps - no room for exploration (too tight)")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Check that only goal achievement provides positive reward
        # The environment should follow binary reward: +1 for success, 0 otherwise
        
        # Verify goal achievement is the primary reward source
        start_pos = world_state.get("start_pos", [4, 0])
        goal_pos = world_state.get("goal_pos", [4, 7])
        
        # Check for potential reward farming opportunities
        if start_pos == goal_pos:
            issues.append("Start and goal positions are identical - trivial solution")
        
        # Verify step budget creates appropriate pressure
        steps_remaining = world_state.get("agent", {}).get("steps_remaining", self.max_steps)
        if steps_remaining != self.max_steps:
            issues.append(f"Initial steps_remaining {steps_remaining} should equal max_steps {self.max_steps}")
        
        return issues
    
    def _check_state_consistency(self, world_state: Dict[str, Any]) -> List[str]:
        """Check for basic state validity"""
        issues = []
        
        # Required fields validation
        required_fields = {
            "agent": dict,
            "tiles": dict,
            "goal_pos": list,
            "start_pos": list,
            "globals": dict
        }
        
        for field, expected_type in required_fields.items():
            if field not in world_state:
                issues.append(f"Missing required field: {field}")
            elif not isinstance(world_state[field], expected_type):
                issues.append(f"Field {field} should be {expected_type.__name__}, got {type(world_state[field]).__name__}")
        
        # Agent state validation
        agent_state = world_state.get("agent", {})
        if "pos" not in agent_state:
            issues.append("Agent missing position")
        elif agent_state["pos"] != world_state.get("start_pos", []):
            issues.append("Agent position should match start_pos")
        
        # Tiles validation
        tiles = world_state.get("tiles", {})
        if "layout" not in tiles:
            issues.append("Tiles missing layout")
        if "size" not in tiles:
            issues.append("Tiles missing size")
        
        # Valid tile types check
        valid_tile_types = {"ice", "water"}
        layout = tiles.get("layout", [])
        for r, row in enumerate(layout):
            for c, tile in enumerate(row):
                if tile not in valid_tile_types:
                    issues.append(f"Invalid tile type '{tile}' at position ({r}, {c})")
        
        return issues
    
    def _is_valid_position(self, pos: List[int], grid_size: List[int]) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1])
    
    def _has_valid_path(self, start: List[int], goal: List[int], layout: List[List[str]], grid_size: List[int]) -> bool:
        """BFS to check if path exists from start to goal avoiding water"""
        queue = deque([tuple(start)])
        visited = {tuple(start)}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
        
        while queue:
            r, c = queue.popleft()
            
            if [r, c] == goal:
                return True
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_size[0] and 0 <= nc < grid_size[1] and 
                    (nr, nc) not in visited and layout[nr][nc] != "water"):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return False
    
    def _find_shortest_path_length(self, start: List[int], goal: List[int], layout: List[List[str]], grid_size: List[int]) -> Optional[int]:
        """BFS to find shortest path length from start to goal"""
        queue = deque([(tuple(start), 0)])
        visited = {tuple(start)}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            (r, c), dist = queue.popleft()
            
            if [r, c] == goal:
                return dist
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_size[0] and 0 <= nc < grid_size[1] and 
                    (nr, nc) not in visited and layout[nr][nc] != "water"):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
        
        return None
    
    def validate_batch(self, levels_dir: str) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate multiple levels in a directory"""
        results = {}
        
        if not os.path.exists(levels_dir):
            return {"error": (False, [f"Directory {levels_dir} does not exist"])}
        
        for filename in os.listdir(levels_dir):
            if filename.endswith('.yaml'):
                world_path = os.path.join(levels_dir, filename)
                world_id = filename[:-5]  # Remove .yaml extension
                results[world_id] = self.validate_level(world_path)
        
        return results

# Usage function for easy integration
def validate_ice_lake_level(world_path: str, config_path: str = "./config.yaml") -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a single ice lake level
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validator = IceLakeValidator(config)
    return validator.validate_level(world_path)

def validate_ice_lake_batch(levels_dir: str, config_path: str = "./config.yaml") -> Dict[str, Tuple[bool, List[str]]]:
    """
    Convenience function to validate multiple ice lake levels
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validator = IceLakeValidator(config)
    return validator.validate_batch(levels_dir)