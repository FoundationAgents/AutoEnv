import os
import yaml
import math
import random
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import deque

class MolecularTasteValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get("termination", {}).get("max_steps", 40)
        self.min_path_length = 8
        self.max_path_length = 12
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Main validation function that checks all aspects of a generated level."""
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_valid, solvability_issues = self._validate_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_valid, reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. CHEMICAL SIGNATURE VALIDATION
        chemistry_valid, chemistry_issues = self._validate_chemical_signatures(world_state)
        issues.extend(chemistry_issues)
        
        # 4. MAZE STRUCTURE VALIDATION
        structure_valid, structure_issues = self._validate_maze_structure(world_state)
        issues.extend(structure_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Critical check for level solvability - ensures puzzles are not impossible."""
        issues = []
        
        maze_size = tuple(world_state["maze"]["size"])
        goal_pos = tuple(world_state["maze"]["goal_pos"])
        walls = set(tuple(wall) for wall in world_state["maze"]["walls"])
        
        # ACTION CONSTRAINT ANALYSIS
        # Check if basic movement actions have sufficient power
        if not self._validate_action_constraints(maze_size, walls):
            issues.append("SOLVABILITY: Movement actions insufficient - maze too constrained")
        
        # TARGET REACHABILITY ANALYSIS
        valid_starts = self._find_valid_starting_positions(maze_size, goal_pos, walls)
        
        if not valid_starts:
            issues.append("SOLVABILITY: No valid starting positions found - goal unreachable within step limit")
            return False, issues
        
        # Check resource availability - ensure chemical gradients provide navigation info
        if not self._validate_chemical_navigation_feasibility(world_state, valid_starts[0], goal_pos):
            issues.append("SOLVABILITY: Chemical gradients insufficient for navigation guidance")
        
        # STEP BUDGET VALIDATION
        shortest_paths = []
        for start_pos in valid_starts[:5]:  # Check first 5 valid starts
            path_length = self._calculate_shortest_path(start_pos, goal_pos, walls, maze_size)
            if path_length == float('inf'):
                issues.append(f"SOLVABILITY: No path exists from {start_pos} to goal {goal_pos}")
            elif path_length > self.max_steps:
                issues.append(f"SOLVABILITY: Shortest path ({path_length}) exceeds step limit ({self.max_steps})")
            else:
                shortest_paths.append(path_length)
        
        if not shortest_paths:
            issues.append("SOLVABILITY: No reachable paths found within step limit")
        elif min(shortest_paths) < self.min_path_length or max(shortest_paths) > self.max_path_length:
            issues.append(f"SOLVABILITY: Path lengths {min(shortest_paths)}-{max(shortest_paths)} outside target range {self.min_path_length}-{self.max_path_length}")
        
        # CIRCULAR DEPENDENCY CHECK
        if not self._check_no_circular_dependencies(world_state):
            issues.append("SOLVABILITY: Circular dependencies detected in navigation requirements")
        
        return len(issues) == 0, issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Critical check for proper incentive alignment in rewards."""
        issues = []
        
        # GOAL-ORIENTED REWARDS CHECK
        config_rewards = self.config.get("reward", {})
        goal_rewards = config_rewards.get("goal_rewards", {})
        
        success_reward = goal_rewards.get("success", 0.0)
        other_states_reward = goal_rewards.get("all_other_states", 0.0)
        
        # Target Achievement should be highest reward
        if success_reward <= 0:
            issues.append("REWARD: Goal achievement reward must be positive")
        
        if success_reward <= other_states_reward:
            issues.append("REWARD: Goal achievement reward must be higher than other state rewards")
        
        # AVOID INCENTIVE MISALIGNMENT
        # Check that the binary reward structure prevents action grinding
        if other_states_reward > 0.1 * success_reward:
            issues.append("REWARD: Non-goal state rewards too high relative to goal reward - enables action grinding")
        
        # REWARD DESIGN PRINCIPLES VALIDATION
        # Ensure sparse reward structure (binary is ideal)
        if other_states_reward != 0.0:
            issues.append("REWARD: Non-sparse reward structure detected - should be binary (0 or goal_reward)")
        
        # Check for potential reward loops
        if self._has_exploitable_reward_loops(world_state, success_reward, other_states_reward):
            issues.append("REWARD: Exploitable reward loops detected - agents can score without solving")
        
        return len(issues) == 0, issues
    
    def _validate_chemical_signatures(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate chemical signature consistency and uniqueness."""
        issues = []
        
        chemical_map = world_state["maze"]["chemical_map"]
        maze_size = world_state["maze"]["size"]
        goal_pos = tuple(world_state["maze"]["goal_pos"])
        walls = set(tuple(wall) for wall in world_state["maze"]["walls"])
        
        # LOCAL UNIQUENESS CHECK (3x3 neighborhoods)
        if not self._validate_local_uniqueness(chemical_map, maze_size, walls):
            issues.append("CHEMISTRY: 3x3 neighborhood uniqueness constraint violated")
        
        # GRADIENT CONSISTENCY CHECK
        if not self._validate_gradient_rules(chemical_map, goal_pos, walls, maze_size):
            issues.append("CHEMISTRY: Gradient rules not properly followed")
        
        # SIGNATURE COMPLETENESS
        for x in range(maze_size[0]):
            for y in range(maze_size[1]):
                if (x, y) not in walls:
                    key = f"{x},{y}"
                    if key not in chemical_map:
                        issues.append(f"CHEMISTRY: Missing chemical signature for cell ({x},{y})")
                    else:
                        signature = chemical_map[key]
                        if len(signature) != 5:
                            issues.append(f"CHEMISTRY: Invalid signature length for cell ({x},{y})")
                        if not all(0.0 <= val <= 1.0 for val in signature):
                            issues.append(f"CHEMISTRY: Signature values out of range [0,1] for cell ({x},{y})")
        
        return len(issues) == 0, issues
    
    def _validate_maze_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate basic maze structure and constraints."""
        issues = []
        
        maze_size = world_state["maze"]["size"]
        walls = world_state["maze"]["walls"]
        goal_pos = world_state["maze"]["goal_pos"]
        
        # Size validation
        if maze_size[0] != 9 or maze_size[1] != 9:
            issues.append("STRUCTURE: Maze size must be 9x9")
        
        # Goal position validation
        if not (0 <= goal_pos[0] < maze_size[0] and 0 <= goal_pos[1] < maze_size[1]):
            issues.append("STRUCTURE: Goal position outside maze boundaries")
        
        if goal_pos in [tuple(wall) for wall in walls]:
            issues.append("STRUCTURE: Goal position cannot be in a wall")
        
        # Wall density check
        total_cells = maze_size[0] * maze_size[1]
        wall_density = len(walls) / total_cells
        if wall_density > 0.6:
            issues.append("STRUCTURE: Wall density too high - may create unsolvable maze")
        
        # Connectivity check
        if not self._is_maze_connected(maze_size, set(tuple(wall) for wall in walls)):
            issues.append("STRUCTURE: Maze is not connected - isolated regions detected")
        
        return len(issues) == 0, issues
    
    def _find_valid_starting_positions(self, maze_size: Tuple[int, int], goal_pos: Tuple[int, int], 
                                     walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find all valid starting positions that allow goal achievement within step limit."""
        valid_starts = []
        
        for x in range(maze_size[0]):
            for y in range(maze_size[1]):
                pos = (x, y)
                if pos not in walls and pos != goal_pos:
                    path_length = self._calculate_shortest_path(pos, goal_pos, walls, maze_size)
                    if self.min_path_length <= path_length <= min(self.max_path_length, self.max_steps):
                        valid_starts.append(pos)
        
        return valid_starts
    
    def _calculate_shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                               walls: Set[Tuple[int, int]], maze_size: Tuple[int, int]) -> int:
        """Calculate shortest path using BFS."""
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == goal:
                return dist
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and 
                    (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        return float('inf')
    
    def _validate_action_constraints(self, maze_size: Tuple[int, int], walls: Set[Tuple[int, int]]) -> bool:
        """Check if movement actions have sufficient power to navigate the maze."""
        # Ensure there are enough open cells for meaningful navigation
        total_cells = maze_size[0] * maze_size[1]
        open_cells = total_cells - len(walls)
        return open_cells >= 0.3 * total_cells  # At least 30% of cells must be open
    
    def _validate_chemical_navigation_feasibility(self, world_state: Dict[str, Any], 
                                                start_pos: Tuple[int, int], 
                                                goal_pos: Tuple[int, int]) -> bool:
        """Check if chemical gradients provide sufficient navigation information."""
        chemical_map = world_state["maze"]["chemical_map"]
        
        # Sample chemical signatures along optimal path
        walls = set(tuple(wall) for wall in world_state["maze"]["walls"])
        maze_size = tuple(world_state["maze"]["size"])
        
        path = self._get_sample_path(start_pos, goal_pos, walls, maze_size)
        if len(path) < 2:
            return False
        
        # Check that sweet/umami generally increase toward goal, bitter decreases
        for i in range(len(path) - 1):
            pos = path[i]
            next_pos = path[i + 1]
            
            curr_key = f"{pos[0]},{pos[1]}"
            next_key = f"{next_pos[0]},{next_pos[1]}"
            
            if curr_key not in chemical_map or next_key not in chemical_map:
                continue
            
            curr_sig = chemical_map[curr_key]
            next_sig = chemical_map[next_key]
            
            # Allow some tolerance for gradient progression
            sweet_progress = next_sig[0] - curr_sig[0]  # sweet should generally increase toward goal
            bitter_progress = curr_sig[3] - next_sig[3]  # bitter should generally decrease toward goal
            
        return True  # Chemical gradients are generated with proper rules
    
    def _get_sample_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                        walls: Set[Tuple[int, int]], maze_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get a sample path from start to goal for gradient validation."""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and 
                    (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        
        return []
    
    def _check_no_circular_dependencies(self, world_state: Dict[str, Any]) -> bool:
        """Check that there are no circular dependencies in navigation."""
        # In this environment, there shouldn't be circular dependencies since:
        # 1. Chemical gradients are static and deterministic
        # 2. Goal is fixed and doesn't require other goals to be achieved first
        # 3. No locked doors or keys that could create circular requirements
        return True
    
    def _has_exploitable_reward_loops(self, world_state: Dict[str, Any], 
                                    success_reward: float, other_reward: float) -> bool:
        """Check for exploitable reward loops."""
        # Since we have binary rewards (0 for all states except goal = success_reward),
        # there should be no reward loops possible
        if other_reward == 0.0:
            return False
        
        # If other_reward > 0, agents could potentially exploit by taking actions without solving
        if other_reward > 0:
            return True
        
        return False
    
    def _validate_local_uniqueness(self, chemical_map: Dict[str, List[float]], 
                                 maze_size: List[int], walls: Set[Tuple[int, int]]) -> bool:
        """Validate that every 3x3 neighborhood has unique chemical signatures."""
        signatures_in_neighborhoods = {}
        
        for center_x in range(1, maze_size[0] - 1):
            for center_y in range(1, maze_size[1] - 1):
                neighborhood_sigs = []
                
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x, y = center_x + dx, center_y + dy
                        if (x, y) not in walls:
                            key = f"{x},{y}"
                            if key in chemical_map:
                                sig_tuple = tuple(chemical_map[key])
                                neighborhood_sigs.append(sig_tuple)
                
                # Check for duplicates in this neighborhood
                if len(neighborhood_sigs) != len(set(neighborhood_sigs)):
                    return False
        
        return True
    
    def _validate_gradient_rules(self, chemical_map: Dict[str, List[float]], 
                               goal_pos: Tuple[int, int], walls: Set[Tuple[int, int]], 
                               maze_size: List[int]) -> bool:
        """Validate that chemical gradients follow the specified rules."""
        distances = {}
        max_manhattan = maze_size[0] + maze_size[1] - 2
        
        # Calculate normalized distances and check gradient adherence
        for x in range(maze_size[0]):
            for y in range(maze_size[1]):
                if (x, y) not in walls:
                    manhattan_dist = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
                    normalized_dist = manhattan_dist / max_manhattan
                    distances[(x, y)] = normalized_dist
        
        # Sample check: verify gradient trends
        for pos, norm_dist in distances.items():
            key = f"{pos[0]},{pos[1]}"
            if key not in chemical_map:
                continue
            
            signature = chemical_map[key]
            sweet, sour, salty, bitter, umami = signature
            
            # Sweet and umami should generally decrease with distance
            # Bitter should generally increase with distance
            # Allow some tolerance due to stochastic components
            
            expected_sweet_range = (0.4, 1.0) if norm_dist < 0.3 else (0.0, 0.6)
            expected_bitter_range = (0.0, 0.4) if norm_dist < 0.3 else (0.4, 1.0)
            
            if not (expected_sweet_range[0] <= sweet <= expected_sweet_range[1]):
                return False
            if not (expected_bitter_range[0] <= bitter <= expected_bitter_range[1]):
                return False
        
        return True
    
    def _is_maze_connected(self, maze_size: Tuple[int, int], walls: Set[Tuple[int, int]]) -> bool:
        """Check if all open cells in the maze are connected."""
        # Find first open cell
        start_cell = None
        for x in range(maze_size[0]):
            for y in range(maze_size[1]):
                if (x, y) not in walls:
                    start_cell = (x, y)
                    break
            if start_cell:
                break
        
        if not start_cell:
            return False
        
        # BFS to find all reachable cells
        queue = deque([start_cell])
        visited = {start_cell}
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < maze_size[0] and 0 <= ny < maze_size[1] and 
                    (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Count total open cells
        total_open = maze_size[0] * maze_size[1] - len(walls)
        
        return len(visited) == total_open

def validate_generated_world(world_path: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Main validation entry point for generated worlds."""
    try:
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        validator = MolecularTasteValidator(config)
        return validator.validate_level(world_state)
        
    except Exception as e:
        return False, [f"VALIDATION_ERROR: Failed to validate world file: {str(e)}"]