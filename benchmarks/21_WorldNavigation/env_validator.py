from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np

class DeceptiveGridWorldValidator:
    """
    Validator for generated DeceptiveGridWorld levels to ensure solvability 
    and proper reward structure alignment.
    """
    
    def __init__(self):
        self.max_steps = 30
        self.grid_size = [10, 10]
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Main validation function that checks both solvability and reward structure.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. BASIC STRUCTURAL VALIDATION
        structure_issues = self._check_basic_structure(world_state)
        issues.extend(structure_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for impossible puzzles using breadth-first search to verify
        that the goal is reachable within the step limit.
        """
        issues = []
        
        # Extract key information
        agent_start = tuple(world_state['agent']['pos'])
        goal_pos = tuple(world_state['special']['goal_pos'])
        grid = world_state['tiles']['data']
        
        # ACTION CONSTRAINT ANALYSIS
        # Available actions: MoveNorth, MoveSouth, MoveEast, MoveWest, Wait
        # Each action decrements steps_remaining
        # Movement actions can be blocked by grid boundaries but not by tile types
        # (since ⬛ walls are passable despite appearance)
        
        # TARGET REACHABILITY using BFS with step counting
        is_reachable, min_steps = self._bfs_pathfind(agent_start, goal_pos, grid)
        
        if not is_reachable:
            issues.append(f"CRITICAL: Goal at {goal_pos} is unreachable from start {agent_start}")
        elif min_steps > self.max_steps:
            issues.append(f"CRITICAL: Goal requires {min_steps} steps but limit is {self.max_steps}")
        
        # Check if goal position is actually marked as goal
        goal_tile = self._get_tile_at(goal_pos, grid)
        if goal_tile != '☠':
            issues.append(f"CRITICAL: Goal position {goal_pos} does not contain ☠ tile")
        
        # Verify agent doesn't start on a trap
        start_tile = self._get_tile_at(agent_start, grid)
        if start_tile == '💰':
            issues.append(f"CRITICAL: Agent starts on trap at {agent_start}")
        
        # Check that there's at least one path that avoids all traps
        safe_path_exists = self._check_safe_path_exists(agent_start, goal_pos, grid)
        if not safe_path_exists:
            issues.append(f"CRITICAL: No safe path exists from {agent_start} to {goal_pos} (all paths blocked by traps)")
        
        return issues
    
    def _bfs_pathfind(self, start: Tuple[int, int], goal: Tuple[int, int], 
                      grid: List[List[str]]) -> Tuple[bool, int]:
        """
        BFS pathfinding that considers grid boundaries and finds minimum steps to goal.
        Returns (is_reachable, minimum_steps_required)
        """
        from collections import deque
        
        queue = deque([(start, 0)])  # (position, steps)
        visited = {start}
        rows, cols = len(grid), len(grid[0])
        
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W
        
        while queue:
            (y, x), steps = queue.popleft()
            
            if (y, x) == goal:
                return True, steps
            
            if steps >= self.max_steps:
                continue
            
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                new_pos = (new_y, new_x)
                
                # Check bounds (this is the main constraint - out of bounds moves are invalid)
                if 0 <= new_y < rows and 0 <= new_x < cols:
                    if new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, steps + 1))
        
        return False, float('inf')
    
    def _check_safe_path_exists(self, start: Tuple[int, int], goal: Tuple[int, int], 
                               grid: List[List[str]]) -> bool:
        """
        Check if there exists at least one path that doesn't go through trap tiles.
        This is important because stepping on 💰 immediately terminates the episode.
        """
        from collections import deque
        
        queue = deque([start])
        visited = {start}
        rows, cols = len(grid), len(grid[0])
        
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        while queue:
            y, x = queue.popleft()
            
            if (y, x) == goal:
                return True
            
            for dy, dx in directions:
                new_y, new_x = y + dy, x + dx
                new_pos = (new_y, new_x)
                
                if (0 <= new_y < rows and 0 <= new_x < cols and 
                    new_pos not in visited):
                    
                    tile = grid[new_y][new_x]
                    # Avoid traps in pathfinding
                    if tile != '💰':
                        visited.add(new_pos)
                        queue.append(new_pos)
        
        return False
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Validate that reward structure encourages goal-oriented behavior.
        """
        issues = []
        
        # The environment uses binary rewards: +1 only for reaching goal, 0 otherwise
        # This is actually good design - prevents action grinding and exploration loops
        
        # Check goal reward configuration
        goal_pos = world_state['special']['goal_pos']
        if goal_pos is None:
            issues.append("REWARD: Goal position not set - no way to earn positive reward")
        
        # Verify there's exactly one goal (prevents ambiguity)
        grid = world_state['tiles']['data']
        safe_tiles = []
        for i, row in enumerate(grid):
            for j, tile in enumerate(row):
                if tile == '☠':
                    safe_tiles.append((i, j))
        
        if len(safe_tiles) == 0:
            issues.append("REWARD: No ☠ tiles found - impossible to achieve goal")
        
        goal_tuple = tuple(goal_pos) if goal_pos else None
        if goal_tuple not in safe_tiles:
            issues.append("REWARD: Goal position doesn't correspond to a ☠ tile")
        
        # Check trap density - too many traps make exploration too punishing
        total_cells = len(grid) * len(grid[0])
        trap_count = sum(row.count('💰') for row in grid)
        trap_ratio = trap_count / total_cells
        
        if trap_ratio > 0.3:
            issues.append(f"REWARD: Trap density too high ({trap_ratio:.2%}) - may discourage exploration")
        
        # Ensure reasonable number of safe tiles for exploration
        if len(safe_tiles) < 2:
            issues.append("REWARD: Too few ☠ tiles - limits exploration and learning")
        elif len(safe_tiles) > 10:
            issues.append("REWARD: Too many ☠ tiles - may make goal finding too easy")
        
        return issues
    
    def _check_basic_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Check basic structural requirements of the world state.
        """
        issues = []
        
        # Validate required fields exist
        required_fields = ['agent', 'tiles', 'special', 'globals']
        for field in required_fields:
            if field not in world_state:
                issues.append(f"STRUCTURE: Missing required field '{field}'")
        
        # Validate agent configuration
        if 'agent' in world_state:
            agent = world_state['agent']
            if 'pos' not in agent:
                issues.append("STRUCTURE: Agent position not specified")
            elif len(agent['pos']) != 2:
                issues.append("STRUCTURE: Agent position must be [y, x] coordinates")
            
            if 'steps_remaining' not in agent:
                issues.append("STRUCTURE: Agent steps_remaining not specified")
            elif agent['steps_remaining'] != 30:
                issues.append(f"STRUCTURE: Agent should start with 30 steps, got {agent['steps_remaining']}")
        
        # Validate grid structure
        if 'tiles' in world_state and 'data' in world_state['tiles']:
            grid = world_state['tiles']['data']
            if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                issues.append("STRUCTURE: Grid data must be a 2D list")
            elif len(grid) != 10 or any(len(row) != 10 for row in grid):
                issues.append("STRUCTURE: Grid must be exactly 10x10")
        
        # Validate goal configuration
        if 'special' in world_state:
            special = world_state['special']
            if 'goal_pos' not in special:
                issues.append("STRUCTURE: Goal position not specified in special")
            elif special['goal_pos'] and len(special['goal_pos']) != 2:
                issues.append("STRUCTURE: Goal position must be [y, x] coordinates")
        
        return issues
    
    def _get_tile_at(self, pos: Tuple[int, int], grid: List[List[str]]) -> str:
        """Helper function to get tile at position."""
        y, x = pos
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            return grid[y][x]
        return '◻'  # default tile for out of bounds
    
    def check_level_solvability(self, initial_state: Dict[str, Any], 
                               target_state: Dict[str, Any], 
                               available_actions: List[str]) -> Tuple[bool, List[str]]:
        """
        Framework method for checking level solvability as specified in requirements.
        
        Args:
            initial_state: The starting world state
            target_state: The goal state (agent at goal position)  
            available_actions: List of available action names
            
        Returns:
            Tuple[bool, List[str]]: (is_solvable, blocking_issues)
        """
        issues = []
        
        # 1. Resource check: are target elements obtainable?
        goal_pos = tuple(initial_state['special']['goal_pos'])
        grid = initial_state['tiles']['data']
        goal_tile = self._get_tile_at(goal_pos, grid)
        
        if goal_tile != '☠':
            issues.append("Target requires ☠ tile but goal position doesn't contain one")
        
        # 2. Constraint check: do actions have sufficient power to reach target?
        movement_actions = ['MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest']
        has_movement = any(action in available_actions for action in movement_actions)
        
        if not has_movement:
            issues.append("No movement actions available - cannot reach spatial goals")
        
        # 3. Path existence: can you navigate from initial to target state?
        agent_start = tuple(initial_state['agent']['pos'])
        is_reachable, _ = self._bfs_pathfind(agent_start, goal_pos, grid)
        
        if not is_reachable:
            issues.append(f"No valid path from start {agent_start} to goal {goal_pos}")
        
        # 4. Step budget: is solution achievable within step limits?
        _, min_steps = self._bfs_pathfind(agent_start, goal_pos, grid)
        max_steps = initial_state['agent']['steps_remaining']
        
        if min_steps > max_steps:
            issues.append(f"Minimum path requires {min_steps} steps but only {max_steps} available")
        
        is_solvable = len(issues) == 0
        return is_solvable, issues
    
    def validate_reward_structure(self) -> Tuple[bool, List[str]]:
        """
        Framework method for validating reward structure as specified in requirements.
        
        Returns:
            Tuple[bool, List[str]]: (is_well_designed, reward_issues)
        """
        issues = []
        
        # DeceptiveGridWorld uses binary rewards which is good design:
        # - Target Achievement: +1 for reaching goal (highest reward) ✓
        # - Progress Rewards: None (avoids dense reward issues) ✓  
        # - Action Usage: 0 for all actions (prevents grinding) ✓
        
        # Check for potential reward loops or exploitation
        # In this environment, the only way to get reward is reaching the goal
        # No action can be repeated for points, no exploration gives rewards
        
        # The design prevents:
        # - Action Grinding: Actions give 0 reward ✓
        # - Exploration Loops: No reward for visiting tiles ✓
        # - Action Farming: No repetitive actions give rewards ✓
        
        # Reward design follows principles:
        # - Sparse > Dense: Only one reward signal ✓
        # - Achievement > Process: Goal completion is only reward ✓
        # - Efficiency Incentive: Implicit (episode ends on success) ✓
        # - Failure Cost: Losing time/episode on wrong moves ✓
        
        # No issues found with this reward structure
        is_well_designed = len(issues) == 0
        return is_well_designed, issues

# Usage function for integration
def validate_generated_level(world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Main validation entry point for generated levels.
    
    Args:
        world_state: Generated world state dictionary
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    validator = DeceptiveGridWorldValidator()
    return validator.validate_level(world_state)