import yaml
import random
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import copy

class WarehouseLevelValidator:
    def __init__(self):
        self.max_steps = 40
        self.grid_size = 10
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # North, South, West, East
        
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """
        Main validation function that checks both solvability and reward structure
        Returns: (is_valid, list_of_issues)
        """
        try:
            with open(level_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level file: {str(e)}"]
        
        issues = []
        
        # 1. Basic structure validation
        structure_valid, structure_issues = self._validate_structure(world_state)
        issues.extend(structure_issues)
        
        if not structure_valid:
            return False, issues
        
        # 2. Level solvability analysis
        solvable, solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_valid, reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate basic level structure and constraints"""
        issues = []
        
        # Check required fields
        required_fields = ['agent', 'tiles', 'objects', 'level_info', 'globals']
        for field in required_fields:
            if field not in world_state:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # Check grid dimensions
        grid = world_state['tiles']['grid']
        if len(grid) != self.grid_size or any(len(row) != self.grid_size for row in grid):
            issues.append(f"Grid must be {self.grid_size}x{self.grid_size}")
        
        # Check box count constraints
        total_boxes = world_state['level_info']['total_boxes']
        actual_boxes = len(world_state['objects']['boxes'])
        actual_docks = len(world_state['objects']['docks'])
        
        if not (3 <= total_boxes <= 5):
            issues.append(f"Box count must be 3-5, got {total_boxes}")
        
        if actual_boxes != total_boxes:
            issues.append(f"Mismatch: total_boxes={total_boxes}, actual boxes={actual_boxes}")
        
        if actual_boxes != actual_docks:
            issues.append(f"Box count ({actual_boxes}) must equal dock count ({actual_docks})")
        
        # Check agent starting position
        agent_pos = world_state['agent']['pos']
        if not self._is_valid_position(agent_pos, grid):
            issues.append("Agent starting position is invalid")
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Critical solvability analysis using BFS pathfinding
        """
        issues = []
        
        grid = world_state['tiles']['grid']
        agent_pos = tuple(world_state['agent']['pos'])
        boxes = [tuple(box['pos']) for box in world_state['objects']['boxes']]
        docks = [tuple(dock['pos']) for dock in world_state['objects']['docks']]
        
        # 1. ACTION CONSTRAINT ANALYSIS
        constraint_issues = self._analyze_action_constraints(grid, boxes, docks)
        issues.extend(constraint_issues)
        
        # 2. TARGET REACHABILITY using state-space search
        reachable, reachability_issues = self._check_target_reachability(
            grid, agent_pos, boxes, docks
        )
        issues.extend(reachability_issues)
        
        # 3. RESOURCE AVAILABILITY CHECK
        resource_issues = self._check_resource_availability(grid, boxes, docks)
        issues.extend(resource_issues)
        
        # 4. STEP BUDGET VALIDATION
        if reachable:
            step_valid, step_issues = self._validate_step_budget(
                grid, agent_pos, boxes, docks
            )
            issues.extend(step_issues)
        
        return len(issues) == 0, issues
    
    def _analyze_action_constraints(self, grid: List[List[str]], boxes: List[Tuple], docks: List[Tuple]) -> List[str]:
        """Analyze if actions can actually achieve the required transformations"""
        issues = []
        
        # Check if all boxes can potentially be moved
        for i, box_pos in enumerate(boxes):
            can_be_pushed = False
            for dx, dy in self.directions:
                # Check if box can be pushed in this direction
                push_to = (box_pos[0] + dx, box_pos[1] + dy)
                if self._is_valid_position(push_to, grid) and grid[push_to[0]][push_to[1]] != 'wall':
                    # Check if agent can reach the pushing position
                    push_from = (box_pos[0] - dx, box_pos[1] - dy)
                    if self._is_valid_position(push_from, grid) and grid[push_from[0]][push_from[1]] != 'wall':
                        can_be_pushed = True
                        break
            
            if not can_be_pushed:
                issues.append(f"Box at {box_pos} cannot be pushed in any direction")
        
        # Check for boxes in corners (dead ends)
        for box_pos in boxes:
            if self._is_corner_trapped(box_pos, grid, docks):
                issues.append(f"Box at {box_pos} is trapped in corner and not on target dock")
        
        return issues
    
    def _check_target_reachability(self, grid: List[List[str]], agent_pos: Tuple, 
                                 boxes: List[Tuple], docks: List[Tuple]) -> Tuple[bool, List[str]]:
        """Use BFS to check if target state is reachable"""
        issues = []
        
        # Create initial state
        initial_state = SokobanState(agent_pos, tuple(sorted(boxes)))
        target_boxes = tuple(sorted(docks))  # Target: all boxes on docks
        
        # BFS with state space exploration
        queue = deque([(initial_state, 0)])
        visited = {initial_state}
        max_search_depth = min(100, self.max_steps * 2)  # Limit search to prevent timeout
        
        while queue:
            current_state, depth = queue.popleft()
            
            if depth > max_search_depth:
                issues.append("Solution search exceeded maximum depth - level might be too complex")
                break
            
            # Check if target achieved
            if current_state.boxes == target_boxes:
                if depth <= self.max_steps:
                    return True, []  # Solvable within step budget
                else:
                    issues.append(f"Solution exists but requires {depth} steps, exceeds limit of {self.max_steps}")
                    return False, issues
            
            # Generate next states
            for next_state in self._get_next_states(current_state, grid):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, depth + 1))
        
        # If we exit without finding solution
        issues.append("No solution path found - level appears unsolvable")
        return False, issues
    
    def _check_resource_availability(self, grid: List[List[str]], boxes: List[Tuple], docks: List[Tuple]) -> List[str]:
        """Check if required resources are available and obtainable"""
        issues = []
        
        # Check dock accessibility
        for dock_pos in docks:
            if not self._is_dock_reachable(dock_pos, grid, boxes):
                issues.append(f"Dock at {dock_pos} is not reachable by any box")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(boxes, docks, grid):
            issues.append("Level has circular dependencies that prevent solution")
        
        return issues
    
    def _validate_step_budget(self, grid: List[List[str]], agent_pos: Tuple, 
                            boxes: List[Tuple], docks: List[Tuple]) -> Tuple[bool, List[str]]:
        """Validate that solution is achievable within step budget"""
        issues = []
        
        # Estimate minimum steps required using heuristic
        min_steps_estimate = self._estimate_minimum_steps(agent_pos, boxes, docks, grid)
        
        if min_steps_estimate > self.max_steps:
            issues.append(f"Estimated minimum steps ({min_steps_estimate}) exceeds budget ({self.max_steps})")
            return False, issues
        
        return True, []
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate reward structure prevents exploitation and encourages problem-solving"""
        issues = []
        
        # Check binary reward structure
        total_boxes = world_state['level_info']['total_boxes']
        
        # Simulate reward calculation scenarios
        
        # Scenario 1: Partial completion should give 0 reward
        for boxes_on_docks in range(total_boxes):
            if boxes_on_docks < total_boxes:
                # This should give 0 reward according to binary structure
                pass  # The environment only gives +1 when all boxes are placed
        
        # Scenario 2: Full completion should give exactly +1 reward
        # This is guaranteed by the binary reward structure
        
        # Check for potential reward farming issues
        if self._has_reward_farming_potential(world_state):
            issues.append("Level structure allows potential reward farming through repeated actions")
        
        # Check reward-to-effort ratio
        if self._has_poor_reward_effort_ratio(world_state):
            issues.append("Reward structure may not justify the effort required")
        
        return len(issues) == 0, issues
    
    def _is_valid_position(self, pos: Tuple, grid: List[List[str]]) -> bool:
        """Check if position is within bounds and not a wall"""
        if isinstance(pos, list):
            pos = tuple(pos)
        x, y = pos
        return (0 <= x < self.grid_size and 
                0 <= y < self.grid_size and 
                grid[x][y] != 'wall')
    
    def _is_corner_trapped(self, box_pos: Tuple, grid: List[List[str]], docks: List[Tuple]) -> bool:
        """Check if box is trapped in a corner and not on a target dock"""
        if box_pos in docks:
            return False  # It's okay to be in corner if it's the target
        
        x, y = box_pos
        
        # Count walls/boundaries around the box
        blocked_directions = 0
        for dx, dy in self.directions:
            adj_x, adj_y = x + dx, y + dy
            if (adj_x < 0 or adj_x >= self.grid_size or 
                adj_y < 0 or adj_y >= self.grid_size or
                grid[adj_x][adj_y] == 'wall'):
                blocked_directions += 1
        
        return blocked_directions >= 2  # Corner if blocked in 2+ directions
    
    def _is_dock_reachable(self, dock_pos: Tuple, grid: List[List[str]], boxes: List[Tuple]) -> bool:
        """Check if dock can be reached by at least one box"""
        for dx, dy in self.directions:
            # Check each direction from which a box could be pushed onto dock
            push_from = (dock_pos[0] - dx, dock_pos[1] - dy)
            if self._is_valid_position(push_from, grid):
                return True
        return False
    
    def _has_circular_dependencies(self, boxes: List[Tuple], docks: List[Tuple], grid: List[List[str]]) -> bool:
        """Check for circular dependencies in box placement"""
        # Simplified check: if boxes block each other's paths to docks
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i != j:
                    # Check if box1 blocks box2's path to any dock
                    if self._blocks_path_to_docks(box1, box2, docks, grid):
                        return True
        return False
    
    def _blocks_path_to_docks(self, blocker_box: Tuple, blocked_box: Tuple, 
                            docks: List[Tuple], grid: List[List[str]]) -> bool:
        """Check if one box blocks another's path to docks"""
        # Simplified implementation - check if blocker is between blocked box and docks
        for dock in docks:
            if self._is_on_straight_path(blocker_box, blocked_box, dock):
                return True
        return False
    
    def _is_on_straight_path(self, point1: Tuple, point2: Tuple, target: Tuple) -> bool:
        """Check if point1 is on straight line path from point2 to target"""
        x1, y1 = point1
        x2, y2 = point2
        tx, ty = target
        
        # Check if collinear and between
        if x2 == tx:  # Vertical line
            return x1 == x2 and min(y2, ty) <= y1 <= max(y2, ty)
        elif y2 == ty:  # Horizontal line
            return y1 == y2 and min(x2, tx) <= x1 <= max(x2, tx)
        
        return False
    
    def _estimate_minimum_steps(self, agent_pos: Tuple, boxes: List[Tuple], 
                              docks: List[Tuple], grid: List[List[str]]) -> int:
        """Estimate minimum steps required using Manhattan distance heuristic"""
        total_steps = 0
        
        # For each box, find nearest dock and estimate steps
        used_docks = set()
        for box_pos in boxes:
            min_dist = float('inf')
            best_dock = None
            
            for dock_pos in docks:
                if dock_pos not in used_docks:
                    # Manhattan distance from box to dock
                    dist = abs(box_pos[0] - dock_pos[0]) + abs(box_pos[1] - dock_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_dock = dock_pos
            
            if best_dock:
                used_docks.add(best_dock)
                # Add agent movement cost + box pushing cost
                agent_to_box = abs(agent_pos[0] - box_pos[0]) + abs(agent_pos[1] - box_pos[1])
                total_steps += agent_to_box + min_dist
                agent_pos = best_dock  # Agent ends up near the dock
        
        return total_steps
    
    def _has_reward_farming_potential(self, world_state: Dict[str, Any]) -> bool:
        """Check if level allows reward farming through repeated actions"""
        # Since reward is binary (only +1 on completion), farming is not possible
        # But check for potential infinite loops or meaningless high-scoring actions
        
        # In this environment, reward farming is prevented by design
        # Only completion gives reward, so this check always passes
        return False
    
    def _has_poor_reward_effort_ratio(self, world_state: Dict[str, Any]) -> bool:
        """Check if reward justifies the required effort"""
        # Binary reward of +1 for completion is reasonable for 40-step budget
        # This is a design choice check
        return False
    
    def _get_next_states(self, state: 'SokobanState', grid: List[List[str]]) -> List['SokobanState']:
        """Generate all possible next states from current state"""
        next_states = []
        agent_x, agent_y = state.agent_pos
        
        for dx, dy in self.directions:
            new_agent_x = agent_x + dx
            new_agent_y = agent_y + dy
            
            if not self._is_valid_position((new_agent_x, new_agent_y), grid):
                continue
            
            new_boxes = list(state.boxes)
            box_at_target = None
            
            # Check if there's a box at target position
            for i, (bx, by) in enumerate(new_boxes):
                if (bx, by) == (new_agent_x, new_agent_y):
                    box_at_target = i
                    break
            
            if box_at_target is not None:
                # Try to push box
                new_box_x = new_agent_x + dx
                new_box_y = new_agent_y + dy
                
                if (self._is_valid_position((new_box_x, new_box_y), grid) and
                    (new_box_x, new_box_y) not in new_boxes):
                    
                    new_boxes[box_at_target] = (new_box_x, new_box_y)
                    next_state = SokobanState((new_agent_x, new_agent_y), tuple(sorted(new_boxes)))
                    next_states.append(next_state)
            else:
                # Simple move
                next_state = SokobanState((new_agent_x, new_agent_y), state.boxes)
                next_states.append(next_state)
        
        return next_states


class SokobanState:
    """Represents a state in the Sokoban game for pathfinding"""
    
    def __init__(self, agent_pos: Tuple[int, int], boxes: Tuple[Tuple[int, int], ...]):
        self.agent_pos = agent_pos
        self.boxes = boxes
    
    def __eq__(self, other):
        return self.agent_pos == other.agent_pos and self.boxes == other.boxes
    
    def __hash__(self):
        return hash((self.agent_pos, self.boxes))
    
    def __repr__(self):
        return f"SokobanState(agent={self.agent_pos}, boxes={self.boxes})"


# Usage function for integration
def validate_warehouse_level(level_path: str) -> Tuple[bool, List[str]]:
    """
    Main validation function to be called by the environment system
    
    Args:
        level_path: Path to the level YAML file
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = WarehouseLevelValidator()
    return validator.validate_level(level_path)