import random
import yaml
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import deque
import copy

class QuantumMazeValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get("termination", {}).get("max_steps", 40)
        self.grid_size = config.get("state_template", {}).get("globals", {}).get("grid_size", [10, 10])
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of generated quantum maze levels.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. Basic structural validation
        structural_issues = self._validate_structure(world_state)
        issues.extend(structural_issues)
        
        # 2. Level solvability analysis
        solvability_issues = self._validate_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 4. Action constraint analysis
        action_issues = self._validate_action_constraints(world_state)
        issues.extend(action_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic structural requirements"""
        issues = []
        
        # Check required keys exist
        required_keys = ["globals", "agent", "maze"]
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required key: {key}")
        
        if "globals" in world_state:
            globals_data = world_state["globals"]
            if "start_pos" not in globals_data or "exit_pos" not in globals_data:
                issues.append("Missing start_pos or exit_pos in globals")
            
            # Validate positions are within bounds
            if "start_pos" in globals_data and "grid_size" in globals_data:
                start = globals_data["start_pos"]
                size = globals_data["grid_size"]
                if not (0 <= start[0] < size[0] and 0 <= start[1] < size[1]):
                    issues.append(f"Start position {start} outside grid bounds {size}")
            
            if "exit_pos" in globals_data and "grid_size" in globals_data:
                exit_pos = globals_data["exit_pos"]
                size = globals_data["grid_size"]
                if not (0 <= exit_pos[0] < size[0] and 0 <= exit_pos[1] < size[1]):
                    issues.append(f"Exit position {exit_pos} outside grid bounds {size}")
        
        if "maze" in world_state:
            maze_data = world_state["maze"]
            required_maze_keys = ["wall_probabilities", "collapsed_walls"]
            for key in required_maze_keys:
                if key not in maze_data:
                    issues.append(f"Missing required maze key: {key}")
            
            # Validate wall probabilities are in valid range
            if "wall_probabilities" in maze_data:
                for cell_key, prob in maze_data["wall_probabilities"].items():
                    if not (0.0 <= prob <= 1.0):
                        issues.append(f"Invalid wall probability {prob} for cell {cell_key}")
        
        return issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical solvability analysis using quantum probability simulation"""
        issues = []
        
        try:
            start_pos = world_state["globals"]["start_pos"]
            exit_pos = world_state["globals"]["exit_pos"]
            wall_probs = world_state["maze"]["wall_probabilities"]
            
            # Run multiple simulations to estimate solvability probability
            solvable_count = 0
            total_simulations = 100
            min_solvable_rate = 0.1  # At least 10% of random collapses should be solvable
            
            for _ in range(total_simulations):
                if self._simulate_random_collapse_solvability(start_pos, exit_pos, wall_probs):
                    solvable_count += 1
            
            solvable_rate = solvable_count / total_simulations
            if solvable_rate < min_solvable_rate:
                issues.append(f"Level has very low solvability rate: {solvable_rate:.2%} (minimum: {min_solvable_rate:.2%})")
            
            # Check for fundamental impossibilities
            impossibility_issues = self._check_impossible_patterns(world_state)
            issues.extend(impossibility_issues)
            
        except Exception as e:
            issues.append(f"Error during solvability analysis: {str(e)}")
        
        return issues
    
    def _simulate_random_collapse_solvability(self, start_pos: List[int], exit_pos: List[int], 
                                            wall_probs: Dict[str, float]) -> bool:
        """Simulate a random quantum collapse and check if path exists"""
        # Generate a random collapsed maze state
        collapsed_maze = {}
        for cell_key, prob in wall_probs.items():
            collapsed_maze[cell_key] = "wall" if random.random() < prob else "empty"
        
        # BFS to find path from start to exit
        return self._bfs_path_exists(start_pos, exit_pos, collapsed_maze)
    
    def _bfs_path_exists(self, start: List[int], exit: List[int], collapsed_maze: Dict[str, str]) -> bool:
        """BFS pathfinding in collapsed maze state"""
        queue = deque([(tuple(start), 0)])  # (position, steps)
        visited = set()
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            (x, y), steps = queue.popleft()
            
            if steps > self.max_steps:
                continue
                
            if [x, y] == exit:
                return True
            
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if (new_x < 0 or new_x >= self.grid_size[0] or 
                    new_y < 0 or new_y >= self.grid_size[1]):
                    continue
                
                cell_key = f"{new_x},{new_y}"
                if collapsed_maze.get(cell_key, "empty") != "wall":
                    queue.append(((new_x, new_y), steps + 1))
        
        return False
    
    def _check_impossible_patterns(self, world_state: Dict[str, Any]) -> List[str]:
        """Check for patterns that make levels fundamentally impossible"""
        issues = []
        
        start_pos = world_state["globals"]["start_pos"]
        exit_pos = world_state["globals"]["exit_pos"]
        wall_probs = world_state["maze"]["wall_probabilities"]
        
        # Check if start or exit have wall probability > 0 (they should be guaranteed empty)
        start_key = f"{start_pos[0]},{start_pos[1]}"
        exit_key = f"{exit_pos[0]},{exit_pos[1]}"
        
        if wall_probs.get(start_key, 0) > 0:
            issues.append(f"Start position {start_pos} has wall probability > 0: {wall_probs[start_key]}")
        
        if wall_probs.get(exit_key, 0) > 0:
            issues.append(f"Exit position {exit_pos} has wall probability > 0: {wall_probs[exit_key]}")
        
        # Check if all adjacent cells to start have very high wall probability
        start_adjacent_high_prob = self._check_position_surrounded(start_pos, wall_probs, 0.9)
        if start_adjacent_high_prob:
            issues.append("Start position likely surrounded by walls (all adjacent cells have >90% wall probability)")
        
        # Check if all adjacent cells to exit have very high wall probability  
        exit_adjacent_high_prob = self._check_position_surrounded(exit_pos, wall_probs, 0.9)
        if exit_adjacent_high_prob:
            issues.append("Exit position likely surrounded by walls (all adjacent cells have >90% wall probability)")
        
        # Check corridor bottlenecks - if there are narrow passages with very high wall probability
        bottleneck_issues = self._check_bottlenecks(start_pos, exit_pos, wall_probs)
        issues.extend(bottleneck_issues)
        
        return issues
    
    def _check_position_surrounded(self, pos: List[int], wall_probs: Dict[str, float], 
                                 threshold: float) -> bool:
        """Check if position is likely surrounded by walls"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        high_prob_count = 0
        total_adjacent = 0
        
        for dx, dy in directions:
            adj_x, adj_y = pos[0] + dx, pos[1] + dy
            
            # Check bounds
            if (adj_x >= 0 and adj_x < self.grid_size[0] and 
                adj_y >= 0 and adj_y < self.grid_size[1]):
                total_adjacent += 1
                cell_key = f"{adj_x},{adj_y}"
                if wall_probs.get(cell_key, 0) > threshold:
                    high_prob_count += 1
        
        return high_prob_count == total_adjacent and total_adjacent > 0
    
    def _check_bottlenecks(self, start_pos: List[int], exit_pos: List[int], 
                          wall_probs: Dict[str, float]) -> List[str]:
        """Check for critical bottlenecks that could make level unsolvable"""
        issues = []
        
        # Simple heuristic: check if there are rows/columns with very high wall probability
        # that could create impassable barriers
        
        high_prob_threshold = 0.8
        grid_w, grid_h = self.grid_size
        
        # Check for problematic rows (horizontal barriers)
        for y in range(grid_h):
            high_prob_cells = 0
            total_cells = 0
            for x in range(grid_w):
                if [x, y] not in [start_pos, exit_pos]:  # Exclude start/exit
                    total_cells += 1
                    cell_key = f"{x},{y}"
                    if wall_probs.get(cell_key, 0) > high_prob_threshold:
                        high_prob_cells += 1
            
            if total_cells > 0 and high_prob_cells / total_cells > 0.8:
                # Check if this creates a barrier between start and exit
                if ((start_pos[1] < y < exit_pos[1]) or (exit_pos[1] < y < start_pos[1])):
                    issues.append(f"Potential horizontal barrier at row {y} with {high_prob_cells}/{total_cells} high-probability walls")
        
        # Check for problematic columns (vertical barriers)
        for x in range(grid_w):
            high_prob_cells = 0
            total_cells = 0
            for y in range(grid_h):
                if [x, y] not in [start_pos, exit_pos]:  # Exclude start/exit
                    total_cells += 1
                    cell_key = f"{x},{y}"
                    if wall_probs.get(cell_key, 0) > high_prob_threshold:
                        high_prob_cells += 1
            
            if total_cells > 0 and high_prob_cells / total_cells > 0.8:
                # Check if this creates a barrier between start and exit
                if ((start_pos[0] < x < exit_pos[0]) or (exit_pos[0] < x < start_pos[0])):
                    issues.append(f"Potential vertical barrier at column {x} with {high_prob_cells}/{total_cells} high-probability walls")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate reward structure follows good design principles"""
        issues = []
        
        # Check reward configuration from config
        reward_config = self.config.get("reward", {})
        
        # Verify sparse reward structure
        if not reward_config.get("sparse_reward", False):
            issues.append("Reward structure should be sparse for this environment type")
        
        # Check that main reward comes from reaching exit
        events = reward_config.get("events", [])
        reach_exit_reward = None
        for event in events:
            if event.get("trigger") == "reach_exit":
                reach_exit_reward = event.get("value", 0)
                break
        
        if reach_exit_reward is None:
            issues.append("Missing reward for reaching exit")
        elif reach_exit_reward <= 0:
            issues.append("Reward for reaching exit should be positive")
        elif reach_exit_reward < 1.0:
            issues.append("Reward for reaching exit should be substantial (>=1.0) to encourage goal achievement")
        
        # Check that there are no action-grinding rewards
        action_rewards = []
        for event in events:
            trigger = event.get("trigger", "")
            if any(action in trigger.lower() for action in ["move", "observe", "action"]):
                action_rewards.append(event)
        
        # Warn if action rewards are too high compared to goal reward
        if reach_exit_reward and action_rewards:
            for action_event in action_rewards:
                action_value = action_event.get("value", 0)
                if action_value > reach_exit_reward * 0.1:  # Action rewards shouldn't be >10% of goal reward
                    issues.append(f"Action reward {action_value} for '{action_event.get('trigger')}' is too high compared to goal reward {reach_exit_reward}")
        
        return issues
    
    def _validate_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that actions have appropriate constraints and power"""
        issues = []
        
        # Check step budget is reasonable
        if self.max_steps < 10:
            issues.append(f"Max steps ({self.max_steps}) may be too low for meaningful exploration")
        elif self.max_steps > 200:
            issues.append(f"Max steps ({self.max_steps}) may be too high, reducing challenge")
        
        # Validate that the step budget allows for reasonable solutions
        start_pos = world_state["globals"]["start_pos"]
        exit_pos = world_state["globals"]["exit_pos"]
        
        # Manhattan distance as minimum possible steps
        manhattan_distance = abs(exit_pos[0] - start_pos[0]) + abs(exit_pos[1] - start_pos[1])
        
        if self.max_steps < manhattan_distance * 2:
            issues.append(f"Max steps ({self.max_steps}) may be insufficient. Manhattan distance is {manhattan_distance}, recommend at least {manhattan_distance * 2} steps")
        
        # Check action space completeness
        transition_config = self.config.get("transition", {})
        actions = transition_config.get("actions", [])
        action_names = [action.get("name", "") for action in actions]
        
        required_actions = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]
        for required_action in required_actions:
            if required_action not in action_names:
                issues.append(f"Missing required action: {required_action}")
        
        if "OBSERVE" not in action_names:
            issues.append("Missing OBSERVE action - critical for quantum maze mechanics")
        
        return issues

def validate_quantum_maze_level(world_state: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Main validation function"""
    validator = QuantumMazeValidator(config)
    return validator.validate_level(world_state)

# Example usage for integration:
if __name__ == "__main__":
    # Load config and world state for testing
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with a generated level
    from env_generate import QuantumMazeGenerator
    generator = QuantumMazeGenerator("test", config)
    world_id = generator.generate(seed=42)
    
    with open(f"./levels/{world_id}.yaml", 'r') as f:
        world_state = yaml.safe_load(f)
    
    is_valid, issues = validate_quantum_maze_level(world_state, config)
    
    print(f"Level validation result: {'VALID' if is_valid else 'INVALID'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")