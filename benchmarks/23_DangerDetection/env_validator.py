import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

class TreasureHuntValidator:
    def __init__(self):
        self.grid_size = 8
        self.max_steps = 30
        self.start_pos = (0, 0)
        
    def _key_to_pos(self, key: str) -> Tuple[int, int]:
        """Convert string key 'x,y' to position tuple (x, y)"""
        parts = key.split(',')
        return (int(parts[0]), int(parts[1]))
    
    def validate_level(self, world_file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated treasure hunt level for solvability and correctness.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Load world file
            with open(world_file_path, 'r') as f:
                world_state = yaml.safe_load(f)
            
            # 1. LEVEL STRUCTURE VALIDATION
            structure_issues = self._validate_structure(world_state)
            issues.extend(structure_issues)
            
            # 2. RESOURCE COUNT VALIDATION  
            resource_issues = self._validate_resources(world_state)
            issues.extend(resource_issues)
            
            # 3. LEVEL SOLVABILITY ANALYSIS
            solvability_issues = self._validate_solvability(world_state)
            issues.extend(solvability_issues)
            
            # 4. REWARD STRUCTURE VALIDATION (environment-level check)
            reward_issues = self._validate_reward_structure(world_state)
            issues.extend(reward_issues)
            
        except Exception as e:
            issues.append(f"Failed to load or parse world file: {str(e)}")
            
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic world structure and required components."""
        issues = []
        
        # Check required top-level keys
        required_keys = ["globals", "agent", "grid"]
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required key: {key}")
        
        # Validate globals
        if "globals" in world_state:
            globals_data = world_state["globals"]
            if "grid_size" not in globals_data or globals_data["grid_size"] != [8, 8]:
                issues.append("Invalid grid_size: must be [8, 8]")
            if "max_steps" not in globals_data or globals_data["max_steps"] != 30:
                issues.append("Invalid max_steps: must be 30")
        
        # Validate agent starting position
        if "agent" in world_state:
            agent_data = world_state["agent"]
            if "pos" not in agent_data or agent_data["pos"] != [0, 0]:
                issues.append("Invalid agent starting position: must be [0, 0]")
        
        # Validate grid structure
        if "grid" in world_state:
            grid_data = world_state["grid"]
            if "icons" not in grid_data:
                issues.append("Missing grid.icons")
            if "revealed" not in grid_data:
                issues.append("Missing grid.revealed")
        
        return issues
    
    def _validate_resources(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate icon counts and placement."""
        issues = []
        
        if "grid" not in world_state or "icons" not in world_state["grid"]:
            issues.append("Cannot validate resources: missing grid.icons")
            return issues
        
        icons = world_state["grid"]["icons"]
        
        # Count each icon type
        bomb_count = 0
        flower_count = 0
        empty_count = 0
        total_count = 0
        
        for pos_key, icon_type in icons.items():
            total_count += 1
            
            # Validate position format - expecting string format "x,y"
            try:
                x, y = self._key_to_pos(pos_key)
                if not (0 <= x < 8 and 0 <= y < 8):
                    issues.append(f"Invalid position {pos_key}: out of grid bounds")
            except (ValueError, IndexError):
                issues.append(f"Invalid position format {pos_key}: must be 'x,y' coordinates")
                continue
            
            # Count icon types
            if icon_type == "bomb":
                bomb_count += 1
            elif icon_type == "flower":
                flower_count += 1
            elif icon_type == "empty":
                empty_count += 1
            else:
                issues.append(f"Invalid icon type '{icon_type}' at position {pos_key}")
        
        # Validate counts
        if bomb_count != 1:
            issues.append(f"Invalid bomb count: expected 1, found {bomb_count}")
        if flower_count != 10:
            issues.append(f"Invalid flower count: expected 10, found {flower_count}")
        if empty_count != 53:
            issues.append(f"Invalid empty count: expected 53, found {empty_count}")
        if total_count != 64:
            issues.append(f"Invalid total tile count: expected 64, found {total_count}")
        
        return issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Critical solvability analysis - ensure the bomb is reachable within step limit.
        """
        issues = []
        
        if "grid" not in world_state or "icons" not in world_state["grid"]:
            issues.append("Cannot validate solvability: missing grid data")
            return issues
        
        icons = world_state["grid"]["icons"]
        
        # Find bomb position
        bomb_pos = None
        flower_positions = set()
        
        for pos_key, icon_type in icons.items():
            try:
                pos = self._key_to_pos(pos_key)
                if icon_type == "bomb":
                    bomb_pos = pos
                elif icon_type == "flower":
                    flower_positions.add(pos)
            except (ValueError, IndexError):
                continue  # Skip invalid position formats
        
        if bomb_pos is None:
            issues.append("CRITICAL: No bomb found - level is unsolvable")
            return issues
        
        # ACTION CONSTRAINT ANALYSIS: Understand movement limitations
        # Agent can move in 4 directions, reveal tiles, or wait
        # Moving to unrevealed tiles auto-reveals them
        # Stepping on flowers terminates episode immediately
        
        # TARGET REACHABILITY: Use BFS to find shortest path avoiding flowers
        path_length = self._find_shortest_safe_path(self.start_pos, bomb_pos, flower_positions)
        
        if path_length == -1:
            issues.append("CRITICAL: Bomb position is unreachable - blocked by flowers or boundaries")
        elif path_length > self.max_steps:
            issues.append(f"CRITICAL: Bomb requires {path_length} steps but only {self.max_steps} available")
        
        # Additional solvability checks
        
        # Check if starting position is safe
        if self.start_pos in flower_positions:
            issues.append("CRITICAL: Agent starts on flower tile - immediate death")
        
        if self.start_pos == bomb_pos:
            # Technically solvable in 0 moves, but unusual
            issues.append("WARNING: Bomb is at starting position - trivial solution")
        
        # Check for flower clusters that might create impossible barriers
        barrier_issues = self._check_flower_barriers(flower_positions, self.start_pos, bomb_pos)
        issues.extend(barrier_issues)
        
        return issues
    
    def _find_shortest_safe_path(self, start: Tuple[int, int], target: Tuple[int, int], 
                                flower_positions: set) -> int:
        """
        Find shortest path from start to target avoiding flower positions.
        Returns path length or -1 if unreachable.
        """
        if start == target:
            return 0
        
        if start in flower_positions or target in flower_positions:
            return -1
        
        # BFS for shortest path
        queue = deque([(start, 0)])
        visited = {start}
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        
        while queue:
            (x, y), dist = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                
                # Check if already visited
                if (nx, ny) in visited:
                    continue
                
                # Check if it's a flower (deadly)
                if (nx, ny) in flower_positions:
                    continue
                
                # Check if we reached target
                if (nx, ny) == target:
                    return dist + 1
                
                # Add to queue
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        return -1  # Unreachable
    
    def _check_flower_barriers(self, flower_positions: set, start: Tuple[int, int], 
                             target: Tuple[int, int]) -> List[str]:
        """Check for problematic flower arrangements that create impassable barriers."""
        issues = []
        
        # Check for complete walls that might block access
        # This is a simplified check - in a fully robust validator, you'd want more sophisticated analysis
        
        # Check if flowers form a complete horizontal barrier
        for y in range(self.grid_size):
            flowers_in_row = sum(1 for x in range(self.grid_size) if (x, y) in flower_positions)
            if flowers_in_row >= self.grid_size - 1:  # Nearly complete row
                # Check if this separates start and target
                if (start[1] < y < target[1]) or (target[1] < y < start[1]):
                    issues.append(f"WARNING: Nearly complete flower barrier at row {y} may block path")
        
        # Check for complete vertical barriers
        for x in range(self.grid_size):
            flowers_in_col = sum(1 for y in range(self.grid_size) if (x, y) in flower_positions)
            if flowers_in_col >= self.grid_size - 1:  # Nearly complete column
                # Check if this separates start and target
                if (start[0] < x < target[0]) or (target[0] < x < start[0]):
                    issues.append(f"WARNING: Nearly complete flower barrier at column {x} may block path")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """
        Validate that the reward structure promotes good problem-solving behavior.
        Note: Most reward validation is at environment level, but we can check for level-specific issues.
        """
        issues = []
        
        # The reward structure is defined in the environment code, not the level file
        # But we can validate that the level supports proper reward distribution
        
        # Check that there's exactly one bomb (one treasure to find)
        if "grid" in world_state and "icons" in world_state["grid"]:
            icons = world_state["grid"]["icons"]
            bomb_count = sum(1 for icon in icons.values() if icon == "bomb")
            
            if bomb_count == 0:
                issues.append("REWARD ISSUE: No treasure (bomb) to find - no positive reward possible")
            elif bomb_count > 1:
                issues.append("REWARD ISSUE: Multiple treasures may lead to unexpected reward behavior")
        
        # The binary reward structure (1.0 for treasure, 0 otherwise) is well-designed:
        # - Target Achievement: High reward (1.0) for main objective
        # - No action grinding: No rewards for individual moves
        # - No exploration loops: No rewards for revealing tiles
        # - Efficiency incentive: Longer solutions get no extra penalty, encouraging exploration
        
        return issues

def validate_generated_levels(levels_directory: str = "./levels/") -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate all generated levels in the specified directory.
    Returns dict mapping world_id -> (is_valid, issues)
    """
    validator = TreasureHuntValidator()
    results = {}
    
    if not os.path.exists(levels_directory):
        return {"ERROR": (False, [f"Levels directory not found: {levels_directory}"])}
    
    for filename in os.listdir(levels_directory):
        if filename.endswith('.yaml'):
            world_id = filename[:-5]  # Remove .yaml extension
            file_path = os.path.join(levels_directory, filename)
            
            is_valid, issues = validator.validate_level(file_path)
            results[world_id] = (is_valid, issues)
    
    return results

# Example usage function
def main():
    """Example validation run"""
    results = validate_generated_levels()
    
    print("=== TREASURE HUNT LEVEL VALIDATION RESULTS ===")
    for world_id, (is_valid, issues) in results.items():
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{world_id}: {status}")
        
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        print()

if __name__ == "__main__":
    main()
