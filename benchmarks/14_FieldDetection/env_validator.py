from typing import Dict, Any, List, Tuple, Optional
import yaml
from collections import deque
from copy import deepcopy

class EMEnvironmentValidator:
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_level(self, level_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Main validation function for electromagnetic anomaly levels.
        Returns: (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        try:
            with open(level_path, 'r') as f:
                level_data = yaml.safe_load(f)
        except Exception as e:
            self.validation_errors.append(f"Failed to load level file: {e}")
            return False, self.validation_errors, self.validation_warnings
        
        # Core validation checks
        self._validate_basic_structure(level_data)
        self._validate_grid_consistency(level_data)
        self._validate_connectivity(level_data)
        self._validate_solvability(level_data)
        self._validate_em_field_calculation(level_data)
        self._validate_reward_structure(level_data)
        self._validate_termination_conditions(level_data)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_basic_structure(self, level_data: Dict[str, Any]):
        """Validate basic level structure and required fields."""
        required_fields = [
            'globals', 'agent', 'grid', 'walls', 
            'vulnerability_node', 'em_field', 'steps_remaining'
        ]
        
        for field in required_fields:
            if field not in level_data:
                self.validation_errors.append(f"Missing required field: {field}")
        
        # Check agent structure
        if 'agent' in level_data:
            if 'pos' not in level_data['agent']:
                self.validation_errors.append("Agent missing position")
            if 'facing' not in level_data['agent']:
                self.validation_errors.append("Agent missing facing direction")
            elif level_data['agent']['facing'] not in ['north', 'south', 'east', 'west']:
                self.validation_errors.append(f"Invalid agent facing: {level_data['agent']['facing']}")
        
        # Check grid structure
        if 'grid' in level_data:
            if 'size' not in level_data['grid']:
                self.validation_errors.append("Grid missing size")
            elif len(level_data['grid']['size']) != 2:
                self.validation_errors.append("Grid size must be [width, height]")
        
        # Check vulnerability node
        if 'vulnerability_node' in level_data:
            if 'pos' not in level_data['vulnerability_node']:
                self.validation_errors.append("Vulnerability node missing position")
    
    def _validate_grid_consistency(self, level_data: Dict[str, Any]):
        """Validate grid dimensions and coordinate consistency."""
        if 'grid' not in level_data or 'size' not in level_data['grid']:
            return  # Already flagged in basic structure
        
        grid_width, grid_height = level_data['grid']['size']
        
        if grid_width != 15 or grid_height != 15:
            self.validation_errors.append(f"Grid must be 15x15, got {grid_width}x{grid_height}")
        
        # Validate agent position
        if 'agent' in level_data and 'pos' in level_data['agent']:
            agent_x, agent_y = level_data['agent']['pos']
            if not (0 <= agent_x < grid_width and 0 <= agent_y < grid_height):
                self.validation_errors.append(f"Agent position {level_data['agent']['pos']} outside grid bounds")
        
        # Validate vulnerability node position
        if 'vulnerability_node' in level_data and 'pos' in level_data['vulnerability_node']:
            node_x, node_y = level_data['vulnerability_node']['pos']
            if not (0 <= node_x < grid_width and 0 <= node_y < grid_height):
                self.validation_errors.append(f"Vulnerability node position {level_data['vulnerability_node']['pos']} outside grid bounds")
        
        # Validate wall positions
        for wall in level_data.get('walls', []):
            if isinstance(wall, list) and len(wall) == 2:
                wall_x, wall_y = wall
                if not (0 <= wall_x < grid_width and 0 <= wall_y < grid_height):
                    self.validation_errors.append(f"Wall position {wall} outside grid bounds")
        
        # Validate EM field dimensions
        if 'em_field' in level_data and 'values' in level_data['em_field']:
            field_values = level_data['em_field']['values']
            if len(field_values) != grid_height:
                self.validation_errors.append(f"EM field height {len(field_values)} doesn't match grid height {grid_height}")
            for i, row in enumerate(field_values):
                if len(row) != grid_width:
                    self.validation_errors.append(f"EM field row {i} width {len(row)} doesn't match grid width {grid_width}")
    
    def _validate_connectivity(self, level_data: Dict[str, Any]):
        """Validate that all free tiles form a connected component."""
        if 'grid' not in level_data or 'walls' not in level_data:
            return
        
        grid_width, grid_height = level_data['grid']['size']
        walls = set()
        
        for wall in level_data['walls']:
            if isinstance(wall, list) and len(wall) == 2:
                walls.add(tuple(wall))
        
        # Find all free tiles
        free_tiles = set()
        for x in range(grid_width):
            for y in range(grid_height):
                if (x, y) not in walls:
                    free_tiles.add((x, y))
        
        if not free_tiles:
            self.validation_errors.append("No free tiles available")
            return
        
        # Check connectivity using BFS
        start = next(iter(free_tiles))
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in free_tiles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        if len(visited) != len(free_tiles):
            self.validation_errors.append(f"Grid not fully connected: {len(visited)}/{len(free_tiles)} tiles reachable")
    
    def _validate_solvability(self, level_data: Dict[str, Any]):
        """
        Critical solvability check: Ensure the vulnerability node is reachable within 30 steps.
        """
        if any(field not in level_data for field in ['agent', 'vulnerability_node', 'walls', 'grid']):
            return  # Dependencies not available
        
        agent_pos = tuple(level_data['agent']['pos'])
        node_pos = tuple(level_data['vulnerability_node']['pos'])
        grid_width, grid_height = level_data['grid']['size']
        max_steps = level_data.get('steps_remaining', 30)
        
        walls = set()
        for wall in level_data['walls']:
            if isinstance(wall, list) and len(wall) == 2:
                walls.add(tuple(wall))
        
        # Check if agent starts on a wall (impossible scenario)
        if agent_pos in walls:
            self.validation_errors.append("Agent starts on a wall tile")
            return
        
        # Check if vulnerability node is on a wall (impossible scenario)
        if node_pos in walls:
            self.validation_errors.append("Vulnerability node is on a wall tile")
            return
        
        # Find shortest path from agent to any tile within Manhattan distance 1 of node
        target_tiles = set()
        node_x, node_y = node_pos
        
        # Add node position and all adjacent positions as valid targets
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) + abs(dy) <= 1:  # Manhattan distance <= 1
                    tx, ty = node_x + dx, node_y + dy
                    if (0 <= tx < grid_width and 0 <= ty < grid_height and 
                        (tx, ty) not in walls):
                        target_tiles.add((tx, ty))
        
        if not target_tiles:
            self.validation_errors.append("No valid tiles within marking range of vulnerability node")
            return
        
        # BFS to find shortest path to any target tile
        queue = deque([(agent_pos, 0)])  # (position, steps)
        visited = set([agent_pos])
        min_steps_to_target = float('inf')
        
        while queue:
            (x, y), steps = queue.popleft()
            
            if (x, y) in target_tiles:
                min_steps_to_target = min(min_steps_to_target, steps)
                continue
            
            if steps >= max_steps:  # Don't explore beyond step limit
                continue
            
            # Explore adjacent tiles
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_width and 0 <= ny < grid_height and 
                    (nx, ny) not in walls and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), steps + 1))
        
        if min_steps_to_target == float('inf'):
            self.validation_errors.append("Vulnerability node is unreachable from agent starting position")
        elif min_steps_to_target >= max_steps:
            self.validation_errors.append(f"Vulnerability node requires {min_steps_to_target} steps but only {max_steps} available")
        elif min_steps_to_target > max_steps * 0.8:
            self.validation_warnings.append(f"Vulnerability node requires {min_steps_to_target}/{max_steps} steps - very tight timing")
    
    def _validate_em_field_calculation(self, level_data: Dict[str, Any]):
        """Validate electromagnetic field values are calculated correctly."""
        if any(field not in level_data for field in ['vulnerability_node', 'walls', 'grid', 'em_field']):
            return
        
        node_pos = tuple(level_data['vulnerability_node']['pos'])
        grid_width, grid_height = level_data['grid']['size']
        walls = set(tuple(wall) for wall in level_data['walls'] if isinstance(wall, list))
        actual_field = level_data['em_field']['values']
        
        # Recalculate expected field values
        expected_field = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # BFS to calculate field with Faraday shielding
        visited = set()
        queue = deque([(node_pos, 3)])
        
        while queue:
            (x, y), strength = queue.popleft()
            
            if (x, y) in visited or strength <= 0:
                continue
            if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
                continue
            if (x, y) in walls:
                continue
            
            visited.add((x, y))
            expected_field[y][x] = strength
            
            # Propagate to adjacent tiles
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and (nx, ny) not in walls:
                    queue.append(((nx, ny), strength - 1))
        
        # Compare actual vs expected
        field_errors = 0
        for y in range(grid_height):
            for x in range(grid_width):
                if actual_field[y][x] != expected_field[y][x]:
                    field_errors += 1
        
        if field_errors > 0:
            self.validation_errors.append(f"EM field calculation incorrect at {field_errors} positions")
        
        # Validate field values are in valid range [0, 3]
        for y in range(grid_height):
            for x in range(grid_width):
                value = actual_field[y][x]
                if not isinstance(value, int) or value < 0 or value > 3:
                    self.validation_errors.append(f"Invalid EM field value {value} at position ({x}, {y})")
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]):
        """Validate reward structure prevents exploitation and encourages problem-solving."""
        # Check that only mark action within range gives positive reward
        # This is primarily structural validation since reward logic is in the environment
        
        if 'vulnerability_node' not in level_data:
            return
        
        # Validate that the binary reward system is correctly set up
        max_steps = level_data.get('steps_remaining', 30)
        
        # Check for potential reward exploitation scenarios
        if max_steps > 100:
            self.validation_warnings.append(f"Very high step limit ({max_steps}) might allow reward exploitation")
        
        # Ensure vulnerability node position allows for successful completion
        node_pos = level_data['vulnerability_node']['pos']
        grid_width, grid_height = level_data['grid']['size']
        
        # Check if node is in corner/edge where marking might be difficult
        node_x, node_y = node_pos
        walls = set(tuple(wall) for wall in level_data['walls'] if isinstance(wall, list))
        
        # Count accessible tiles around the node for marking
        accessible_marking_positions = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) + abs(dy) <= 1:
                    tx, ty = node_x + dx, node_y + dy
                    if (0 <= tx < grid_width and 0 <= ty < grid_height and 
                        (tx, ty) not in walls):
                        accessible_marking_positions += 1
        
        if accessible_marking_positions == 0:
            self.validation_errors.append("No accessible positions for marking vulnerability node")
        elif accessible_marking_positions == 1:
            self.validation_warnings.append("Only one accessible position for marking - very constrained")
    
    def _validate_termination_conditions(self, level_data: Dict[str, Any]):
        """Validate termination conditions are properly set up."""
        max_steps = level_data.get('steps_remaining', 30)
        
        if max_steps != 30:
            self.validation_warnings.append(f"Non-standard step limit: {max_steps} (expected 30)")
        
        if max_steps <= 0:
            self.validation_errors.append(f"Invalid step limit: {max_steps}")
        
        # Check global max_steps consistency if present
        if 'globals' in level_data and 'max_steps' in level_data['globals']:
            global_max = level_data['globals']['max_steps']
            if global_max != max_steps:
                self.validation_warnings.append(f"Inconsistent step limits: globals.max_steps={global_max}, steps_remaining={max_steps}")

def validate_level_file(level_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a single level file.
    Returns: (is_valid, errors, warnings)
    """
    validator = EMEnvironmentValidator()
    return validator.validate_level(level_path)

def validate_all_levels(levels_directory: str) -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """
    Validate all level files in a directory.
    Returns: Dict mapping level_name -> (is_valid, errors, warnings)
    """
    import os
    import glob
    
    results = {}
    validator = EMEnvironmentValidator()
    
    level_files = glob.glob(os.path.join(levels_directory, "*.yaml"))
    
    for level_file in level_files:
        level_name = os.path.basename(level_file)
        results[level_name] = validator.validate_level(level_file)
    
    return results