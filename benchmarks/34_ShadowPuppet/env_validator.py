from typing import Dict, Any, List, Tuple, Optional, Set
from copy import deepcopy
import yaml

class ShadowPuppetLevelValidator:
    def __init__(self):
        self.property_mappings = {
            'square': 'Heavy',
            'circle': 'Light', 
            'triangle': 'Bouncy',
            'cross': 'Sticky'
        }
        self.max_validation_steps = 40
        
    def validate_level(self, level_file: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated Shadow Puppet level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(level_file, 'r') as f:
                state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level file: {e}"]
            
        issues = []
        
        # 1. Basic structure validation
        structure_issues = self._validate_structure(state)
        issues.extend(structure_issues)
        
        # 2. Level solvability analysis
        solvability_issues = self._validate_solvability(state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_issues = self._validate_reward_structure(state)
        issues.extend(reward_issues)
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate basic level structure and constraints"""
        issues = []
        
        # Check required components exist
        required_keys = ['globals', 'shadow', 'objects']
        for key in required_keys:
            if key not in state:
                issues.append(f"Missing required key: {key}")
                return issues
        
        # Validate grid bounds
        grid_size = state['globals'].get('grid_size', [8, 8])
        if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
            issues.append("Invalid grid_size")
        
        # Validate goal area
        goal_area = state['globals'].get('goal_area')
        if not goal_area or len(goal_area) != 2:
            issues.append("Invalid goal_area format")
        else:
            for coord in goal_area:
                if (len(coord) != 2 or 
                    coord[0] < 0 or coord[0] >= grid_size[0] or
                    coord[1] < 0 or coord[1] >= grid_size[1]):
                    issues.append("Goal area coordinates out of bounds")
                    break
        
        # Validate objects
        target_count = 0
        occupied_positions = set()
        
        for obj in state.get('objects', []):
            if obj.get('is_target', False):
                target_count += 1
            
            pos = tuple(obj['position'])
            if pos in occupied_positions:
                issues.append(f"Multiple objects at same position: {pos}")
            occupied_positions.add(pos)
            
            # Check position bounds
            if (obj['position'][0] < 0 or obj['position'][0] >= grid_size[0] or
                obj['position'][1] < 0 or obj['position'][1] >= grid_size[1]):
                issues.append(f"Object {obj.get('id', 'unknown')} position out of bounds")
        
        if target_count != 1:
            issues.append(f"Must have exactly 1 target object, found {target_count}")
        
        return issues
    
    def _validate_solvability(self, state: Dict[str, Any]) -> List[str]:
        """Critical solvability analysis using BFS pathfinding"""
        issues = []
        
        # Find target object
        target_obj = None
        for obj in state['objects']:
            if obj.get('is_target', False):
                target_obj = obj
                break
        
        if not target_obj:
            issues.append("No target object found")
            return issues
        
        # Extract goal area bounds
        goal_area = state['globals']['goal_area']
        goal_positions = set()
        for x in range(goal_area[0][0], goal_area[1][0] + 1):
            for y in range(goal_area[0][1], goal_area[1][1] + 1):
                goal_positions.add((x, y))
        
        # Check if target starts in goal (invalid - too easy)
        target_pos = tuple(target_obj['position'])
        if target_pos in goal_positions:
            issues.append("Target object starts in goal area - level too trivial")
        
        # Perform solvability analysis using state space search
        is_solvable = self._bfs_solvability_check(state, goal_positions)
        if not is_solvable:
            issues.append("Level appears unsolvable within step limit through shadow manipulation")
        
        # Check minimum difficulty - target should not be immediately adjacent to goal
        min_distance = min(abs(target_pos[0] - gx) + abs(target_pos[1] - gy) 
                          for gx, gy in goal_positions)
        if min_distance < 2:
            issues.append("Target too close to goal - insufficient challenge (min distance: 2)")
        
        # Validate that level requires actual shadow manipulation (not just movement)
        if self._can_solve_without_shadows(state, goal_positions):
            issues.append("Level solvable without shadow transformations - lacks core mechanic usage")
        
        return issues
    
    def _bfs_solvability_check(self, initial_state: Dict[str, Any], goal_positions: Set[Tuple[int, int]]) -> bool:
        """
        BFS search to verify target can reach goal through shadow transformations.
        Simplified but comprehensive solvability check.
        """
        from collections import deque
        
        # Create initial search state
        def state_key(objects, shadow_pos, shadow_shape, shadow_active):
            # Create hashable state representation
            obj_positions = tuple(sorted((obj['id'], tuple(obj['position']), obj['property']) 
                                       for obj in objects))
            return (obj_positions, tuple(shadow_pos), shadow_shape, shadow_active)
        
        initial_key = state_key(initial_state['objects'], 
                               initial_state['shadow']['position'],
                               initial_state['shadow']['shape'], 
                               initial_state['shadow']['active'])
        
        visited = {initial_key}
        queue = deque([(deepcopy(initial_state), 0)])  # (state, steps)
        
        grid_size = initial_state['globals']['grid_size']
        actions = [
            # Shadow movements
            ('move_shadow', {'dx': 1, 'dy': 0}),
            ('move_shadow', {'dx': -1, 'dy': 0}),
            ('move_shadow', {'dx': 0, 'dy': 1}),
            ('move_shadow', {'dx': 0, 'dy': -1}),
            # Shape changes
            ('cycle_shape', {'shape': 'square'}),
            ('cycle_shape', {'shape': 'circle'}),
            ('cycle_shape', {'shape': 'triangle'}),
            ('cycle_shape', {'shape': 'cross'}),
            # Toggle shadow
            ('toggle_shadow', {}),
            # Wind pulse
            ('wind_pulse', {}),
        ]
        
        while queue and len(queue) < 1000:  # Limit search to prevent infinite loops
            current_state, steps = queue.popleft()
            
            if steps >= self.max_validation_steps:
                continue
            
            # Check if target is in goal
            target_pos = None
            for obj in current_state['objects']:
                if obj.get('is_target', False):
                    target_pos = tuple(obj['position'])
                    break
            
            if target_pos and target_pos in goal_positions:
                return True
            
            # Try all possible actions
            for action_name, action_params in actions:
                new_state = self._simulate_action(deepcopy(current_state), action_name, action_params)
                if new_state is None:
                    continue
                
                new_key = state_key(new_state['objects'],
                                  new_state['shadow']['position'],
                                  new_state['shadow']['shape'],
                                  new_state['shadow']['active'])
                
                if new_key not in visited:
                    visited.add(new_key)
                    queue.append((new_state, steps + 1))
        
        return False
    
    def _simulate_action(self, state: Dict[str, Any], action_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate a single action and return resulting state"""
        try:
            grid_size = state['globals']['grid_size']
            
            if action_name == 'move_shadow':
                dx, dy = params.get('dx', 0), params.get('dy', 0)
                shadow_pos = state['shadow']['position']
                new_x = max(0, min(grid_size[0] - 1, shadow_pos[0] + dx))
                new_y = max(0, min(grid_size[1] - 1, shadow_pos[1] + dy))
                state['shadow']['position'] = [new_x, new_y]
                
            elif action_name == 'cycle_shape':
                shape = params.get('shape', 'square')
                if shape in ['square', 'circle', 'triangle', 'cross']:
                    state['shadow']['shape'] = shape
                    
            elif action_name == 'toggle_shadow':
                state['shadow']['active'] = not state['shadow']['active']
                
            elif action_name == 'wind_pulse':
                if state['shadow']['active']:
                    shadow_pos = state['shadow']['position']
                    for obj in state['objects']:
                        if obj['property'] == 'Light':
                            obj_pos = obj['position']
                            dx = obj_pos[0] - shadow_pos[0]
                            dy = obj_pos[1] - shadow_pos[1]
                            
                            if dx == 0 and dy == 0:
                                obj['velocity'] = [1, 0]  # Default push
                            else:
                                if abs(dx) >= abs(dy):
                                    obj['velocity'] = [1 if dx > 0 else -1, 0]
                                else:
                                    obj['velocity'] = [0, 1 if dy > 0 else -1]
            
            # Apply shadow transformations
            self._apply_shadow_transformations(state)
            
            # Physics step
            self._physics_simulation(state)
            
            return state
            
        except Exception:
            return None
    
    def _apply_shadow_transformations(self, state: Dict[str, Any]):
        """Apply shadow transformations to overlapping objects"""
        if not state['shadow']['active']:
            return
        
        shadow_pos = state['shadow']['position']
        shadow_shape = state['shadow']['shape']
        
        for obj in state['objects']:
            if obj['position'] == shadow_pos:
                obj['property'] = self.property_mappings[shadow_shape]
    
    def _physics_simulation(self, state: Dict[str, Any]):
        """Simplified physics simulation for validation"""
        grid_size = state['globals']['grid_size']
        
        # Move objects based on velocity
        for obj in state['objects']:
            if obj['velocity'] != [0, 0]:
                new_x = max(0, min(grid_size[0] - 1, obj['position'][0] + obj['velocity'][0]))
                new_y = max(0, min(grid_size[1] - 1, obj['position'][1] + obj['velocity'][1]))
                obj['position'] = [new_x, new_y]
                
                # Stop non-bouncy objects
                if obj['property'] != 'Bouncy':
                    obj['velocity'] = [0, 0]
                # Bounce off walls for bouncy objects
                elif (new_x == 0 or new_x == grid_size[0] - 1 or 
                      new_y == 0 or new_y == grid_size[1] - 1):
                    if new_x == 0 or new_x == grid_size[0] - 1:
                        obj['velocity'][0] = -obj['velocity'][0]
                    if new_y == 0 or new_y == grid_size[1] - 1:
                        obj['velocity'][1] = -obj['velocity'][1]
    
    def _can_solve_without_shadows(self, state: Dict[str, Any], goal_positions: Set[Tuple[int, int]]) -> bool:
        """Check if level can be solved without using shadow transformations"""
        # Simple check: if target is Light and can be pushed directly to goal with wind
        target_obj = None
        for obj in state['objects']:
            if obj.get('is_target', False):
                target_obj = obj
                break
        
        if not target_obj:
            return False
        
        # If target is already Light and path exists via wind pushes
        if target_obj['property'] == 'Light':
            target_pos = target_obj['position']
            # Check if target can reach goal through direct wind pushes
            for goal_pos in goal_positions:
                # Simple Manhattan distance check - if very close and Light, might be too easy
                distance = abs(target_pos[0] - goal_pos[0]) + abs(target_pos[1] - goal_pos[1])
                if distance <= 3:  # Can likely be pushed directly
                    return True
        
        return False
    
    def _validate_reward_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate that reward structure encourages proper problem solving"""
        issues = []
        
        # The Shadow Puppet environment uses binary rewards (1 for success, 0 otherwise)
        # This is actually good design - no action grinding possible
        
        # Check that success condition is properly defined
        goal_area = state['globals'].get('goal_area')
        if not goal_area:
            issues.append("No goal area defined - success condition unclear")
        
        # Verify step limit is reasonable for complexity
        max_steps = state['globals'].get('max_steps', 40)
        if max_steps > 100:
            issues.append("Step limit too high - may encourage inefficient exploration")
        elif max_steps < 10:
            issues.append("Step limit too low - may make solvable levels impossible")
        
        # Check that binary reward system is maintained (no intermediate rewards that could be gamed)
        # This environment's design already prevents action grinding through binary rewards
        
        return issues

# Usage function
def validate_shadow_puppet_level(level_file: str) -> Tuple[bool, List[str]]:
    """
    Validate a Shadow Puppet level file.
    
    Args:
        level_file: Path to the YAML level file
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = ShadowPuppetLevelValidator()
    return validator.validate_level(level_file)

# Batch validation function
def validate_all_levels(levels_directory: str) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate all level files in a directory.
    
    Args:
        levels_directory: Path to directory containing level YAML files
        
    Returns:
        Dictionary mapping level filenames to (is_valid, issues) tuples
    """
    import os
    
    validator = ShadowPuppetLevelValidator()
    results = {}
    
    for filename in os.listdir(levels_directory):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            filepath = os.path.join(levels_directory, filename)
            try:
                is_valid, issues = validator.validate_level(filepath)
                results[filename] = (is_valid, issues)
            except Exception as e:
                results[filename] = (False, [f"Validation error: {e}"])
    
    return results
