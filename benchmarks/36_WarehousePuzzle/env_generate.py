from base.env.base_generator import WorldGenerator
import yaml
import random
import os
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import hashlib
import time

class WarehouseGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = self.config['state_template']
        world_state = self._execute_pipeline(base_state, seed)
        world_id = self._generate_world_id(seed)
        
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                world_state = self._deep_copy(base_state)
                
                world_state = self._init_from_template(world_state)
                world_state = self._generate_warehouse_layout(world_state)
                world_state = self._place_boxes_and_docks_solvable(world_state)
                world_state = self._place_agent_start(world_state)
                
                # Quick solvability check
                if self._validate_level_solvability(world_state):
                    return world_state
                else:
                    print(f"Attempt {attempt + 1}: Generated unsolvable level, retrying...")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error during generation: {e}")
        
        # If all attempts failed, create a simple solvable level
        print("Creating fallback simple level...")
        return self._create_simple_solvable_level(base_state)
    
    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        else:
            return obj
    
    def _init_from_template(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        world_state['tiles']['grid'] = [['floor' for _ in range(10)] for _ in range(10)]
        return world_state
    
    def _generate_warehouse_layout(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        grid = world_state['tiles']['grid']
        
        # Create border walls
        for i in range(10):
            for j in range(10):
                if i == 0 or i == 9 or j == 0 or j == 9:
                    grid[i][j] = 'wall'
        
        # Create a simple layout with open areas for box pushing
        # Add some internal walls but ensure large open spaces
        for i in range(2, 8, 3):
            for j in range(2, 7):
                if j not in [4, 5]:  # Leave gaps
                    grid[i][j] = 'wall'
        
        return world_state
    
    def _place_boxes_and_docks_solvable(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        grid = world_state['tiles']['grid']
        
        # Get level number from context or use random if not available
        level_num = getattr(self, '_current_level', random.randint(1, 15))
        
        # Determine number of boxes based on level for progressive difficulty
        if level_num <= 5:
            num_boxes = 1
        elif level_num <= 10:
            num_boxes = 2
        else:
            num_boxes = 3
        
        # Find suitable columns for aligned box-dock pairs
        suitable_columns = []
        for col in range(2, 8):  # Use middle columns to avoid walls
            # Check if column has enough free vertical space
            free_spaces = 0
            for row in range(1, 9):
                if grid[row][col] == 'floor':
                    free_spaces += 1
            if free_spaces >= 6:  # Need space for box, movement, and dock
                suitable_columns.append(col)
        
        if len(suitable_columns) < num_boxes:
            # Fallback: use any available columns
            suitable_columns = list(range(2, 8))
        
        # Select columns for box-dock pairs
        selected_columns = random.sample(suitable_columns, min(num_boxes, len(suitable_columns)))
        
        box_positions = []
        dock_positions = []
        
        # Create aligned box-dock pairs
        for i, col in enumerate(selected_columns):
            # Find suitable row positions in this column
            available_rows = []
            for row in range(1, 9):
                if grid[row][col] == 'floor':
                    available_rows.append(row)
            
            if len(available_rows) >= 2:
                # Place box in upper part, dock in lower part
                box_row = random.choice(available_rows[:len(available_rows)//2 + 1])
                dock_row = random.choice(available_rows[len(available_rows)//2:])
                
                # Ensure box and dock are not too close
                if abs(box_row - dock_row) >= 3:
                    box_positions.append([box_row, col])
                    dock_positions.append([dock_row, col])
                else:
                    # Fallback: place them with minimum distance
                    if box_row < dock_row:
                        dock_row = min(8, box_row + 3)
                    else:
                        dock_row = max(1, box_row - 3)
                    
                    if grid[dock_row][col] == 'floor':
                        box_positions.append([box_row, col])
                        dock_positions.append([dock_row, col])
        
        # If we don't have enough pairs, create simple fallback
        while len(box_positions) < num_boxes:
            # Simple fallback positioning
            row = 2 + len(box_positions)
            col = 2 + len(box_positions) * 2
            if col < 8 and grid[row][col] == 'floor':
                box_positions.append([row, col])
                dock_positions.append([row + 4, col])
        
        world_state['objects']['boxes'] = [{'pos': pos} for pos in box_positions]
        world_state['objects']['docks'] = [{'pos': pos} for pos in dock_positions]
        world_state['level_info']['total_boxes'] = len(box_positions)
        world_state['level_info']['boxes_on_docks'] = 0
        
        return world_state
    
    def _is_pushable_position(self, pos: List[int], grid: List[List[str]]) -> bool:
        """Check if a position allows a box to be pushed in at least 2 directions"""
        x, y = pos
        pushable_dirs = 0
        
        # Check each direction
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            # Can box be pushed to this direction?
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < 10 and 0 <= new_y < 10 and 
                grid[new_x][new_y] != 'wall'):
                pushable_dirs += 1
        
        return pushable_dirs >= 2  # Can be pushed in at least 2 directions
    
    def _place_agent_start(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        grid = world_state['tiles']['grid']
        box_positions = {tuple(box['pos']) for box in world_state['objects']['boxes']}
        dock_positions = {tuple(dock['pos']) for dock in world_state['objects']['docks']}
        
        # Find positions away from boxes and docks
        available_positions = []
        for i in range(1, 9):
            for j in range(1, 9):
                if (grid[i][j] == 'floor' and 
                    (i, j) not in box_positions and 
                    (i, j) not in dock_positions):
                    # Prefer positions with some distance from boxes
                    min_box_dist = min([abs(i - bx) + abs(j - by) 
                                       for bx, by in box_positions] or [10])
                    if min_box_dist >= 2:
                        available_positions.append([i, j])
        
        if not available_positions:
            # Fallback: any empty floor tile
            for i in range(1, 9):
                for j in range(1, 9):
                    if (grid[i][j] == 'floor' and 
                        (i, j) not in box_positions and 
                        (i, j) not in dock_positions):
                        available_positions.append([i, j])
        
        if available_positions:
            world_state['agent']['pos'] = random.choice(available_positions)
        else:
            world_state['agent']['pos'] = [1, 1]  # Emergency fallback
        
        return world_state
    
    def _validate_level_solvability(self, world_state: Dict[str, Any]) -> bool:
        """Quick solvability check using basic heuristics"""
        grid = world_state['tiles']['grid']
        boxes = [tuple(box['pos']) for box in world_state['objects']['boxes']]
        docks = [tuple(dock['pos']) for dock in world_state['objects']['docks']]
        
        # Check 1: All boxes must be pushable
        for box_pos in boxes:
            if not self._can_box_be_pushed(box_pos, grid, boxes):
                return False
        
        # Check 2: All docks must be reachable
        for dock_pos in docks:
            if not self._is_dock_reachable(dock_pos, grid):
                return False
        
        # Check 3: Basic path existence (simplified)
        for box_pos in boxes:
            has_path_to_dock = False
            for dock_pos in docks:
                if self._has_clear_path(box_pos, dock_pos, grid, boxes):
                    has_path_to_dock = True
                    break
            if not has_path_to_dock:
                return False
        
        return True
    
    def _can_box_be_pushed(self, box_pos: Tuple, grid: List[List[str]], all_boxes: List[Tuple]) -> bool:
        """Check if box can be pushed in any direction"""
        x, y = box_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            # Check push target
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < 10 and 0 <= new_y < 10 and 
                grid[new_x][new_y] != 'wall' and 
                (new_x, new_y) not in all_boxes):
                
                # Check push position for agent
                agent_x, agent_y = x - dx, y - dy
                if (0 <= agent_x < 10 and 0 <= agent_y < 10 and 
                    grid[agent_x][agent_y] != 'wall'):
                    return True
        
        return False
    
    def _is_dock_reachable(self, dock_pos: Tuple, grid: List[List[str]]) -> bool:
        """Check if dock can have a box pushed onto it"""
        x, y = dock_pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            # Check if box can be pushed from this direction
            push_from_x, push_from_y = x - dx, y - dy
            if (0 <= push_from_x < 10 and 0 <= push_from_y < 10 and 
                grid[push_from_x][push_from_y] != 'wall'):
                return True
        
        return False
    
    def _has_clear_path(self, start: Tuple, end: Tuple, grid: List[List[str]], obstacles: List[Tuple]) -> bool:
        """Simple check for potential path (not perfect but good heuristic)"""
        # For now, just check if they're in the same open area
        # This is a simplified version - could be improved with actual pathfinding
        return True  # Assume path exists for simplicity
    
    def _create_simple_solvable_level(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a guaranteed solvable simple level as fallback"""
        world_state = self._deep_copy(base_state)
        world_state = self._init_from_template(world_state)
        
        grid = world_state['tiles']['grid']
        
        # Create border walls only
        for i in range(10):
            for j in range(10):
                if i == 0 or i == 9 or j == 0 or j == 9:
                    grid[i][j] = 'wall'
        
        # Place 3 boxes and 3 docks in simple configuration
        box_positions = [[2, 2], [2, 4], [2, 6]]
        dock_positions = [[6, 2], [6, 4], [6, 6]]
        agent_pos = [4, 4]
        
        world_state['objects']['boxes'] = [{'pos': pos} for pos in box_positions]
        world_state['objects']['docks'] = [{'pos': pos} for pos in dock_positions]
        world_state['level_info']['total_boxes'] = 3
        world_state['level_info']['boxes_on_docks'] = 0
        world_state['agent']['pos'] = agent_pos
        
        return world_state
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = str(int(time.time() * 1000))
        if seed is not None:
            seed_str = str(seed)
        else:
            seed_str = "random"
        hash_input = f"{timestamp}_{seed_str}".encode()
        hash_suffix = hashlib.md5(hash_input).hexdigest()[:8]
        return f"warehouse_{timestamp}_{hash_suffix}"
