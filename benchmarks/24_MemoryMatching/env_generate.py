from base.env.base_generator import WorldGenerator
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import random
import yaml
import os
from collections import deque
from copy import deepcopy
import uuid

class ChaosSlideGenerator(WorldGenerator):
    """World generator for Chaos Slide Puzzle environment."""
    
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.state_template = config['state_template']
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        """Generate complete world instance and save to file."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        base_state = deepcopy(self.state_template)
        generated_state = self._execute_pipeline(base_state, seed)
        
        if save_path is None:
            save_dir = f"./levels/"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{world_id}.yaml")
        
        with open(save_path, 'w') as f:
            yaml.dump(generated_state, f)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """Execute the generation pipeline."""
        current_state = deepcopy(base_state)
        
        # Init from template - already done in base_state
        
        # Scramble board
        current_state = self._scramble_board(current_state, seed)
        
        # Validate starting state
        current_state = self._validate_starting_state(current_state)
        
        # Verify solvability
        current_state = self._verify_solvability(current_state)
        
        return current_state
    
    def _scramble_board(self, state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """Apply random valid moves to create starting position."""
        min_moves = 50
        max_moves = 100
        num_moves = random.randint(min_moves, max_moves)
        
        # Start with ordered state for scrambling
        board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
        moves = ['SLIDE_UP', 'SLIDE_DOWN', 'SLIDE_LEFT', 'SLIDE_RIGHT']
        
        for _ in range(num_moves):
            action = random.choice(moves)
            board = self._apply_move(board, action)
        
        state['board']['grid'] = board
        return state
    
    def _apply_move(self, board: List[List[int]], action: str) -> List[List[int]]:
        """Apply a single move to the board."""
        board = deepcopy(board)
        blank_pos = self._find_blank(board)
        
        if action == 'SLIDE_UP' and blank_pos[0] > 0:
            # Move blank up, tile below slides down
            board[blank_pos[0]][blank_pos[1]], board[blank_pos[0]-1][blank_pos[1]] = \
                board[blank_pos[0]-1][blank_pos[1]], board[blank_pos[0]][blank_pos[1]]
        elif action == 'SLIDE_DOWN' and blank_pos[0] < 2:
            # Move blank down, tile above slides up
            board[blank_pos[0]][blank_pos[1]], board[blank_pos[0]+1][blank_pos[1]] = \
                board[blank_pos[0]+1][blank_pos[1]], board[blank_pos[0]][blank_pos[1]]
        elif action == 'SLIDE_LEFT' and blank_pos[1] > 0:
            # Move blank left, tile on right slides left
            board[blank_pos[0]][blank_pos[1]], board[blank_pos[0]][blank_pos[1]-1] = \
                board[blank_pos[0]][blank_pos[1]-1], board[blank_pos[0]][blank_pos[1]]
        elif action == 'SLIDE_RIGHT' and blank_pos[1] < 2:
            # Move blank right, tile on left slides right
            board[blank_pos[0]][blank_pos[1]], board[blank_pos[0]][blank_pos[1]+1] = \
                board[blank_pos[0]][blank_pos[1]+1], board[blank_pos[0]][blank_pos[1]]
        
        return board
    
    def _find_blank(self, board: List[List[int]]) -> Tuple[int, int]:
        """Find position of blank space (0)."""
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return (i, j)
        return (0, 0)
    
    def _validate_starting_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure starting state is valid."""
        current_board = state['board']['grid']
        chaos_pattern = state['targets']['chaos_pattern']
        forbidden_pattern = state['targets']['forbidden_pattern']
        
        # Ensure we don't start at success or failure states
        while (current_board == chaos_pattern or current_board == forbidden_pattern):
            # Apply a few more random moves
            for _ in range(5):
                moves = ['SLIDE_UP', 'SLIDE_DOWN', 'SLIDE_LEFT', 'SLIDE_RIGHT']
                action = random.choice(moves)
                current_board = self._apply_move(current_board, action)
            state['board']['grid'] = current_board
        
        return state
    
    def _verify_solvability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify target is reachable within step limit using BFS."""
        start_board = state['board']['grid']
        target_board = state['targets']['chaos_pattern']
        max_steps = 30
        
        if self._bfs_solvable(start_board, target_board, max_steps):
            return state
        else:
            # If not solvable, generate a new scrambled state
            return self._scramble_board(state, None)
    
    def _bfs_solvable(self, start: List[List[int]], target: List[List[int]], max_depth: int) -> bool:
        """Use BFS to check if target is reachable within max_depth steps."""
        if start == target:
            return True
        
        queue = deque([(start, 0)])
        visited = set()
        visited.add(self._board_to_tuple(start))
        
        moves = ['SLIDE_UP', 'SLIDE_DOWN', 'SLIDE_LEFT', 'SLIDE_RIGHT']
        
        while queue:
            current_board, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for move in moves:
                next_board = self._apply_move(current_board, move)
                next_tuple = self._board_to_tuple(next_board)
                
                if next_tuple not in visited:
                    if next_board == target:
                        return True
                    
                    visited.add(next_tuple)
                    queue.append((next_board, depth + 1))
        
        return False
    
    def _board_to_tuple(self, board: List[List[int]]) -> Tuple:
        """Convert board to tuple for hashing."""
        return tuple(tuple(row) for row in board)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        """Generate unique world identifier."""
        if seed is not None:
            return f"world_{seed}_{uuid.uuid4().hex[:8]}"
        else:
            return f"world_{uuid.uuid4().hex[:12]}"