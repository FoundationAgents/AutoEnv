from typing import Dict, Any, List, Tuple, Set
import numpy as np
from collections import deque
import yaml
import os

class ChaosSlideValidator:
    """Validator for Chaos Slide Puzzle environment levels."""
    
    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize validator with environment configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chaos_pattern = self.config['state_template']['targets']['chaos_pattern']
        self.forbidden_pattern = self.config['state_template']['targets']['forbidden_pattern']
        self.max_steps = self.config['state_template']['globals']['max_steps']
        
        # Track validation results
        self.validation_results = {
            'solvability_issues': [],
            'reward_issues': [],
            'constraint_issues': [],
            'passed_levels': 0,
            'failed_levels': 0
        }
    
    def validate_level(self, level_path: str) -> Dict[str, Any]:
        """Validate a single generated level."""
        try:
            with open(level_path, 'r') as f:
                level_state = yaml.safe_load(f)
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Failed to load level: {str(e)}"],
                'solvability': False,
                'reward_structure': False
            }
        
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_result = self._check_level_solvability(level_state)
        if not solvability_result['is_solvable']:
            issues.extend(solvability_result['blocking_issues'])
        
        # 2. REWARD STRUCTURE VALIDATION  
        reward_result = self._validate_reward_structure()
        if not reward_result['is_well_designed']:
            issues.extend(reward_result['reward_issues'])
        
        # 3. STATE CONSTRAINT VALIDATION
        constraint_result = self._validate_state_constraints(level_state)
        if not constraint_result['valid_constraints']:
            issues.extend(constraint_result['constraint_issues'])
        
        is_valid = len(issues) == 0
        
        if is_valid:
            self.validation_results['passed_levels'] += 1
        else:
            self.validation_results['failed_levels'] += 1
            self.validation_results['solvability_issues'].extend(
                solvability_result.get('blocking_issues', []))
            self.validation_results['reward_issues'].extend(
                reward_result.get('reward_issues', []))
            self.validation_results['constraint_issues'].extend(
                constraint_result.get('constraint_issues', []))
        
        return {
            'valid': is_valid,
            'issues': issues,
            'solvability': solvability_result['is_solvable'],
            'reward_structure': reward_result['is_well_designed'],
            'state_constraints': constraint_result['valid_constraints'],
            'solution_length': solvability_result.get('solution_length', -1)
        }
    
    def _check_level_solvability(self, level_state: Dict[str, Any]) -> Dict[str, Any]:
        """Critical check for puzzle solvability within step constraints."""
        initial_board = level_state['board']['grid']
        target_board = self.chaos_pattern
        forbidden_board = self.forbidden_pattern
        
        blocking_issues = []
        
        # 1. ACTION CONSTRAINT ANALYSIS
        action_analysis = self._analyze_action_constraints(initial_board)
        if not action_analysis['sufficient_actions']:
            blocking_issues.extend(action_analysis['issues'])
        
        # 2. TARGET REACHABILITY using BFS
        reachability = self._check_target_reachability(initial_board, target_board, self.max_steps)
        if not reachability['reachable']:
            blocking_issues.extend(reachability['issues'])
        
        # 3. FORBIDDEN STATE ANALYSIS
        forbidden_analysis = self._analyze_forbidden_state_risk(initial_board, forbidden_board)
        if forbidden_analysis['high_risk']:
            blocking_issues.extend(forbidden_analysis['warnings'])
        
        # 4. VALIDATE INITIAL STATE
        initial_state_check = self._validate_initial_state(initial_board, target_board, forbidden_board)
        if not initial_state_check['valid']:
            blocking_issues.extend(initial_state_check['issues'])
        
        return {
            'is_solvable': len(blocking_issues) == 0,
            'blocking_issues': blocking_issues,
            'solution_length': reachability.get('solution_length', -1),
            'forbidden_risk': forbidden_analysis['risk_level']
        }
    
    def _analyze_action_constraints(self, board: List[List[int]]) -> Dict[str, Any]:
        """Analyze if available actions have sufficient power to solve puzzle."""
        issues = []
        
        # Check if board represents valid 3x3 sliding puzzle
        flat_board = [item for row in board for item in row]
        expected_tiles = set(range(9))  # 0-8
        actual_tiles = set(flat_board)
        
        if expected_tiles != actual_tiles:
            issues.append(f"Invalid tile set: expected {expected_tiles}, got {actual_tiles}")
        
        if len(flat_board) != 9:
            issues.append(f"Invalid board size: expected 9 tiles, got {len(flat_board)}")
        
        # Check blank space exists and is unique
        blank_count = flat_board.count(0)
        if blank_count != 1:
            issues.append(f"Invalid blank space count: expected 1, got {blank_count}")
        
        return {
            'sufficient_actions': len(issues) == 0,
            'issues': issues
        }
    
    def _check_target_reachability(self, start: List[List[int]], target: List[List[int]], max_steps: int) -> Dict[str, Any]:
        """Use BFS to verify target state is reachable within step budget."""
        if self._boards_equal(start, target):
            return {
                'reachable': True,
                'solution_length': 0,
                'issues': []
            }
        
        # BFS with step counting
        queue = deque([(start, 0)])
        visited = set()
        visited.add(self._board_to_tuple(start))
        
        moves = ['SLIDE_UP', 'SLIDE_DOWN', 'SLIDE_LEFT', 'SLIDE_RIGHT']
        
        while queue:
            current_board, steps = queue.popleft()
            
            if steps >= max_steps:
                continue
            
            for move in moves:
                next_board = self._apply_move(current_board, move)
                if next_board is None:  # Invalid move
                    continue
                    
                next_tuple = self._board_to_tuple(next_board)
                
                if next_tuple not in visited:
                    if self._boards_equal(next_board, target):
                        return {
                            'reachable': True,
                            'solution_length': steps + 1,
                            'issues': []
                        }
                    
                    visited.add(next_tuple)
                    queue.append((next_board, steps + 1))
        
        return {
            'reachable': False,
            'solution_length': -1,
            'issues': [f"Target chaos pattern not reachable within {max_steps} steps from starting position"]
        }
    
    def _analyze_forbidden_state_risk(self, initial_board: List[List[int]], forbidden_board: List[List[int]]) -> Dict[str, Any]:
        """Analyze risk of accidentally reaching forbidden ordered state."""
        # Calculate minimum distance to forbidden state
        forbidden_distance = self._calculate_minimum_distance(initial_board, forbidden_board, 15)  # Check up to 15 steps
        
        risk_level = "low"
        warnings = []
        
        if forbidden_distance != -1 and forbidden_distance <= 5:
            risk_level = "high" 
            warnings.append(f"Forbidden ordered pattern reachable in {forbidden_distance} steps - high risk of accidental failure")
        elif forbidden_distance != -1 and forbidden_distance <= 10:
            risk_level = "medium"
            warnings.append(f"Forbidden pattern reachable in {forbidden_distance} steps - moderate risk")
        
        return {
            'high_risk': risk_level == "high",
            'risk_level': risk_level,
            'warnings': warnings,
            'forbidden_distance': forbidden_distance
        }
    
    def _validate_initial_state(self, initial: List[List[int]], target: List[List[int]], forbidden: List[List[int]]) -> Dict[str, Any]:
        """Validate initial state doesn't start at success/failure conditions."""
        issues = []
        
        if self._boards_equal(initial, target):
            issues.append("Initial state equals target chaos pattern - episode would end immediately")
        
        if self._boards_equal(initial, forbidden):
            issues.append("Initial state equals forbidden ordered pattern - episode would fail immediately")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_reward_structure(self) -> Dict[str, Any]:
        """Validate reward system prevents exploitation and incentivizes problem-solving."""
        reward_issues = []
        
        # Check reward configuration from config
        reward_events = self.config.get('reward', {}).get('events', [])
        
        # Analyze each reward event
        chaos_reward = None
        forbidden_reward = None
        step_reward = None
        
        for event in reward_events:
            if event['trigger'] == 'chaos_pattern_reached':
                chaos_reward = event['value']
            elif event['trigger'] == 'forbidden_pattern_reached':
                forbidden_reward = event['value']
            elif event['trigger'] == 'step_taken':
                step_reward = event['value']
        
        # GOAL-ORIENTED REWARDS validation
        if chaos_reward is None or chaos_reward <= 0:
            reward_issues.append("Missing or non-positive reward for achieving chaos pattern")
        elif chaos_reward < 10:  # Should be high value (15-20 suggested)
            reward_issues.append(f"Chaos pattern reward {chaos_reward} may be too low - recommend 15-20 points")
        
        # AVOID INCENTIVE MISALIGNMENT
        if step_reward is not None and step_reward > 0:
            if chaos_reward is not None and step_reward >= chaos_reward * 0.1:
                reward_issues.append(f"Step reward {step_reward} too high relative to goal reward {chaos_reward} - enables action grinding")
        
        if forbidden_reward is not None and forbidden_reward > 0:
            reward_issues.append("Forbidden pattern should not provide positive reward")
        
        # Check for reward loops (step rewards should be 0 to prevent exploitation)
        if step_reward is not None and step_reward > 0:
            reward_issues.append("Non-zero step rewards create exploitation opportunities for endless action sequences")
        
        # EFFICIENCY INCENTIVE check
        if chaos_reward is not None and step_reward is not None:
            if step_reward >= 0:  # Should be slightly negative to incentivize efficiency
                reward_issues.append("Consider small negative step rewards to encourage efficient solutions")
        
        return {
            'is_well_designed': len(reward_issues) == 0,
            'reward_issues': reward_issues
        }
    
    def _validate_state_constraints(self, level_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state meets environment constraints."""
        constraint_issues = []
        
        # Check steps_remaining initialization
        steps_remaining = level_state.get('agent', {}).get('steps_remaining')
        if steps_remaining != self.max_steps:
            constraint_issues.append(f"Initial steps_remaining {steps_remaining} doesn't match max_steps {self.max_steps}")
        
        # Check board dimensions
        board = level_state.get('board', {}).get('grid', [])
        if len(board) != 3 or any(len(row) != 3 for row in board):
            constraint_issues.append("Board must be 3x3 grid")
        
        # Check target patterns are preserved
        if level_state.get('targets', {}).get('chaos_pattern') != self.chaos_pattern:
            constraint_issues.append("Chaos pattern doesn't match expected target")
        
        if level_state.get('targets', {}).get('forbidden_pattern') != self.forbidden_pattern:
            constraint_issues.append("Forbidden pattern doesn't match expected pattern")
        
        return {
            'valid_constraints': len(constraint_issues) == 0,
            'constraint_issues': constraint_issues
        }
    
    def _apply_move(self, board: List[List[int]], action: str) -> List[List[int]]:
        """Apply action to board, return None if invalid move."""
        board = [row[:] for row in board]  # Deep copy
        blank_pos = self._find_blank(board)
        
        if blank_pos is None:
            return None
        
        row, col = blank_pos
        
        if action == 'SLIDE_UP' and row > 0:
            board[row][col], board[row-1][col] = board[row-1][col], board[row][col]
        elif action == 'SLIDE_DOWN' and row < 2:
            board[row][col], board[row+1][col] = board[row+1][col], board[row][col]
        elif action == 'SLIDE_LEFT' and col > 0:
            board[row][col], board[row][col-1] = board[row][col-1], board[row][col]
        elif action == 'SLIDE_RIGHT' and col < 2:
            board[row][col], board[row][col+1] = board[row][col+1], board[row][col]
        else:
            return None  # Invalid move
        
        return board
    
    def _find_blank(self, board: List[List[int]]) -> Tuple[int, int]:
        """Find blank space position."""
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 0:
                    return (i, j)
        return None
    
    def _boards_equal(self, board1: List[List[int]], board2: List[List[int]]) -> bool:
        """Check if two boards are equal."""
        return board1 == board2
    
    def _board_to_tuple(self, board: List[List[int]]) -> Tuple:
        """Convert board to tuple for hashing."""
        return tuple(tuple(row) for row in board)
    
    def _calculate_minimum_distance(self, start: List[List[int]], target: List[List[int]], max_depth: int) -> int:
        """Calculate minimum steps to reach target, return -1 if not reachable."""
        if self._boards_equal(start, target):
            return 0
        
        queue = deque([(start, 0)])
        visited = set()
        visited.add(self._board_to_tuple(start))
        
        moves = ['SLIDE_UP', 'SLIDE_DOWN', 'SLIDE_LEFT', 'SLIDE_RIGHT']
        
        while queue:
            current_board, steps = queue.popleft()
            
            if steps >= max_depth:
                continue
            
            for move in moves:
                next_board = self._apply_move(current_board, move)
                if next_board is None:
                    continue
                    
                if self._boards_equal(next_board, target):
                    return steps + 1
                
                next_tuple = self._board_to_tuple(next_board)
                if next_tuple not in visited:
                    visited.add(next_tuple)
                    queue.append((next_board, steps + 1))
        
        return -1
    
    def validate_levels_batch(self, levels_directory: str) -> Dict[str, Any]:
        """Validate all levels in a directory."""
        results = []
        
        for filename in os.listdir(levels_directory):
            if filename.endswith('.yaml'):
                level_path = os.path.join(levels_directory, filename)
                result = self.validate_level(level_path)
                result['filename'] = filename
                results.append(result)
        
        # Summary statistics
        total_levels = len(results)
        passed = sum(1 for r in results if r['valid'])
        failed = total_levels - passed
        
        summary = {
            'total_levels': total_levels,
            'passed_levels': passed,
            'failed_levels': failed,
            'pass_rate': passed / total_levels if total_levels > 0 else 0,
            'individual_results': results,
            'validation_summary': self.validation_results
        }
        
        return summary

def validate_chaos_slide_levels(levels_dir: str = "./levels/", config_path: str = "./config.yaml") -> Dict[str, Any]:
    """Main validation function for Chaos Slide Puzzle levels."""
    validator = ChaosSlideValidator(config_path)
    return validator.validate_levels_batch(levels_dir)