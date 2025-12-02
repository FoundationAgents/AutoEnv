from typing import Dict, Any, List, Tuple, Optional
import copy

class ConnectFourValidator:
    """
    Validator for Connect-Four environment levels to ensure solvability and proper reward structure.
    """
    
    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.max_steps = 40
        
    def validate_level(self, level_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Main validation function that checks level solvability and reward structure.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(level_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION  
        reward_issues = self._validate_reward_structure(level_state)
        issues.extend(reward_issues)
        
        # 3. BASIC STATE VALIDATION
        basic_issues = self._validate_basic_state(level_state)
        issues.extend(basic_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for impossible puzzles - ensures the agent can potentially win.
        """
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(level_state)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(level_state)
        issues.extend(reachability_issues)
        
        # COMMON IMPOSSIBLE PATTERNS
        pattern_issues = self._check_impossible_patterns(level_state)
        issues.extend(pattern_issues)
        
        return issues
    
    def _analyze_action_constraints(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Understand environment's fundamental limitations and action capabilities.
        """
        issues = []
        
        # Check if board grid exists and has correct dimensions
        if 'board' not in level_state or 'grid' not in level_state['board']:
            issues.append("SOLVABILITY: Missing board grid - agent cannot place disks")
            return issues
            
        grid = level_state['board']['grid']
        
        # Validate board dimensions
        if len(grid) != self.board_height:
            issues.append(f"SOLVABILITY: Invalid board height {len(grid)}, expected {self.board_height}")
            
        if len(grid) > 0 and len(grid[0]) != self.board_width:
            issues.append(f"SOLVABILITY: Invalid board width {len(grid[0])}, expected {self.board_width}")
            
        # Check if any columns are available for moves
        available_columns = self._get_available_columns(grid)
        if len(available_columns) == 0:
            issues.append("SOLVABILITY: No available columns - agent cannot make any moves")
            
        # Verify action space coverage
        if len(available_columns) < 3:  # Connect-4 needs reasonable move options
            issues.append("SOLVABILITY: Too few available columns - severely limited strategic options")
            
        return issues
    
    def _check_target_reachability(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Verify that winning state is actually achievable within step limits.
        """
        issues = []
        
        grid = level_state['board']['grid']
        max_steps = level_state.get('globals', {}).get('max_steps', self.max_steps)
        moves_made = level_state.get('game', {}).get('moves_made', 0)
        remaining_steps = max_steps - moves_made
        
        # Check if game is already over
        if level_state.get('game', {}).get('game_over', False):
            winner = level_state.get('game', {}).get('winner', 0)
            if winner == 2:  # Opponent already won
                issues.append("SOLVABILITY: Game already over - opponent has won, agent cannot achieve victory")
            elif winner == 1:  # Agent already won
                issues.append("SOLVABILITY: Game already over - agent has already won, no challenge remaining")
            return issues
        
        # Resource availability check - ensure agent can still make meaningful progress
        available_moves = self._count_available_moves(grid)
        if available_moves == 0:
            issues.append("SOLVABILITY: No moves available - board is completely full")
            
        # Step counting - verify minimum moves needed vs. remaining steps
        min_moves_to_win = self._estimate_minimum_moves_to_win(grid, player=1)
        if min_moves_to_win > remaining_steps:
            issues.append(f"SOLVABILITY: Insufficient steps - need at least {min_moves_to_win} moves but only {remaining_steps} remaining")
            
        # Check if opponent can force a win faster than agent
        opponent_min_moves = self._estimate_minimum_moves_to_win(grid, player=2)
        if opponent_min_moves == 1:  # Opponent has immediate winning move
            # Check if agent can also win in 1 move (simultaneous win condition)
            if min_moves_to_win > 1:
                issues.append("SOLVABILITY: Opponent has immediate winning threat that agent cannot counter")
        
        return issues
    
    def _check_impossible_patterns(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Check for common impossible patterns that make the puzzle unsolvable.
        """
        issues = []
        
        grid = level_state['board']['grid']
        
        # Pattern 1: Check for board states that violate Connect-4 physics (floating pieces)
        physics_issues = self._check_physics_violations(grid)
        issues.extend(physics_issues)
        
        # Pattern 2: Check for board states where opponent has overwhelming advantage
        balance_issues = self._check_competitive_balance(grid)
        issues.extend(balance_issues)
        
        # Pattern 3: Check for impossible disk counts (more disks than possible moves)
        count_issues = self._check_disk_count_validity(grid, level_state)
        issues.extend(count_issues)
        
        return issues
    
    def _validate_reward_structure(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for incentive alignment - ensure rewards promote goal achievement.
        """
        issues = []
        
        # The Connect-Four environment uses binary rewards which is good design
        # Check that game state allows for proper reward calculation
        
        game_state = level_state.get('game', {})
        
        # Ensure reward can be determined from game state
        if 'winner' not in game_state:
            issues.append("REWARD: Missing winner field - cannot determine victory condition")
            
        if 'game_over' not in game_state:
            issues.append("REWARD: Missing game_over field - cannot determine terminal state")
            
        # Binary reward structure prevents action grinding and exploration loops
        # This is good design - agent only gets reward for actual victory
        
        # Check that timeout condition exists to prevent infinite episodes
        if 'globals' not in level_state or 'max_steps' not in level_state['globals']:
            issues.append("REWARD: Missing max_steps - could lead to infinite episodes without proper termination")
            
        return issues
    
    def _validate_basic_state(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Validate basic state structure and consistency.
        """
        issues = []
        
        required_keys = ['globals', 'agent', 'opponent', 'board', 'game']
        for key in required_keys:
            if key not in level_state:
                issues.append(f"STRUCTURE: Missing required key '{key}' in level state")
        
        # Validate game state consistency
        if 'game' in level_state:
            game = level_state['game']
            current_player = game.get('current_player', 1)
            if current_player not in [1, 2]:
                issues.append(f"STRUCTURE: Invalid current_player {current_player}, must be 1 or 2")
                
        # Validate board structure
        if 'board' in level_state and 'grid' in level_state['board']:
            grid = level_state['board']['grid']
            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    if cell not in [0, 1, 2]:
                        issues.append(f"STRUCTURE: Invalid cell value {cell} at position ({row_idx}, {col_idx}), must be 0, 1, or 2")
        
        return issues
    
    def _get_available_columns(self, grid: List[List[int]]) -> List[int]:
        """Get list of columns that can accept new disks."""
        available = []
        for col in range(len(grid[0])):
            if grid[0][col] == 0:  # Top row is empty
                available.append(col)
        return available
    
    def _count_available_moves(self, grid: List[List[int]]) -> int:
        """Count total number of empty cells."""
        count = 0
        for row in grid:
            for cell in row:
                if cell == 0:
                    count += 1
        return count
    
    def _estimate_minimum_moves_to_win(self, grid: List[List[int]], player: int) -> int:
        """
        Estimate minimum number of moves needed for player to win.
        Returns a conservative estimate.
        """
        # Check for immediate win (1 move)
        for col in range(self.board_width):
            if self._can_drop_disk(grid, col):
                test_grid = copy.deepcopy(grid)
                row = self._drop_disk_simulation(test_grid, col, player)
                if row != -1 and self._check_win(test_grid, row, col, player):
                    return 1
        
        # Check for 2-move wins (player can set up forced win)
        # This is more complex - for now return conservative estimate
        
        # Count existing sequences and estimate moves needed
        max_sequence = self._get_longest_sequence(grid, player)
        gaps_to_fill = max(0, 4 - max_sequence)
        
        return max(gaps_to_fill, 2)  # At least 2 moves needed if no immediate win
    
    def _check_physics_violations(self, grid: List[List[int]]) -> List[str]:
        """Check for floating pieces that violate gravity."""
        issues = []
        
        for col in range(len(grid[0])):
            found_empty = False
            for row in range(len(grid)):
                if grid[row][col] == 0:
                    found_empty = True
                elif found_empty and grid[row][col] != 0:
                    issues.append(f"PHYSICS: Floating disk at position ({row}, {col}) - violates gravity rules")
        
        return issues
    
    def _check_competitive_balance(self, grid: List[List[int]]) -> List[str]:
        """Check if game state is reasonably balanced."""
        issues = []
        
        agent_count = sum(row.count(1) for row in grid)
        opponent_count = sum(row.count(2) for row in grid)
        
        # In Connect-4, player difference should never exceed 1
        if abs(agent_count - opponent_count) > 1:
            issues.append(f"BALANCE: Unrealistic disk count - Agent: {agent_count}, Opponent: {opponent_count}")
        
        # Agent should have equal or one more disk (goes first)
        if opponent_count > agent_count:
            issues.append("BALANCE: Opponent has more disks than agent - violates turn order")
            
        return issues
    
    def _check_disk_count_validity(self, grid: List[List[int]], level_state: Dict[str, Any]) -> List[str]:
        """Validate that disk counts match possible game progression."""
        issues = []
        
        moves_made = level_state.get('game', {}).get('moves_made', 0)
        agent_disks = sum(row.count(1) for row in grid)
        opponent_disks = sum(row.count(2) for row in grid)
        total_disks = agent_disks + opponent_disks
        
        # Total disks should not exceed moves that could have been made
        if total_disks > moves_made + 2:  # +2 for current turn cycle
            issues.append(f"COUNT: Too many disks on board ({total_disks}) for moves made ({moves_made})")
            
        return issues
    
    def _can_drop_disk(self, grid: List[List[int]], col: int) -> bool:
        """Check if a disk can be dropped in the given column."""
        if col < 0 or col >= len(grid[0]):
            return False
        return grid[0][col] == 0
    
    def _drop_disk_simulation(self, grid: List[List[int]], col: int, player: int) -> int:
        """Simulate dropping a disk and return the row it lands in."""
        if not self._can_drop_disk(grid, col):
            return -1
            
        for row in range(len(grid) - 1, -1, -1):
            if grid[row][col] == 0:
                grid[row][col] = player
                return row
        return -1
    
    def _check_win(self, grid: List[List[int]], row: int, col: int, player: int) -> bool:
        """Check if placing a disk at (row, col) creates a win for player."""
        height = len(grid)
        width = len(grid[0])
        
        # Check horizontal
        count = 1
        # Left
        c = col - 1
        while c >= 0 and grid[row][c] == player:
            count += 1
            c -= 1
        # Right  
        c = col + 1
        while c < width and grid[row][c] == player:
            count += 1
            c += 1
        if count >= 4:
            return True
        
        # Check vertical
        count = 1
        r = row + 1
        while r < height and grid[r][col] == player:
            count += 1
            r += 1
        if count >= 4:
            return True
        
        # Check diagonal /
        count = 1
        r, c = row + 1, col - 1
        while r < height and c >= 0 and grid[r][c] == player:
            count += 1
            r += 1
            c -= 1
        r, c = row - 1, col + 1
        while r >= 0 and c < width and grid[r][c] == player:
            count += 1
            r -= 1
            c += 1
        if count >= 4:
            return True
        
        # Check diagonal \
        count = 1
        r, c = row + 1, col + 1
        while r < height and c < width and grid[r][c] == player:
            count += 1
            r += 1
            c += 1
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and grid[r][c] == player:
            count += 1
            r -= 1
            c -= 1
        if count >= 4:
            return True
        
        return False
    
    def _get_longest_sequence(self, grid: List[List[int]], player: int) -> int:
        """Get the longest existing sequence for the player."""
        max_length = 0
        height = len(grid)
        width = len(grid[0])
        
        # Check all positions
        for row in range(height):
            for col in range(width):
                if grid[row][col] == player:
                    # Check all 4 directions from this position
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # right, down, diagonal-right, diagonal-left
                    
                    for dr, dc in directions:
                        length = 1
                        r, c = row + dr, col + dc
                        while (0 <= r < height and 0 <= c < width and grid[r][c] == player):
                            length += 1
                            r += dr
                            c += dc
                        max_length = max(max_length, length)
        
        return max_length


def validate_connect_four_level(level_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Main validation function for Connect-Four levels.
    
    Args:
        level_state: The generated level state to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    validator = ConnectFourValidator()
    return validator.validate_level(level_state)