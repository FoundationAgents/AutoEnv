from base.env.base_env import SkinEnv
from env_obs import ConnectFourObservation
from env_generate import ConnectFourGenerator
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List

class ConnectFourOpponent:
    @staticmethod
    def get_move(board_grid: List[List[int]]) -> int:
        height = len(board_grid)
        width = len(board_grid[0])
        
        # Check for winning move (player 2)
        winning_move = ConnectFourOpponent.check_winning_move(board_grid, 2)
        if winning_move != -1:
            return winning_move
        
        # Check for blocking move (block player 1)
        blocking_move = ConnectFourOpponent.check_blocking_move(board_grid, 1)
        if blocking_move != -1:
            return blocking_move
        
        # Random move among available columns
        return ConnectFourOpponent.get_random_move(board_grid)
    
    @staticmethod
    def check_winning_move(board_grid: List[List[int]], player: int) -> int:
        height = len(board_grid)
        width = len(board_grid[0])
        
        for col in range(width):
            # Find lowest empty row
            row = -1
            for r in range(height-1, -1, -1):
                if board_grid[r][col] == 0:
                    row = r
                    break
            
            if row == -1:  # Column full
                continue
            
            # Simulate dropping disk
            board_grid[row][col] = player
            
            # Check if this creates a win
            if ConnectFourOpponent.check_win(board_grid, row, col, player):
                board_grid[row][col] = 0  # Undo simulation
                return col
            
            board_grid[row][col] = 0  # Undo simulation
        
        return -1
    
    @staticmethod
    def check_blocking_move(board_grid: List[List[int]], opponent_player: int) -> int:
        return ConnectFourOpponent.check_winning_move(board_grid, opponent_player)
    
    @staticmethod
    def get_random_move(board_grid: List[List[int]]) -> int:
        width = len(board_grid[0])
        available_cols = []
        
        for col in range(width):
            if board_grid[0][col] == 0:  # Column not full
                available_cols.append(col)
        
        if available_cols:
            return random.choice(available_cols)
        return 0  # Fallback
    
    @staticmethod
    def check_win(board_grid: List[List[int]], row: int, col: int, player: int) -> bool:
        height = len(board_grid)
        width = len(board_grid[0])
        
        # Check horizontal
        count = 1
        # Left
        c = col - 1
        while c >= 0 and board_grid[row][c] == player:
            count += 1
            c -= 1
        # Right
        c = col + 1
        while c < width and board_grid[row][c] == player:
            count += 1
            c += 1
        if count >= 4:
            return True
        
        # Check vertical
        count = 1
        # Down
        r = row + 1
        while r < height and board_grid[r][col] == player:
            count += 1
            r += 1
        if count >= 4:
            return True
        
        # Check diagonal /
        count = 1
        # Down-left
        r, c = row + 1, col - 1
        while r < height and c >= 0 and board_grid[r][c] == player:
            count += 1
            r += 1
            c -= 1
        # Up-right
        r, c = row - 1, col + 1
        while r >= 0 and c < width and board_grid[r][c] == player:
            count += 1
            r -= 1
            c += 1
        if count >= 4:
            return True
        
        # Check diagonal \
        count = 1
        # Down-right
        r, c = row + 1, col + 1
        while r < height and c < width and board_grid[r][c] == player:
            count += 1
            r += 1
            c += 1
        # Up-left
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0 and board_grid[r][c] == player:
            count += 1
            r -= 1
            c -= 1
        if count >= 4:
            return True
        
        return False

class ConnectFourEnv(SkinEnv):
    def __init__(self, env_id):
        obs_policy = ConnectFourObservation()
        super().__init__(env_id, obs_policy)
        self.generator = ConnectFourGenerator(env_id, self.configs)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "generate", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id required for load mode")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
        
        self._t = 0
        self._history = []
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action')
        params = action.get('params', {})
        
        if action_name != 'drop_disk':
            return self._state
        
        column = params.get('column')
        if column is None or not (0 <= column <= 6):
            return self._state
        
        # Agent move
        board_grid = self._state['board']['grid']
        height = len(board_grid)
        
        # Find lowest empty row in column
        agent_row = -1
        for r in range(height-1, -1, -1):
            if board_grid[r][column] == 0:
                agent_row = r
                break
        
        if agent_row != -1:  # Valid move
            board_grid[agent_row][column] = 1  # Agent disk
            self._state['game']['moves_made'] += 1
            
            # Check if agent wins
            if ConnectFourOpponent.check_win(board_grid, agent_row, column, 1):
                self._state['game']['winner'] = 1
                self._state['game']['game_over'] = True
                return self._state
        
        # Opponent move
        if not self._state['game']['game_over']:
            opponent_col = ConnectFourOpponent.get_move(board_grid)
            self._state['opponent']['last_move'] = opponent_col
            
            # Find lowest empty row for opponent
            opponent_row = -1
            for r in range(height-1, -1, -1):
                if board_grid[r][opponent_col] == 0:
                    opponent_row = r
                    break
            
            if opponent_row != -1:
                board_grid[opponent_row][opponent_col] = 2  # Opponent disk
                
                # Check if opponent wins
                if ConnectFourOpponent.check_win(board_grid, opponent_row, opponent_col, 2):
                    self._state['game']['winner'] = 2
                    self._state['game']['game_over'] = True
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        if self._state['game']['game_over']:
            if self._state['game']['winner'] == 1:
                events.append('game_won')
                reward_info['agent_victory'] = 1.0
                return 1.0, events, reward_info
            elif self._state['game']['winner'] == 2:
                events.append('game_lost')
                reward_info['opponent_victory'] = 0.0
                return 0.0, events, reward_info
        
        if self._t >= self.configs['termination']['max_steps']:
            events.append('game_timeout')
            reward_info['no_winner'] = 0.0
            return 0.0, events, reward_info
        
        return 0.0, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        board_grid = omega['board_grid']
        opponent_last_move = omega['opponent_last_move']
        max_steps = omega['max_steps']
        moves_made = omega['moves_made']
        t = omega['t']
        
        # Format board display
        board_lines = []
        for row in board_grid:
            row_str = ' '.join(str(cell) for cell in row)
            board_lines.append(row_str)
        board_display = '\n'.join(board_lines)
        
        # Game status
        if self._state['game']['game_over']:
            if self._state['game']['winner'] == 1:
                game_status = "You won!"
            elif self._state['game']['winner'] == 2:
                game_status = "Opponent won!"
            else:
                game_status = "Game over"
        else:
            game_status = "In progress"
        
        return f"""Step {t + 1}/{max_steps} | Moves: {moves_made}
Last opponent move: Column {opponent_last_move}

Board (1=You, 2=Opponent, 0=Empty):
{board_display}

Available actions: drop_disk(column) where column in [0,1,2,3,4,5,6]
Game status: {game_status}"""
    
    def done(self, state=None) -> bool:
        return (self._state['game']['game_over'] or 
                self._state['game']['winner'] != 0 or 
                self._t >= self.configs['termination']['max_steps'])