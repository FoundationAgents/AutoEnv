from base.env.base_generator import WorldGenerator
import yaml
import random
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

class MagneticFieldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.hex_chars = '0123456789ABCDEF'
    
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
            yaml.dump(world_state, f)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = {
            'globals': base_state['globals'].copy(),
            'agent': base_state['agent'].copy(),
            'grid': base_state['grid'].copy(),
            'step_count': base_state['step_count']
        }
        
        encoding_table = self._generate_encoding_table()
        world_state['globals']['encoding_table'] = encoding_table
        
        message = self._generate_message()
        world_state['grid']['encoded_message'] = message
        
        grid_pattern = self._create_grid_pattern(message, encoding_table)
        world_state['grid']['pattern'] = grid_pattern
        
        self._validate_encoding(grid_pattern, message, encoding_table)
        
        return world_state
    
    def _generate_encoding_table(self) -> Dict[str, int]:
        patterns = []
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        patterns.append((a, b, c, d))
        
        random.shuffle(patterns)
        
        encoding_table = {}
        bit_values = [0, 1, 2, 3]  # 00, 01, 10, 11 in binary
        
        for i, bit_val in enumerate(bit_values):
            for j in range(4):
                pattern_idx = i * 4 + j
                if pattern_idx < len(patterns):
                    pattern_key = str(patterns[pattern_idx])
                    encoding_table[pattern_key] = bit_val
        
        return encoding_table
    
    def _generate_message(self) -> str:
        message = ''.join(random.choices(self.hex_chars, k=4))
        return message
    
    def _create_grid_pattern(self, message: str, encoding_table: Dict[str, int]) -> List[List[int]]:
        grid = [[0 for _ in range(9)] for _ in range(9)]
        
        message_bits = []
        for char in message:
            hex_val = int(char, 16)
            bits = [(hex_val >> i) & 1 for i in range(3, -1, -1)]
            message_bits.extend(bits)
        
        reverse_table = {}
        for pattern_str, bit_val in encoding_table.items():
            if bit_val not in reverse_table:
                reverse_table[bit_val] = []
            reverse_table[bit_val].append(eval(pattern_str))
        
        bit_pairs = []
        for i in range(0, len(message_bits), 2):
            if i + 1 < len(message_bits):
                pair_val = message_bits[i] * 2 + message_bits[i + 1]
                bit_pairs.append(pair_val)
        
        pattern_positions = []
        for row in range(0, 8, 2):
            for col in range(0, 8, 2):
                pattern_positions.append((row, col))
        
        for i, pair_val in enumerate(bit_pairs):
            if i < len(pattern_positions):
                row, col = pattern_positions[i]
                pattern = random.choice(reverse_table[pair_val])
                grid[row][col] = pattern[0]
                grid[row][col + 1] = pattern[1]
                grid[row + 1][col] = pattern[2]
                grid[row + 1][col + 1] = pattern[3]
        
        remaining_positions = pattern_positions[len(bit_pairs):]
        for row, col in remaining_positions:
            random_val = random.choice(list(reverse_table.keys()))
            pattern = random.choice(reverse_table[random_val])
            grid[row][col] = pattern[0]
            grid[row][col + 1] = pattern[1]
            grid[row + 1][col] = pattern[2]
            grid[row + 1][col + 1] = pattern[3]
        
        for row in range(9):
            if grid[row][8] == 0:
                grid[row][8] = random.choice([0, 1, 2])
        for col in range(8):
            if grid[8][col] == 0:
                grid[8][col] = random.choice([0, 1, 2])
        
        return grid
    
    def _validate_encoding(self, grid: List[List[int]], message: str, encoding_table: Dict[str, int]):
        decoded_bits = []
        
        for row in range(0, 8, 2):
            for col in range(0, 8, 2):
                pattern = (grid[row][col], grid[row][col + 1], 
                          grid[row + 1][col], grid[row + 1][col + 1])
                pattern_str = str(pattern)
                if pattern_str in encoding_table:
                    bit_val = encoding_table[pattern_str]
                    decoded_bits.append((bit_val >> 1) & 1)
                    decoded_bits.append(bit_val & 1)
        
        decoded_message = ""
        for i in range(0, min(16, len(decoded_bits)), 4):
            if i + 3 < len(decoded_bits):
                hex_val = (decoded_bits[i] << 3) + (decoded_bits[i + 1] << 2) + \
                         (decoded_bits[i + 2] << 1) + decoded_bits[i + 3]
                decoded_message += self.hex_chars[hex_val]
        
        assert decoded_message == message, f"Encoding validation failed: {decoded_message} != {message}"
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if seed is not None:
            return f"magnetic_field_{timestamp}_seed_{seed}"
        else:
            return f"magnetic_field_{timestamp}_random"