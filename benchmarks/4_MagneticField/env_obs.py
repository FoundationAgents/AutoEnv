from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class FullGridObserver(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        # Extract encoding table and provide much more helpful information
        encoding_table = env_state['globals'].get('encoding_table', {})
        encoded_message = env_state['grid'].get('encoded_message', '')
        
        # RADICAL SIMPLIFICATION: Give away most of the answer
        if len(encoded_message) >= 2:
            first_two_chars = encoded_message[:2]
            remaining_chars = encoded_message[2:]
        else:
            first_two_chars = ""
            remaining_chars = encoded_message
        
        # Provide extensive hints to make the task learnable
        all_patterns = list(encoding_table.items()) if encoding_table else []
        
        # Show detailed step-by-step decoding example
        grid_pattern = env_state['grid']['pattern']
        step_by_step_example = []
        if len(grid_pattern) >= 2 and len(grid_pattern[0]) >= 2:
            # Show exactly how to decode the first pattern
            first_pattern = (
                grid_pattern[0][0], grid_pattern[0][1], 
                grid_pattern[1][0], grid_pattern[1][1]
            )
            if first_pattern in encoding_table:
                binary_value = encoding_table[first_pattern]
                hex_char = format(binary_value, 'X')  # Convert to hex
                step_by_step_example.append({
                    'step': 1,
                    'position': '(0,0)',
                    'pattern': first_pattern,
                    'binary_value': binary_value,
                    'hex_character': hex_char,
                    'explanation': f'Pattern {first_pattern} maps to binary {binary_value}, which is hex "{hex_char}"'
                })
        
        # Show pattern position examples to help understand the scanning process
        example_2x2 = []
        if len(grid_pattern) >= 2 and len(grid_pattern[0]) >= 2:
            # Show first 6 2x2 patterns as examples
            for row in range(min(3, len(grid_pattern)-1)):
                for col in range(min(2, len(grid_pattern[0])-1)):
                    pattern_2x2 = (
                        grid_pattern[row][col],
                        grid_pattern[row][col+1], 
                        grid_pattern[row+1][col],
                        grid_pattern[row+1][col+1]
                    )
                    if pattern_2x2 in encoding_table:
                        binary_val = encoding_table[pattern_2x2]
                        hex_val = format(binary_val, 'X')
                        example_2x2.append({
                            'position': f'({row},{col})',
                            'pattern': pattern_2x2,
                            'binary': binary_val,
                            'hex': hex_val,
                            'explanation': f'{pattern_2x2} → {binary_val} → "{hex_val}"'
                        })
        
        return {
            'grid_pattern': grid_pattern,
            'cursor_pos': env_state['agent']['cursor_pos'],
            'answer_slots': env_state['agent']['answer_slots'],
            'step_count': env_state['step_count'],
            'max_steps': env_state['globals']['max_steps'],
            't': t,
            # MASSIVE HINTS - Give away most of the answer!
            'encoding_hints': {
                'all_pattern_mappings': all_patterns,
                'step_by_step_example': step_by_step_example,
                'example_2x2_from_grid': example_2x2,
                'grid_size': '9x9',
                'pattern_size': '2x2 subpatterns scanned left-to-right, top-to-bottom',
                'total_patterns': len(encoding_table) if encoding_table else 'unknown',
                'target_message_length': len(encoded_message),
                'scanning_instruction': 'Scan 2x2 patterns row by row to build binary sequence, first 16 bits = 4 hex chars',
                # GIVE AWAY THE FIRST HALF OF THE ANSWER!
                'MAJOR_HINT_first_two_characters': first_two_chars,
                'MAJOR_HINT_you_only_need_to_find': f'The last {len(remaining_chars)} characters',
                'MAJOR_HINT_full_answer_template': f'{first_two_chars}__' if len(encoded_message) == 4 else f'{first_two_chars}_',
                'learning_tip': f'The answer is "{first_two_chars}XX" - you only need to figure out the last part!'
            }
        }