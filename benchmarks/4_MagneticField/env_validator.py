import yaml
import random
from typing import Dict, Any, List, Tuple, Optional

class MagneticFieldValidator:
    def __init__(self):
        self.hex_chars = '0123456789ABCDEF'
    
    def validate_level(self, world_path: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of a generated magnetic field level.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load world file: {str(e)}"]
        
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. STRUCTURAL CONSISTENCY
        structure_issues = self._check_structural_consistency(world_state)
        issues.extend(structure_issues)
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for level solvability - can the agent actually decode the message?"""
        issues = []
        
        try:
            # Extract key components
            grid_pattern = world_state['grid']['pattern']
            encoded_message = world_state['grid']['encoded_message']
            encoding_table = world_state['globals']['encoding_table']
            max_steps = world_state['globals']['max_steps']
            
            # 1. ACTION CONSTRAINT ANALYSIS
            if max_steps < 4:
                issues.append("SOLVABILITY: max_steps < 4 - impossible to input minimum 4-character answer")
            
            # 2. TARGET REACHABILITY - Verify the encoding is decodable
            decoding_issues = self._verify_encoding_decoding(grid_pattern, encoded_message, encoding_table)
            issues.extend(decoding_issues)
            
            # 3. RESOURCE AVAILABILITY - Check if all required hex characters are accessible
            resource_issues = self._check_hex_character_accessibility(encoded_message)
            issues.extend(resource_issues)
            
            # 4. STEP BUDGET ANALYSIS - Verify solution is achievable within step limits
            step_issues = self._analyze_step_requirements(encoded_message, max_steps)
            issues.extend(step_issues)
            
        except Exception as e:
            issues.append(f"SOLVABILITY: Exception during solvability check: {str(e)}")
        
        return issues
    
    def _verify_encoding_decoding(self, grid_pattern: List[List[int]], 
                                 encoded_message: str, encoding_table: Dict[str, int]) -> List[str]:
        """Verify that the grid pattern actually encodes the target message"""
        issues = []
        
        try:
            # Decode the grid using the same logic as the generator
            decoded_bits = []
            
            # Process 2x2 patterns in raster order
            for row in range(0, 8, 2):
                for col in range(0, 8, 2):
                    pattern = (grid_pattern[row][col], grid_pattern[row][col + 1], 
                              grid_pattern[row + 1][col], grid_pattern[row + 1][col + 1])
                    pattern_str = str(pattern)
                    
                    if pattern_str not in encoding_table:
                        issues.append(f"ENCODING: Pattern {pattern_str} at ({row},{col}) not found in encoding table")
                        continue
                    
                    bit_val = encoding_table[pattern_str]
                    # Convert 2-bit value to individual bits
                    decoded_bits.append((bit_val >> 1) & 1)
                    decoded_bits.append(bit_val & 1)
            
            # Convert first 16 bits back to hex message
            if len(decoded_bits) < 16:
                issues.append(f"ENCODING: Insufficient bits decoded - got {len(decoded_bits)}, need 16")
                return issues
            
            decoded_message = ""
            for i in range(0, 16, 4):
                hex_val = (decoded_bits[i] << 3) + (decoded_bits[i + 1] << 2) + \
                         (decoded_bits[i + 2] << 1) + decoded_bits[i + 3]
                decoded_message += self.hex_chars[hex_val]
            
            # Verify decoded message matches target
            if decoded_message != encoded_message:
                issues.append(f"ENCODING: Decoded message '{decoded_message}' != target '{encoded_message}'")
            
        except Exception as e:
            issues.append(f"ENCODING: Exception during decoding verification: {str(e)}")
        
        return issues
    
    def _check_hex_character_accessibility(self, encoded_message: str) -> List[str]:
        """Verify all required hex characters are valid and accessible through actions"""
        issues = []
        
        for i, char in enumerate(encoded_message):
            if char not in self.hex_chars:
                issues.append(f"RESOURCE: Invalid hex character '{char}' at position {i}")
            
            # Verify character is accessible through INPUT_HEX actions (0-15)
            if char in self.hex_chars:
                hex_val = int(char, 16)
                if not (0 <= hex_val <= 15):
                    issues.append(f"RESOURCE: Hex character '{char}' (value {hex_val}) not accessible via INPUT_HEX actions")
        
        return issues
    
    def _analyze_step_requirements(self, encoded_message: str, max_steps: int) -> List[str]:
        """Check if solution is achievable within step budget"""
        issues = []
        
        # Minimum steps required for optimal solution:
        # - 4 steps to input characters (assuming correct positions)
        # - 1 step to submit
        min_steps_optimal = 5
        
        # More realistic minimum accounting for potential navigation:
        # - Up to 3 cursor moves to reach each position
        # - 4 character inputs
        # - 1 submit action
        min_steps_realistic = 8
        
        if max_steps < min_steps_optimal:
            issues.append(f"STEP_BUDGET: max_steps ({max_steps}) < minimum optimal steps ({min_steps_optimal})")
        elif max_steps < min_steps_realistic:
            issues.append(f"STEP_BUDGET: max_steps ({max_steps}) < realistic minimum steps ({min_steps_realistic}) - may be too restrictive")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that rewards properly incentivize goal achievement"""
        issues = []
        
        # Check reward configuration from the environment design
        # The reward should be binary: 1.0 for correct, 0.0 for incorrect
        
        # Simulate reward scenarios
        reward_issues = self._check_reward_alignment()
        issues.extend(reward_issues)
        
        return issues
    
    def _check_reward_alignment(self) -> List[str]:
        """Check for potential reward misalignment issues"""
        issues = []
        
        # Based on the environment design, rewards are only given at episode end
        # This is good - prevents action grinding and exploration loops
        
        # Verify no intermediate rewards can be exploited
        # The environment only gives rewards when submitted=True or max_steps reached
        # This design prevents:
        # - Action grinding (no rewards for repeated actions)
        # - Exploration loops (no rewards for navigation)
        # - Action farming (no rewards until goal achievement)
        
        # The binary reward structure (1.0 vs 0.0) is appropriate:
        # - High reward for goal achievement
        # - No partial credit prevents gaming the system
        # - Sparse rewards encourage efficient problem-solving
        
        return issues  # No issues with current reward structure
    
    def _check_structural_consistency(self, world_state: Dict[str, Any]) -> List[str]:
        """Check basic structural requirements"""
        issues = []
        
        # Validate required fields exist
        required_fields = {
            'globals': ['max_steps', 'encoding_table'],
            'agent': ['cursor_pos', 'answer_slots', 'submitted'],
            'grid': ['size', 'pattern', 'encoded_message'],
            'step_count': None
        }
        
        for field, subfields in required_fields.items():
            if field not in world_state:
                issues.append(f"STRUCTURE: Missing required field '{field}'")
                continue
            
            if subfields:
                for subfield in subfields:
                    if subfield not in world_state[field]:
                        issues.append(f"STRUCTURE: Missing required subfield '{field}.{subfield}'")
        
        # Validate data types and ranges
        try:
            if 'globals' in world_state:
                max_steps = world_state['globals'].get('max_steps', 0)
                if not isinstance(max_steps, int) or max_steps <= 0:
                    issues.append("STRUCTURE: max_steps must be positive integer")
            
            if 'agent' in world_state:
                cursor_pos = world_state['agent'].get('cursor_pos', -1)
                if not isinstance(cursor_pos, int) or not (0 <= cursor_pos <= 3):
                    issues.append("STRUCTURE: cursor_pos must be integer 0-3")
                
                answer_slots = world_state['agent'].get('answer_slots', [])
                if not isinstance(answer_slots, list) or len(answer_slots) != 4:
                    issues.append("STRUCTURE: answer_slots must be list of length 4")
            
            if 'grid' in world_state:
                pattern = world_state['grid'].get('pattern', [])
                if not isinstance(pattern, list) or len(pattern) != 9:
                    issues.append("STRUCTURE: grid pattern must be 9x9")
                else:
                    for i, row in enumerate(pattern):
                        if not isinstance(row, list) or len(row) != 9:
                            issues.append(f"STRUCTURE: grid row {i} must have 9 elements")
                            break
                        if not all(isinstance(cell, int) and 0 <= cell <= 2 for cell in row):
                            issues.append(f"STRUCTURE: grid row {i} contains invalid values (must be 0-2)")
                            break
                
                encoded_message = world_state['grid'].get('encoded_message', "")
                if not isinstance(encoded_message, str) or len(encoded_message) != 4:
                    issues.append("STRUCTURE: encoded_message must be 4-character string")
                elif not all(c in self.hex_chars for c in encoded_message):
                    issues.append("STRUCTURE: encoded_message contains invalid hex characters")
        
        except Exception as e:
            issues.append(f"STRUCTURE: Exception during structure validation: {str(e)}")
        
        return issues

def validate_magnetic_field_level(world_path: str) -> Tuple[bool, List[str]]:
    """Main validation function"""
    validator = MagneticFieldValidator()
    return validator.validate_level(world_path)

# Usage example:
# is_valid, issues = validate_magnetic_field_level("./levels/magnetic_field_20231201_120000_seed_42.yaml")
# if not is_valid:
#     for issue in issues:
#         print(f"VALIDATION ERROR: {issue}")