from typing import Dict, Any, List, Tuple, Optional
import yaml
import copy

class MemoryPairValidator:
    def __init__(self):
        self.validation_results = []
        
    def validate_level(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a Memory Pair Matching level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(level_data)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(level_data)
        issues.extend(reward_issues)
        
        # 3. CONFIGURATION CONSISTENCY
        config_issues = self._check_configuration_consistency(level_data)
        issues.extend(config_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, level_data: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        cards = level_data.get('game', {}).get('cards', [])
        max_steps = level_data.get('globals', {}).get('max_steps', 40)
        
        # Validate card array structure
        if len(cards) != 16:
            issues.append(f"SOLVABILITY: Invalid card count {len(cards)}, must be 16")
            return issues
        
        # Validate symbol pairing
        symbol_counts = {}
        for symbol in cards:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Check that each symbol appears exactly twice
        for symbol, count in symbol_counts.items():
            if count != 2:
                issues.append(f"SOLVABILITY: Symbol {symbol} appears {count} times, must appear exactly 2 times")
        
        # Check that we have exactly 8 unique symbols
        unique_symbols = len(symbol_counts)
        if unique_symbols != 8:
            issues.append(f"SOLVABILITY: Found {unique_symbols} unique symbols, must have exactly 8")
        
        # TARGET REACHABILITY ANALYSIS
        # Calculate minimum steps needed to solve optimally
        min_steps_needed = self._calculate_minimum_steps(level_data)
        if min_steps_needed > max_steps:
            issues.append(f"SOLVABILITY: Minimum steps needed ({min_steps_needed}) exceeds step limit ({max_steps})")
        
        # Check for valid symbol range (symbols should be 0-7 for 8 pairs)
        expected_symbols = set(range(8))
        actual_symbols = set(symbol_counts.keys())
        if actual_symbols != expected_symbols:
            issues.append(f"SOLVABILITY: Symbol set mismatch. Expected {expected_symbols}, got {actual_symbols}")
        
        return issues
    
    def _calculate_minimum_steps(self, level_data: Dict[str, Any]) -> int:
        """
        Calculate theoretical minimum steps needed to solve perfectly.
        In memory matching, optimal play requires:
        - First 8 flips: explore 8 different symbols
        - Next 8 flips: match each previously seen symbol with its pair
        - Total: 16 flips minimum with perfect memory and no mistakes
        """
        return 16  # Theoretical minimum for perfect play
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Simulate reward structure to check for proper incentive alignment
        
        # 1. GOAL-ORIENTED REWARDS CHECK
        # Verify that clearing all pairs gives the highest total reward
        max_pair_reward = 8 * 1.0  # 8 pairs × 1.0 points each
        max_exploration_reward = 16 * 0.05  # 16 positions × 0.05 points each
        
        total_possible_reward = max_pair_reward + max_exploration_reward
        
        # Check that pair clearing dominates the reward structure
        pair_reward_percentage = max_pair_reward / total_possible_reward
        if pair_reward_percentage < 0.8:  # Pair rewards should be at least 80% of total
            issues.append(f"REWARD: Pair clearing rewards ({max_pair_reward}) should dominate total rewards ({total_possible_reward})")
        
        # 2. AVOID INCENTIVE MISALIGNMENT
        # Check that exploration rewards don't encourage excessive exploration
        if max_exploration_reward > max_pair_reward:
            issues.append("REWARD: Exploration rewards exceed pair clearing rewards, may encourage exploration over solving")
        
        # 3. REWARD DESIGN PRINCIPLES VALIDATION
        # Verify reward sparsity vs density balance
        exploration_per_action = 0.05
        pair_reward_per_action = 1.0
        
        if exploration_per_action >= pair_reward_per_action:
            issues.append("REWARD: Exploration reward per action should be much smaller than pair clearing reward")
        
        # Check efficiency incentive - no explicit efficiency bonus in current design
        # This is acceptable as the step limit provides implicit efficiency pressure
        
        return issues
    
    def _check_configuration_consistency(self, level_data: Dict[str, Any]) -> List[str]:
        """Check for consistent configuration across all components"""
        issues = []
        
        # Validate state structure
        required_globals = ['max_steps', 'total_pairs', 'grid_size']
        globals_dict = level_data.get('globals', {})
        
        for key in required_globals:
            if key not in globals_dict:
                issues.append(f"CONFIG: Missing required global parameter: {key}")
        
        # Validate game state structure
        required_game_fields = ['cards', 'card_states', 'revealed_positions', 
                               'cleared_pairs', 'current_revealed_symbol', 'explored_positions']
        game_dict = level_data.get('game', {})
        
        for key in required_game_fields:
            if key not in game_dict:
                issues.append(f"CONFIG: Missing required game field: {key}")
        
        # Validate agent state
        agent_dict = level_data.get('agent', {})
        if 'steps_remaining' not in agent_dict:
            issues.append("CONFIG: Missing required agent field: steps_remaining")
        
        # Validate initial state consistency
        if 'card_states' in game_dict:
            card_states = game_dict['card_states']
            if len(card_states) != 16:
                issues.append(f"CONFIG: card_states length {len(card_states)} != 16")
            
            # All cards should start face-down (state 0)
            if any(state != 0 for state in card_states):
                issues.append("CONFIG: All cards must start face-down (state 0)")
        
        # Validate initial cleared pairs
        if game_dict.get('cleared_pairs', 0) != 0:
            issues.append("CONFIG: cleared_pairs must start at 0")
        
        # Validate initial revealed positions
        if len(game_dict.get('revealed_positions', [])) != 0:
            issues.append("CONFIG: revealed_positions must start empty")
        
        # Validate initial explored positions
        if len(game_dict.get('explored_positions', [])) != 0:
            issues.append("CONFIG: explored_positions must start empty")
        
        # Validate step consistency
        max_steps = globals_dict.get('max_steps', 40)
        steps_remaining = agent_dict.get('steps_remaining', 40)
        if max_steps != steps_remaining:
            issues.append(f"CONFIG: max_steps ({max_steps}) != initial steps_remaining ({steps_remaining})")
        
        return issues
    
    def check_reward_exploitation(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Simulate agent behavior to check for reward exploitation opportunities.
        """
        issues = []
        
        # Simulate pure exploration strategy
        exploration_only_reward = 16 * 0.05  # 0.8 points for visiting all positions
        
        # Simulate optimal solving strategy  
        optimal_reward = 8 * 1.0 + 16 * 0.05  # 8.8 points (pairs + exploration)
        
        # Simulate suboptimal but exploitative strategy
        # An agent could potentially flip all cards for exploration then run out of steps
        worst_case_steps = level_data.get('globals', {}).get('max_steps', 40)
        
        # Check if exploration-only strategy could compete with solving
        exploration_efficiency = exploration_only_reward / 16  # reward per exploration step
        solving_efficiency = optimal_reward / 16  # reward per step in optimal solution
        
        if exploration_efficiency >= solving_efficiency * 0.5:
            issues.append("EXPLOITATION: Pure exploration strategy may be too rewarding compared to solving")
        
        return len(issues) == 0, issues
    
    def validate_level_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate a level file directly"""
        try:
            with open(file_path, 'r') as f:
                level_data = yaml.safe_load(f)
            return self.validate_level(level_data)
        except Exception as e:
            return False, [f"FILE_ERROR: Unable to load level file {file_path}: {str(e)}"]
    
    def batch_validate_levels(self, level_files: List[str]) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate multiple level files and return results"""
        results = {}
        for file_path in level_files:
            results[file_path] = self.validate_level_file(file_path)
        return results

# Utility function for easy validation
def validate_memory_level(level_data_or_file) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a Memory Pair Matching level.
    
    Args:
        level_data_or_file: Either a dict containing level data or a string path to level file
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = MemoryPairValidator()
    
    if isinstance(level_data_or_file, str):
        return validator.validate_level_file(level_data_or_file)
    elif isinstance(level_data_or_file, dict):
        return validator.validate_level(level_data_or_file)
    else:
        return False, ["VALIDATION_ERROR: Input must be dict or file path string"]