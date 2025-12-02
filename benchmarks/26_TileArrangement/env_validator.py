import yaml
from typing import Dict, Any, List, Tuple, Optional
import os

class MismatchedMemoryValidator:
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_level(self, world_path: str) -> Dict[str, Any]:
        """
        Comprehensive validation of a generated Mismatched Memory level.
        Returns validation results with solvability analysis and reward structure checks.
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return {
                "is_valid": False,
                "is_solvable": False,
                "errors": [f"Failed to load world file: {str(e)}"],
                "warnings": [],
                "solvability_analysis": {},
                "reward_analysis": {}
            }
        
        # Core structure validation
        structure_valid = self._validate_structure(world_state)
        
        # Solvability analysis (most critical)
        solvability_result = self._analyze_solvability(world_state)
        
        # Reward structure validation
        reward_analysis = self._validate_reward_structure(world_state)
        
        # Configuration consistency
        config_valid = self._validate_configuration(world_state)
        
        is_valid = structure_valid and config_valid and len(self.validation_errors) == 0
        is_solvable = solvability_result["is_solvable"]
        
        return {
            "is_valid": is_valid,
            "is_solvable": is_solvable,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "solvability_analysis": solvability_result,
            "reward_analysis": reward_analysis
        }
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> bool:
        """Validate basic world structure and required components."""
        required_sections = ["globals", "board", "game"]
        
        for section in required_sections:
            if section not in world_state:
                self.validation_errors.append(f"Missing required section: {section}")
                return False
        
        # Validate globals
        globals_data = world_state["globals"]
        required_globals = ["max_steps", "total_pairs", "grid_size"]
        for key in required_globals:
            if key not in globals_data:
                self.validation_errors.append(f"Missing required global: {key}")
        
        # Validate board structure
        board = world_state["board"]
        if "cards" not in board or "card_states" not in board:
            self.validation_errors.append("Board missing cards or card_states")
            return False
        
        # Check board dimensions
        grid_size = globals_data.get("grid_size", 4)
        cards = board["cards"]
        card_states = board["card_states"]
        
        if len(cards) != grid_size or len(card_states) != grid_size:
            self.validation_errors.append(f"Board dimensions don't match grid_size {grid_size}")
            return False
        
        for i in range(grid_size):
            if len(cards[i]) != grid_size or len(card_states[i]) != grid_size:
                self.validation_errors.append(f"Row {i} has incorrect length")
                return False
        
        # Validate game section
        game = world_state["game"]
        required_game_keys = ["symbol_pairs", "discovered_pairs", "seen_symbols", "step_count", "cumulative_reward"]
        for key in required_game_keys:
            if key not in game:
                self.validation_errors.append(f"Missing required game state: {key}")
        
        return len(self.validation_errors) == 0
    
    def _analyze_solvability(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critical solvability analysis - ensures the level is actually solvable.
        This is the most important validation check.
        """
        analysis = {
            "is_solvable": False,
            "blocking_issues": [],
            "resource_analysis": {},
            "action_constraints": {},
            "step_budget_analysis": {}
        }
        
        try:
            board = world_state["board"]
            game = world_state["game"]
            globals_data = world_state["globals"]
            
            cards = board["cards"]
            symbol_pairs = game["symbol_pairs"]
            max_steps = globals_data["max_steps"]
            total_pairs = globals_data["total_pairs"]
            
            # 1. RESOURCE AVAILABILITY CHECK
            resource_check = self._check_resource_availability(cards, symbol_pairs, total_pairs)
            analysis["resource_analysis"] = resource_check
            
            if not resource_check["sufficient_resources"]:
                analysis["blocking_issues"].extend(resource_check["issues"])
                return analysis
            
            # 2. ACTION CONSTRAINT ANALYSIS  
            action_analysis = self._analyze_action_constraints(cards, symbol_pairs)
            analysis["action_constraints"] = action_analysis
            
            if not action_analysis["actions_sufficient"]:
                analysis["blocking_issues"].extend(action_analysis["issues"])
                return analysis
            
            # 3. STEP BUDGET FEASIBILITY
            step_analysis = self._analyze_step_budget(cards, symbol_pairs, max_steps, total_pairs)
            analysis["step_budget_analysis"] = step_analysis
            
            if not step_analysis["budget_sufficient"]:
                analysis["blocking_issues"].extend(step_analysis["issues"])
                return analysis
            
            # 4. PAIRING RULE CONSISTENCY
            pairing_check = self._validate_pairing_consistency(cards, symbol_pairs)
            if not pairing_check["consistent"]:
                analysis["blocking_issues"].extend(pairing_check["issues"])
                return analysis
            
            # If all checks pass, level is solvable
            analysis["is_solvable"] = True
            
        except Exception as e:
            analysis["blocking_issues"].append(f"Solvability analysis failed: {str(e)}")
        
        return analysis
    
    def _check_resource_availability(self, cards: List[List[str]], symbol_pairs: Dict[str, str], total_pairs: int) -> Dict[str, Any]:
        """Check if all required resources (symbol pairs) are available on the board."""
        # Count symbol occurrences
        symbol_counts = {}
        all_symbols = set()
        
        for row in cards:
            for symbol in row:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                all_symbols.add(symbol)
        
        issues = []
        
        # Each symbol should appear exactly twice
        for symbol, count in symbol_counts.items():
            if count != 2:
                issues.append(f"Symbol '{symbol}' appears {count} times, expected 2")
        
        # Verify all symbols in pairing map exist on board
        for symbol in symbol_pairs.keys():
            if symbol not in all_symbols:
                issues.append(f"Symbol '{symbol}' in pairing map but not on board")
        
        # Verify pairing completeness - should have exactly 8 symbols forming 4 complete pairs
        if len(all_symbols) != 8:
            issues.append(f"Expected 8 unique symbols, found {len(all_symbols)}")
        
        # Check that pairing creates exactly the required number of pairs
        unique_pairs = set()
        for symbol, partner in symbol_pairs.items():
            pair = tuple(sorted([symbol, partner]))
            unique_pairs.add(pair)
        
        if len(unique_pairs) != total_pairs:
            issues.append(f"Symbol pairing creates {len(unique_pairs)} unique pairs, expected {total_pairs}")
        
        return {
            "sufficient_resources": len(issues) == 0,
            "issues": issues,
            "symbol_counts": symbol_counts,
            "unique_pairs_count": len(unique_pairs)
        }
    
    def _analyze_action_constraints(self, cards: List[List[str]], symbol_pairs: Dict[str, str]) -> Dict[str, Any]:
        """Analyze if available actions are sufficient to reach target state."""
        issues = []
        
        # Check inverse pairing rule consistency
        for symbol, partner in symbol_pairs.items():
            if symbol == partner:
                issues.append(f"Symbol '{symbol}' paired with itself - violates inverse matching rule")
            
            # Check bidirectional consistency
            if symbol_pairs.get(partner) != symbol:
                issues.append(f"Asymmetric pairing: {symbol}->{partner} but {partner}->{symbol_pairs.get(partner)}")
        
        # FLIP action analysis - can we reveal all necessary cards?
        grid_size = len(cards)
        total_positions = grid_size * grid_size
        
        # Each position can be flipped (no locked/protected elements)
        # This environment has no action limitations beyond step budget
        
        return {
            "actions_sufficient": len(issues) == 0,
            "issues": issues,
            "flippable_positions": total_positions,
            "pairing_rule_violations": [issue for issue in issues if "pairing" in issue.lower()]
        }
    
    def _analyze_step_budget(self, cards: List[List[str]], symbol_pairs: Dict[str, str], max_steps: int, total_pairs: int) -> Dict[str, Any]:
        """Analyze if the step budget is sufficient for solving the puzzle."""
        issues = []
        
        # Calculate minimum steps required for optimal play
        # Best case: agent needs to flip exactly 2 cards per pair, no wasted flips
        min_steps_optimal = total_pairs * 2  # 8 pairs * 2 flips = 16 steps
        
        # Realistic case: account for exploration and memory limitations
        # Agent needs to discover pairing relationships, may need to re-flip cards
        total_symbols = len(set(symbol for row in cards for symbol in row))
        exploration_buffer = total_symbols * 2  # Need to see each symbol at least once, maybe twice
        memory_overhead = total_pairs * 1  # Additional flips due to memory constraints
        
        realistic_min_steps = min_steps_optimal + exploration_buffer + memory_overhead
        
        if max_steps < min_steps_optimal:
            issues.append(f"Step budget {max_steps} insufficient even for optimal play (needs {min_steps_optimal})")
        elif max_steps < realistic_min_steps:
            self.validation_warnings.append(f"Step budget {max_steps} may be tight for realistic play (recommended: {realistic_min_steps})")
        
        # Check for reasonable upper bound (not too easy)
        generous_budget = total_pairs * 5  # Very generous allowance
        if max_steps > generous_budget:
            self.validation_warnings.append(f"Step budget {max_steps} may be too generous (could reduce challenge)")
        
        return {
            "budget_sufficient": len(issues) == 0,
            "issues": issues,
            "min_steps_optimal": min_steps_optimal,
            "realistic_min_steps": realistic_min_steps,
            "budget_utilization": max_steps / realistic_min_steps if realistic_min_steps > 0 else float('inf')
        }
    
    def _validate_pairing_consistency(self, cards: List[List[str]], symbol_pairs: Dict[str, str]) -> Dict[str, Any]:
        """Validate that the pairing rule is consistently applied and creates solvable pairs."""
        issues = []
        
        # Build position map for each symbol
        symbol_positions = {}
        for i, row in enumerate(cards):
            for j, symbol in enumerate(row):
                if symbol not in symbol_positions:
                    symbol_positions[symbol] = []
                symbol_positions[symbol].append((i, j))
        
        # Verify each symbol's partner relationship
        verified_pairs = set()
        
        for symbol, partner in symbol_pairs.items():
            if symbol == partner:
                issues.append(f"Invalid self-pairing: {symbol} -> {symbol}")
                continue
            
            # Check if partner exists in symbol_pairs
            if partner not in symbol_pairs:
                issues.append(f"Symbol '{symbol}' pairs with '{partner}', but '{partner}' not in pairing map")
                continue
            
            # Check bidirectional consistency
            if symbol_pairs[partner] != symbol:
                issues.append(f"Asymmetric pairing: {symbol}<->{partner} vs {partner}<->{symbol_pairs[partner]}")
                continue
            
            # Verify both symbols appear on board
            if symbol not in symbol_positions:
                issues.append(f"Symbol '{symbol}' in pairing but not on board")
                continue
            if partner not in symbol_positions:
                issues.append(f"Partner symbol '{partner}' not on board")
                continue
            
            # Verify each symbol appears exactly twice
            if len(symbol_positions[symbol]) != 2:
                issues.append(f"Symbol '{symbol}' appears {len(symbol_positions[symbol])} times, expected 2")
            if len(symbol_positions[partner]) != 2:
                issues.append(f"Symbol '{partner}' appears {len(symbol_positions[partner])} times, expected 2")
            
            # Record this pair as verified (avoid double-checking)
            pair_key = tuple(sorted([symbol, partner]))
            verified_pairs.add(pair_key)
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues,
            "verified_pairs": len(verified_pairs)
        }
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the reward structure promotes good problem-solving behavior."""
        analysis = {
            "well_designed": True,
            "issues": [],
            "reward_balance": {},
            "incentive_alignment": {}
        }
        
        # Analyze reward proportions
        pair_reward = 1.0  # Fixed in environment
        exploration_reward = 0.05  # Fixed in environment
        max_steps = world_state["globals"]["max_steps"]
        total_pairs = world_state["globals"]["total_pairs"]
        
        # Calculate potential rewards
        max_pair_rewards = total_pairs * pair_reward  # 8.0
        max_exploration_rewards = 8 * exploration_reward  # 0.40 (8 unique symbols)
        max_total_reward = max_pair_rewards + max_exploration_rewards  # 8.40
        
        analysis["reward_balance"] = {
            "max_pair_rewards": max_pair_rewards,
            "max_exploration_rewards": max_exploration_rewards,
            "max_total_reward": max_total_reward,
            "pair_reward_ratio": max_pair_rewards / max_total_reward,
            "exploration_reward_ratio": max_exploration_rewards / max_total_reward
        }
        
        # Check for proper incentive alignment
        if max_pair_rewards < max_exploration_rewards:
            analysis["issues"].append("Exploration rewards exceed pair discovery rewards - misaligned incentives")
            analysis["well_designed"] = False
        
        # Verify reasonable reward ratios
        pair_dominance_ratio = max_pair_rewards / max_exploration_rewards
        if pair_dominance_ratio < 10:
            analysis["issues"].append(f"Pair rewards only {pair_dominance_ratio:.1f}x exploration - may encourage grinding")
        
        # Check for action farming potential
        # Maximum possible flips = max_steps, but should be heavily outweighed by goal achievement
        max_possible_exploration = max_steps * exploration_reward  # If every flip revealed new symbol (impossible)
        if max_possible_exploration > max_pair_rewards:
            analysis["issues"].append("Theoretical action farming could outweigh goal achievement")
            analysis["well_designed"] = False
        
        # Efficiency incentive analysis
        analysis["incentive_alignment"] = {
            "rewards_goal_achievement": True,  # Pair discovery is highest reward
            "discourages_grinding": max_pair_rewards > max_exploration_rewards,
            "encourages_efficiency": True,  # No penalties for fewer steps, but limited exploration rewards
            "has_diminishing_returns": True  # Exploration rewards limited to 8 unique symbols
        }
        
        return analysis
    
    def _validate_configuration(self, world_state: Dict[str, Any]) -> bool:
        """Validate configuration consistency and reasonable values."""
        try:
            globals_data = world_state["globals"]
            
            # Check reasonable values
            max_steps = globals_data.get("max_steps", 0)
            if max_steps <= 0 or max_steps > 200:
                self.validation_errors.append(f"Unreasonable max_steps: {max_steps}")
            
            total_pairs = globals_data.get("total_pairs", 0)
            if total_pairs != 4:
                self.validation_errors.append(f"Expected total_pairs to match symbol pair count, got {total_pairs}")
            
            grid_size = globals_data.get("grid_size", 0)
            if grid_size != 4:
                self.validation_errors.append(f"Expected grid_size=4, got {grid_size}")
            
            # Validate initial game state
            game = world_state["game"]
            if game.get("discovered_pairs", -1) != 0:
                self.validation_errors.append("Initial discovered_pairs should be 0")
            
            if game.get("step_count", -1) != 0:
                self.validation_errors.append("Initial step_count should be 0")
            
            if game.get("cumulative_reward", -1) != 0.0:
                self.validation_errors.append("Initial cumulative_reward should be 0.0")
            
            # Validate initial card states (all should be face-down)
            card_states = world_state["board"]["card_states"]
            for i, row in enumerate(card_states):
                for j, state in enumerate(row):
                    if state != 0:
                        self.validation_errors.append(f"Initial card state at ({i},{j}) should be 0, got {state}")
        
        except KeyError as e:
            self.validation_errors.append(f"Missing configuration key: {str(e)}")
        
        return len(self.validation_errors) == 0


def validate_generated_level(world_path: str) -> Dict[str, Any]:
    """
    Main validation function for Mismatched Memory levels.
    
    Args:
        world_path: Path to the generated world YAML file
        
    Returns:
        Comprehensive validation results including solvability analysis
    """
    validator = MismatchedMemoryValidator()
    return validator.validate_level(world_path)


def batch_validate_levels(levels_directory: str) -> Dict[str, Dict[str, Any]]:
    """
    Validate multiple levels in a directory.
    
    Args:
        levels_directory: Directory containing level YAML files
        
    Returns:
        Dictionary mapping level names to validation results
    """
    results = {}
    
    if not os.path.exists(levels_directory):
        return {"error": f"Directory {levels_directory} does not exist"}
    
    for filename in os.listdir(levels_directory):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            level_path = os.path.join(levels_directory, filename)
            level_name = os.path.splitext(filename)[0]
            
            try:
                results[level_name] = validate_generated_level(level_path)
            except Exception as e:
                results[level_name] = {
                    "is_valid": False,
                    "is_solvable": False,
                    "errors": [f"Validation failed: {str(e)}"],
                    "warnings": [],
                    "solvability_analysis": {},
                    "reward_analysis": {}
                }
    
    return results