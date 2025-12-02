from typing import Dict, Any, List, Tuple, Optional
import yaml
import random
from copy import deepcopy

class SubterraneanMegacityValidator:
    def __init__(self):
        self.max_steps = 40
        self.required_districts = 6
        self.min_structural_integrity = 80
        self.min_breathable_air = 60
    
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a generated level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(level_path, 'r') as f:
                level_data = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level file: {str(e)}"]
        
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(level_data)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(level_data)
        issues.extend(reward_issues)
        
        # 3. BASIC STRUCTURE VALIDATION
        structure_issues = self._validate_basic_structure(level_data)
        issues.extend(structure_issues)
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, level_data: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        constraint_issues = self._analyze_action_constraints(level_data)
        issues.extend(constraint_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(level_data)
        issues.extend(reachability_issues)
        
        # RESOURCE AVAILABILITY
        resource_issues = self._check_resource_availability(level_data)
        issues.extend(resource_issues)
        
        # STEP BUDGET ANALYSIS
        step_issues = self._check_step_budget_feasibility(level_data)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, level_data: Dict[str, Any]) -> List[str]:
        """Understand environment's fundamental limitations"""
        issues = []
        
        grid_size = level_data['grid']['size']
        total_cells = grid_size[0] * grid_size[1]
        cells = level_data['grid']['cells']
        
        # Check if there are any pre-excavated cells for building districts
        excavated_cells = sum(cells['excavated'])
        if excavated_cells == 0:
            issues.append("SOLVABILITY: No pre-excavated cells available. Agent cannot build districts without excavation capability being demonstrated.")
        
        # Check power storage vs requirements
        initial_power = level_data['agent']['power_storage']
        if initial_power < 20:  # Minimum needed for basic operations
            issues.append("SOLVABILITY: Initial power storage too low. Agent may run out of power before establishing power generation.")
        
        # Check if stress levels are so high that excavation becomes impossible
        high_stress_cells = sum(1 for stress in cells['rock_stress'] if stress > 95)
        if high_stress_cells > total_cells * 0.8:
            issues.append("SOLVABILITY: Too many high-stress cells (>95). Excavation may be blocked or extremely costly.")
        
        return issues
    
    def _check_target_reachability(self, level_data: Dict[str, Any]) -> List[str]:
        """Verify target state is actually achievable"""
        issues = []
        
        grid_size = level_data['grid']['size']
        total_cells = grid_size[0] * grid_size[1]
        cells = level_data['grid']['cells']
        
        # TARGET: Build 6 districts with structural integrity >80% and air >60%
        
        # Check if there's enough space for 6 districts
        if total_cells < 6:
            issues.append("SOLVABILITY: Grid too small for 6 districts. Need at least 6 cells.")
        
        # Simulate basic resource requirements for 6 districts
        districts_needed = self.required_districts
        
        # Each district needs: excavation + support + power conduit
        excavation_power_cost = districts_needed * 2  # 2 power per excavation
        support_power_cost = districts_needed * 1    # 1 power per support
        conduit_power_cost = districts_needed * 1    # 1 power per conduit cell
        district_power_cost = districts_needed * 5   # 5 power per district core
        
        total_min_power_needed = excavation_power_cost + support_power_cost + conduit_power_cost + district_power_cost
        
        # Check if power can be generated through conduit placement
        initial_power = level_data['agent']['power_storage']
        max_possible_power_generation = 25 * 5  # 25 cells max, 5 power per conduit
        total_available_power = initial_power + max_possible_power_generation
        
        if total_available_power < total_min_power_needed:
            issues.append(f"SOLVABILITY: Insufficient power generation potential. Need {total_min_power_needed}, can generate max {total_available_power}")
        
        # Check structural integrity maintenance
        # High stress areas might make it impossible to maintain >80% integrity
        avg_stress = sum(cells['rock_stress']) / len(cells['rock_stress'])
        if avg_stress > 85:
            issues.append("SOLVABILITY: Average stress too high. May be impossible to maintain 80% structural integrity even with supports.")
        
        return issues
    
    def _check_resource_availability(self, level_data: Dict[str, Any]) -> List[str]:
        """Check if all required resources are present or obtainable"""
        issues = []
        
        available_materials = level_data['agent']['available_materials']
        
        # Check basic materials needed for district construction
        required_materials = ['basic_support', 'rock_excavator']  # Minimal requirements
        
        for material in required_materials:
            if material not in available_materials:
                issues.append(f"SOLVABILITY: Required material '{material}' not available initially and may not be researchable within step limit.")
        
        # Check research unlockability
        research_state = level_data['research']
        
        # All research should start locked to provide progression
        locked_research = sum(1 for unlocked in research_state.values() if not unlocked)
        if locked_research == 0:
            issues.append("PROGRESSION: All research already unlocked. No progression incentive.")
        
        return issues
    
    def _check_step_budget_feasibility(self, level_data: Dict[str, Any]) -> List[str]:
        """Check if target is reachable within max_steps limit"""
        issues = []
        
        # Minimum actions needed for 6 districts:
        # - 6 excavations (if not pre-excavated)
        # - 6 support placements  
        # - 6 power conduit installations
        # - 6 district core constructions
        # - Some research actions (at least 1-2 for advanced materials)
        # - Some diagnostic scans for optimization
        
        pre_excavated = sum(level_data['grid']['cells']['excavated'])
        excavations_needed = max(0, self.required_districts - pre_excavated)
        
        min_actions_needed = (
            excavations_needed +  # excavations
            self.required_districts +  # support columns
            self.required_districts +  # power conduits  
            self.required_districts +  # district cores
            2  # minimum research actions
        )
        
        if min_actions_needed > self.max_steps:
            issues.append(f"SOLVABILITY: Minimum actions needed ({min_actions_needed}) exceeds step limit ({self.max_steps})")
        
        # Check if there's reasonable buffer for optimization steps
        optimization_buffer = self.max_steps - min_actions_needed
        if optimization_buffer < 10:
            issues.append(f"SOLVABILITY: Very tight step budget. Only {optimization_buffer} steps for optimization and error recovery.")
        
        return issues
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Simulate reward earning potential through different strategies
        
        # STRATEGY 1: Achievement-focused (solving the actual problem)
        achievement_reward = self._calculate_achievement_reward_potential()
        
        # STRATEGY 2: Action grinding (repetitive actions without progress)
        grinding_reward = self._calculate_grinding_reward_potential()
        
        # STRATEGY 3: Exploration loops (repeated observations)
        exploration_reward = self._calculate_exploration_reward_potential()
        
        # Check for incentive misalignment
        if grinding_reward >= achievement_reward * 0.5:
            issues.append("REWARD: Action grinding potentially more rewarding than problem-solving. Grinding can earn 50%+ of achievement rewards.")
        
        if exploration_reward >= achievement_reward * 0.3:
            issues.append("REWARD: Exploration loops too rewarding compared to actual progress. May encourage meaningless actions.")
        
        # Check reward sparsity vs density balance
        continuous_reward_per_step = 0.5  # operation bonus
        max_continuous_reward = continuous_reward_per_step * self.max_steps
        target_achievement_reward = 40.0  # completion bonus
        
        if max_continuous_reward >= target_achievement_reward:
            issues.append("REWARD: Continuous operation rewards too dense. May overshadow achievement rewards.")
        
        # Check penalty severity
        max_penalty = 40.0  # catastrophic failure
        max_positive = achievement_reward
        
        if max_penalty > max_positive:
            # This is actually good - failure should be costly
            pass
        
        return issues
    
    def _calculate_achievement_reward_potential(self) -> float:
        """Calculate maximum reward for actually solving the problem"""
        reward = 0.0
        
        # Mission complete bonus
        reward += 40.0
        
        # District completion bonuses (6 districts * 5 points each)
        reward += 6 * 5.0
        
        # Research breakthroughs (assume 3 research types * 4 points each)
        reward += 3 * 4.0
        
        # Power milestone bonus
        reward += 3.0
        
        # Structural and air improvement bonuses (conservative estimate)
        reward += 10.0  # Some improvement during construction
        
        # Continuous operation bonus (assume 30 steps above threshold)
        reward += 30 * 0.5
        
        return reward
    
    def _calculate_grinding_reward_potential(self) -> float:
        """Calculate maximum reward from action grinding without progress"""
        reward = 0.0
        
        # If agent could somehow maintain thresholds without progress
        # and just perform diagnostic scans repeatedly
        diagnostic_power_cost = 1
        max_diagnostics = 50  # limited by power
        
        # Continuous operation bonus only
        reward += self.max_steps * 0.5
        
        # No other rewards available without actual progress
        
        return reward
    
    def _calculate_exploration_reward_potential(self) -> float:
        """Calculate reward from pure exploration without achievement"""
        # Similar to grinding - exploration actions like diagnostic_scan
        # don't provide direct rewards, only enable better planning
        
        # Only continuous operation bonus possible
        return self.max_steps * 0.5
    
    def _validate_basic_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Validate basic level structure and data integrity"""
        issues = []
        
        # Check required top-level keys
        required_keys = ['grid', 'agent', 'metrics', 'research', 'physics_state', 'globals']
        for key in required_keys:
            if key not in level_data:
                issues.append(f"STRUCTURE: Missing required key '{key}'")
        
        if 'grid' in level_data:
            grid = level_data['grid']
            
            # Check grid structure
            if 'size' not in grid or len(grid['size']) != 2:
                issues.append("STRUCTURE: Invalid grid size specification")
            else:
                grid_size = grid['size']
                expected_cells = grid_size[0] * grid_size[1]
                
                if 'cells' not in grid:
                    issues.append("STRUCTURE: Missing grid cells")
                else:
                    cells = grid['cells']
                    
                    # Check all cell arrays have correct length
                    required_cell_arrays = [
                        'rock_stress', 'airflow_vector', 'structure_type',
                        'has_support', 'excavated', 'district_core',
                        'power_conduit', 'ventilation_shaft'
                    ]
                    
                    for array_name in required_cell_arrays:
                        if array_name not in cells:
                            issues.append(f"STRUCTURE: Missing cell array '{array_name}'")
                        elif len(cells[array_name]) != expected_cells:
                            issues.append(f"STRUCTURE: Cell array '{array_name}' has wrong length. Expected {expected_cells}, got {len(cells[array_name])}")
        
        # Check metrics are within valid ranges
        if 'metrics' in level_data:
            metrics = level_data['metrics']
            
            if 'structural_integrity' in metrics:
                integrity = metrics['structural_integrity']
                if not (0 <= integrity <= 100):
                    issues.append(f"STRUCTURE: Invalid structural_integrity value: {integrity}. Must be 0-100.")
            
            if 'breathable_air_index' in metrics:
                air_index = metrics['breathable_air_index']
                if not (0 <= air_index <= 100):
                    issues.append(f"STRUCTURE: Invalid breathable_air_index value: {air_index}. Must be 0-100.")
        
        # Check agent state
        if 'agent' in level_data:
            agent = level_data['agent']
            
            if 'districts_built' in agent and agent['districts_built'] < 0:
                issues.append("STRUCTURE: districts_built cannot be negative")
            
            if 'power_storage' in agent and agent['power_storage'] < 0:
                issues.append("STRUCTURE: power_storage cannot be negative")
        
        return issues

def validate_generated_level(level_path: str) -> Tuple[bool, List[str]]:
    """
    Main validation function to be called by the environment system.
    Returns (is_valid, list_of_issues)
    """
    validator = SubterraneanMegacityValidator()
    return validator.validate_level(level_path)

# Additional utility function for batch validation
def validate_multiple_levels(level_paths: List[str]) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate multiple levels and return results for each.
    """
    validator = SubterraneanMegacityValidator()
    results = {}
    
    for path in level_paths:
        results[path] = validator.validate_level(path)
    
    return results