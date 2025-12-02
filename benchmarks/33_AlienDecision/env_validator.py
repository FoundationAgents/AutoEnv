from typing import Dict, Any, List, Tuple, Optional
import yaml
import copy

class AlienColonyValidator:
    """Validator for Alien Colony Environment levels"""
    
    def __init__(self, config_path: str = "./config.yaml"):
        # Load configuration to get actual reward values
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.semantic_mappings = {
            'resources': {
                'toxic_waste': 'nutrition_boost',
                'rotten_food': 'health_boost', 
                'contaminated_water': 'happiness_boost',
                'broken_machinery': 'efficiency_boost',
                'poisonous_plants': 'growth_stimulant',
                'radioactive_ore': 'energy_source',
                'infected_samples': 'medical_supplies',
                'corroded_metal': 'construction_material'
            },
            'buildings': {
                'decay_chamber': 'population_accelerator',
                'poison_distributor': 'happiness_generator',
                'waste_processor': 'resource_multiplier',
                'contamination_center': 'research_facility',
                'corruption_hub': 'coordination_center'
            }
        }
    
    def validate_level(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Main validation function that checks level solvability and reward structure
        Returns: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check level solvability
        solvability_issues = self._check_level_solvability(state)
        issues.extend(solvability_issues)
        
        # Check reward structure
        reward_issues = self._check_reward_structure(state)
        issues.extend(reward_issues)
        
        # Check configuration validity
        config_issues = self._check_configuration_validity(state)
        issues.extend(config_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, state: Dict[str, Any]) -> List[str]:
        """Check if the level is actually solvable"""
        issues = []
        
        # Action constraint analysis
        action_issues = self._analyze_action_constraints(state)
        issues.extend(action_issues)
        
        # Target reachability analysis
        reachability_issues = self._analyze_target_reachability(state)
        issues.extend(reachability_issues)
        
        # Resource availability check
        resource_issues = self._check_resource_availability(state)
        issues.extend(resource_issues)
        
        # Step budget analysis
        step_issues = self._check_step_feasibility(state)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, state: Dict[str, Any]) -> List[str]:
        """Analyze if actions have sufficient power to reach objectives"""
        issues = []
        
        # Check if there are enough resources to potentially grow population
        total_nutrition_potential = 0
        for resource, amount in state['resources'].items():
            true_effect = self.semantic_mappings['resources'].get(resource)
            if true_effect in ['nutrition_boost', 'growth_stimulant']:
                total_nutrition_potential += amount
        
        # Add explorable resources
        if '_hidden' in state and 'explorable_resources' in state['_hidden']:
            for resource, info in state['_hidden']['explorable_resources'].items():
                true_effect = self.semantic_mappings['resources'].get(resource)
                if true_effect in ['nutrition_boost', 'growth_stimulant']:
                    total_nutrition_potential += info['initial_amount']
        
        # Check if we can gather enough resources through actions
        max_steps = state['globals']['max_steps']
        max_gatherable = max_steps * 10  # Max gather per action is 10
        
        current_pop = state['colony']['population']
        target_pop = state['globals']['target_population']
        pop_needed = target_pop - current_pop
        
        if total_nutrition_potential + max_gatherable < pop_needed * 0.5:  # Conservative estimate
            issues.append(f"Insufficient nutrition resources available. Need ~{pop_needed} population growth but only {total_nutrition_potential + max_gatherable} potential nutrition available")
        
        # Check building construction constraints
        grid_size = state['grid']['size'][0] * state['grid']['size'][1]
        occupied_positions = len(state['grid']['occupied'])
        available_positions = grid_size - occupied_positions
        
        if available_positions < 2:  # Need some space for expansion
            issues.append("Grid too crowded - insufficient space for building construction")
        
        return issues
    
    def _analyze_target_reachability(self, state: Dict[str, Any]) -> List[str]:
        """Check if target population is reachable from initial state"""
        issues = []
        
        current_pop = state['colony']['population']
        target_pop = state['globals']['target_population']
        min_pop = state['globals']['min_population']
        
        # Basic sanity checks
        if target_pop <= current_pop:
            issues.append(f"Target population {target_pop} is not greater than initial population {current_pop}")
        
        if current_pop <= min_pop:
            issues.append(f"Initial population {current_pop} is at or below failure threshold {min_pop}")
        
        # Check if population growth is theoretically possible
        population_boosting_resources = []
        for resource, amount in state['resources'].items():
            true_effect = self.semantic_mappings['resources'].get(resource)
            if true_effect in ['nutrition_boost', 'growth_stimulant']:
                population_boosting_resources.append((resource, amount))
        
        # Check for population-boosting buildings
        population_boosting_buildings = []
        for building in state['buildings']:
            true_effect = self.semantic_mappings['buildings'].get(building['type'])
            if true_effect in ['population_accelerator']:
                population_boosting_buildings.append(building['type'])
        
        if not population_boosting_resources and not population_boosting_buildings:
            # Check if we can discover resources through exploration
            explorable_nutrition = False
            if '_hidden' in state and 'explorable_resources' in state['_hidden']:
                for resource, info in state['_hidden']['explorable_resources'].items():
                    true_effect = self.semantic_mappings['resources'].get(resource)
                    if true_effect in ['nutrition_boost', 'growth_stimulant']:
                        explorable_nutrition = True
                        break
            
            if not explorable_nutrition:
                issues.append("No population-boosting resources or buildings available, and none discoverable through exploration")
        
        return issues
    
    def _check_resource_availability(self, state: Dict[str, Any]) -> List[str]:
        """Check if required resources are available or obtainable"""
        issues = []
        
        # Check initial resource quantities
        for resource, amount in state['resources'].items():
            if amount < 0:
                issues.append(f"Invalid negative resource amount: {resource} = {amount}")
            if amount > 1000:  # Reasonable upper bound
                issues.append(f"Suspiciously high resource amount: {resource} = {amount}")
        
        # Check if happiness can be maintained
        happiness_resources = []
        for resource, amount in state['resources'].items():
            true_effect = self.semantic_mappings['resources'].get(resource)
            if true_effect in ['happiness_boost', 'health_boost']:
                happiness_resources.append((resource, amount))
        
        current_happiness = state['colony']['happiness']
        if current_happiness < 30 and not happiness_resources:
            issues.append("Low initial happiness with no happiness-boosting resources available")
        
        # Verify explorable resources are reasonable
        if '_hidden' in state and 'explorable_resources' in state['_hidden']:
            for resource, info in state['_hidden']['explorable_resources'].items():
                req_level = info.get('required_exploration_level', 0)
                if req_level > 20:  # Unreasonably high exploration requirement
                    issues.append(f"Explorable resource {resource} requires unreachable exploration level {req_level}")
                
                initial_amount = info.get('initial_amount', 0)
                if initial_amount <= 0:
                    issues.append(f"Explorable resource {resource} has invalid initial amount {initial_amount}")
        
        return issues
    
    def _check_step_feasibility(self, state: Dict[str, Any]) -> List[str]:
        """Check if solution is achievable within step limits"""
        issues = []
        
        max_steps = state['globals']['max_steps']
        current_pop = state['colony']['population']
        target_pop = state['globals']['target_population']
        pop_growth_needed = target_pop - current_pop
        
        # Minimum steps analysis
        # Need steps for: resource gathering, allocation, possibly exploration, building
        min_steps_needed = 0
        
        # Exploration steps (if needed to unlock resources)
        max_exploration_required = 0
        if '_hidden' in state and 'explorable_resources' in state['_hidden']:
            for resource, info in state['_hidden']['explorable_resources'].items():
                req_level = info.get('required_exploration_level', 0)
                max_exploration_required = max(max_exploration_required, req_level)
        
        current_exploration = state['colony']['area_exploration']
        exploration_steps_needed = max(0, max_exploration_required - current_exploration)
        min_steps_needed += exploration_steps_needed
        
        # Resource allocation steps (conservative estimate)
        min_allocation_steps = max(1, pop_growth_needed // 10)  # Assume 10 pop per efficient allocation
        min_steps_needed += min_allocation_steps
        
        # Building construction steps (if beneficial)
        min_steps_needed += 2  # At least some building actions
        
        if min_steps_needed > max_steps:
            issues.append(f"Minimum estimated steps ({min_steps_needed}) exceeds maximum steps ({max_steps})")
        
        # Check if step limit is reasonable (not too generous either)
        if max_steps > 100:
            issues.append(f"Maximum steps ({max_steps}) is unusually high and may allow reward exploitation")
        
        return issues
    
    def _check_reward_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate reward structure design"""
        issues = []
        
        # Get actual reward values from config
        discovery_reward = self.config['reward']['discovery_rewards']['resource_effect']  # Should be 0.05
        goal_reward = self.config['reward']['goal_rewards']['population_target']  # Should be 1.5
        
        max_discoveries = 8  # Max possible resource + building discoveries
        max_discovery_rewards = discovery_reward * max_discoveries
        
        if max_discovery_rewards >= goal_reward:
            issues.append(f"Discovery rewards ({max_discovery_rewards}) can exceed goal achievement reward ({goal_reward})")
        
        # Check for action grinding potential
        max_steps = state['globals']['max_steps']
        if max_steps > 50:
            issues.append("High step limit may allow agents to exploit action repetition for rewards")
        
        # Verify that goal achievement is the primary reward
        total_max_reward = goal_reward + max_discovery_rewards
        goal_proportion = goal_reward / total_max_reward
        
        if goal_proportion < 0.6:  # Goal should be majority of reward
            issues.append(f"Goal achievement reward proportion ({goal_proportion:.2f}) is too low - should be primary reward source")
        
        return issues
    
    def _check_configuration_validity(self, state: Dict[str, Any]) -> List[str]:
        """Check basic configuration validity"""
        issues = []
        
        # Required fields check
        required_fields = ['globals', 'colony', 'resources', 'buildings', 'environment', 'grid']
        for field in required_fields:
            if field not in state:
                issues.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'colony' in state:
            pop = state['colony'].get('population', 0)
            happiness = state['colony'].get('happiness', 0)
            
            if pop <= 0:
                issues.append(f"Invalid population: {pop}")
            if happiness < 0 or happiness > 100:
                issues.append(f"Invalid happiness level: {happiness}")
        
        if 'globals' in state:
            target_pop = state['globals'].get('target_population', 0)
            min_pop = state['globals'].get('min_population', 0)
            max_steps = state['globals'].get('max_steps', 0)
            
            if target_pop <= 0:
                issues.append(f"Invalid target population: {target_pop}")
            if min_pop >= target_pop:
                issues.append(f"Minimum population ({min_pop}) should be less than target ({target_pop})")
            if max_steps <= 0:
                issues.append(f"Invalid max steps: {max_steps}")
        
        # Grid validation
        if 'grid' in state:
            grid_size = state['grid'].get('size', [0, 0])
            occupied = state['grid'].get('occupied', [])
            
            if len(grid_size) != 2 or grid_size[0] <= 0 or grid_size[1] <= 0:
                issues.append(f"Invalid grid size: {grid_size}")
            
            # Check occupied positions are within grid
            for pos in occupied:
                if len(pos) != 2 or pos[0] < 0 or pos[1] < 0 or pos[0] >= grid_size[0] or pos[1] >= grid_size[1]:
                    issues.append(f"Occupied position {pos} is outside grid bounds {grid_size}")
        
        # Validate semantic consistency
        if '_hidden' in state:
            resource_mappings = state['_hidden'].get('resource_mappings', {})
            for resource in state.get('resources', {}):
                if resource not in resource_mappings:
                    issues.append(f"Resource {resource} missing from semantic mappings")
        
        return issues

def validate_generated_level(level_file_path: str) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a generated level file
    """
    try:
        with open(level_file_path, 'r') as f:
            state = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load level file: {str(e)}"]
    
    validator = AlienColonyValidator()
    return validator.validate_level(state)

# Example usage function for testing
def example_validation():
    """Example of how to use the validator"""
    
    # Sample state for testing
    sample_state = {
        'globals': {'max_steps': 50, 'target_population': 50, 'min_population': 5},
        'colony': {'population': 10, 'happiness': 60, 'area_exploration': 1},
        'resources': {'toxic_waste': 15, 'rotten_food': 12, 'contaminated_water': 8},
        'buildings': [{'type': 'decay_chamber', 'location': [2, 2], 'operational': True}],
        'environment': {'season': 'harsh_season', 'weather': 'corrosive_weather'},
        'grid': {'size': [5, 5], 'occupied': [[2, 2]]},
        'discovery': {'resource_effects_found': [], 'building_effects_found': []},
        '_hidden': {
            'resource_mappings': {'toxic_waste': 'nutrition_boost', 'rotten_food': 'health_boost', 'contaminated_water': 'happiness_boost'},
            'building_mappings': {'decay_chamber': 'population_accelerator'},
            'explorable_resources': {'poisonous_plants': {'required_exploration_level': 3, 'initial_amount': 10}}
        }
    }
    
    validator = AlienColonyValidator()
    is_valid, issues = validator.validate_level(sample_state)
    
    print(f"Level valid: {is_valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_valid, issues
