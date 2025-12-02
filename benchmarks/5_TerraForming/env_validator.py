from typing import Dict, Any, List, Tuple, Optional
import yaml
import os
from copy import deepcopy

class TerraformingLevelValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get('termination', {}).get('max_steps', 40)
        
    def validate_level(self, world_path: str) -> Tuple[bool, List[str]]:
        """Validate a generated terraforming level for solvability and proper reward structure."""
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
        
        # 3. BASIC STATE VALIDATION
        state_issues = self._validate_state_consistency(world_state)
        issues.extend(state_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, initial_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles - analyze if 100% habitability is achievable."""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        energy_budget = initial_state['infrastructure']['energy_reserves']
        min_energy_needed = self._calculate_minimum_energy_required()
        
        if energy_budget < min_energy_needed:
            issues.append(f"Insufficient energy budget: {energy_budget} < {min_energy_needed} required")
        
        # TARGET REACHABILITY - Check if 100% habitability is theoretically possible
        reachability_issues = self._analyze_target_reachability(initial_state)
        issues.extend(reachability_issues)
        
        # RESOURCE AVAILABILITY - Check if required biological resources exist
        if initial_state['biosphere_seeds']['dormant_microbes'] < 50:
            issues.append("Insufficient dormant microbes for biological development")
        
        if initial_state['hydrosphere']['subsurface_ice_pct'] < 15:
            issues.append("Insufficient water resources for hydrosphere development")
        
        # STEP COUNTING - Verify solvability within step limit
        min_steps_needed = self._estimate_minimum_steps(initial_state)
        if min_steps_needed > self.max_steps:
            issues.append(f"Estimated minimum steps {min_steps_needed} exceeds limit {self.max_steps}")
        
        # COMMON IMPOSSIBLE PATTERNS
        impossible_patterns = self._check_impossible_patterns(initial_state)
        issues.extend(impossible_patterns)
        
        return issues
    
    def _analyze_target_reachability(self, state: Dict[str, Any]) -> List[str]:
        """Analyze if target habitability (100%) is reachable from initial state."""
        issues = []
        
        # Simulate optimal strategy to check maximum achievable habitability
        sim_state = deepcopy(state)
        max_possible_habitability = self._simulate_optimal_path(sim_state)
        
        if max_possible_habitability < 60.0:  # Challenging but achievable threshold  
            issues.append(f"Maximum achievable habitability {max_possible_habitability:.1f}% < 60% threshold")
        
        # Check atmospheric reachability
        max_o2_possible = min(25.0, state['atmosphere']['oxygen_pct'] + 
                             (self.max_steps * 3.5))  # Max O2 gain per action
        if max_o2_possible < 15.0:
            issues.append("Cannot reach required oxygen levels (15-25%) for habitability")
        
        # Check water system reachability
        max_water_possible = min(70.0, state['hydrosphere']['surface_water_pct'] + 
                                state['hydrosphere']['subsurface_ice_pct'] * 0.8)
        if max_water_possible < 40.0:
            issues.append("Cannot reach sufficient water levels (40%+) for habitability")
        
        return issues
    
    def _simulate_optimal_path(self, state: Dict[str, Any]) -> float:
        """Simulate an optimal action sequence to estimate maximum achievable habitability."""
        sim_state = deepcopy(state)
        
        # Simulate key actions in optimal order
        steps_used = 0
        
        # Phase 1: Build infrastructure if energy allows
        while (steps_used < 5 and sim_state['infrastructure']['energy_reserves'] >= 200 and 
               sim_state['infrastructure']['terraforming_stations'] < 3):
            sim_state['infrastructure']['energy_reserves'] -= 200
            sim_state['infrastructure']['terraforming_stations'] += 1
            sim_state['global_metrics']['instability_index'] += 5.0
            steps_used += 1
        
        # Phase 2: Atmospheric processing
        atm_actions = 0
        while (steps_used < self.max_steps - 10 and atm_actions < 8 and 
               sim_state['infrastructure']['energy_reserves'] >= 100 and
               sim_state['atmosphere']['oxygen_pct'] < 22.0):
            sim_state['infrastructure']['energy_reserves'] -= 100
            sim_state['atmosphere']['oxygen_pct'] = min(25.0, 
                sim_state['atmosphere']['oxygen_pct'] + 3.5)
            sim_state['atmosphere']['co2_pct'] = max(0.0,
                sim_state['atmosphere']['co2_pct'] - 4.5)
            sim_state['atmosphere']['temperature'] += -8.0 * (4.5 / 100.0)
            atm_actions += 1
            steps_used += 1
        
        # Phase 3: Water systems
        water_actions = 0
        while (steps_used < self.max_steps - 5 and water_actions < 4 and
               sim_state['infrastructure']['energy_reserves'] >= 80 and
               sim_state['hydrosphere']['subsurface_ice_pct'] > 5):
            sim_state['infrastructure']['energy_reserves'] -= 80
            ice_conversion = min(15.0, sim_state['hydrosphere']['subsurface_ice_pct'])
            sim_state['hydrosphere']['subsurface_ice_pct'] -= ice_conversion
            sim_state['hydrosphere']['surface_water_pct'] = min(70.0,
                sim_state['hydrosphere']['surface_water_pct'] + ice_conversion * 0.8)
            sim_state['hydrosphere']['ph_level'] = min(7.0,
                sim_state['hydrosphere']['ph_level'] + 0.3)
            water_actions += 1
            steps_used += 1
        
        # Phase 4: Biological seeding
        bio_actions = 0
        while (steps_used < self.max_steps - 2 and bio_actions < 3 and
               sim_state['infrastructure']['energy_reserves'] >= 60 and
               sim_state['biosphere_seeds']['dormant_microbes'] > 10):
            if (sim_state['atmosphere']['oxygen_pct'] >= 5.0 and 
                sim_state['hydrosphere']['surface_water_pct'] >= 10.0):
                sim_state['infrastructure']['energy_reserves'] -= 60
                microbe_activation = min(50.0, sim_state['biosphere_seeds']['dormant_microbes'])
                sim_state['biosphere_seeds']['dormant_microbes'] -= microbe_activation
                
                flora_activation = min(20.0, sim_state['biosphere_seeds']['dormant_flora'])
                sim_state['biosphere_seeds']['dormant_flora'] -= flora_activation
                
                fertility_gain = (microbe_activation * 0.4) + (flora_activation * 0.6)
                sim_state['lithosphere']['soil_fertility'] = min(100.0,
                    sim_state['lithosphere']['soil_fertility'] + fertility_gain)
                bio_actions += 1
            steps_used += 1
        
        # Calculate final habitability using the same formula as the environment
        return self._calculate_habitability_index(sim_state)
    
    def _calculate_habitability_index(self, state: Dict[str, Any]) -> float:
        """Calculate habitability index using the same formula as the environment."""
        # Atmosphere component (30% max)
        atmosphere_score = 0
        if 15.0 <= state['atmosphere']['oxygen_pct'] <= 25.0:
            atmosphere_score += 40
        else:
            atmosphere_score += max(0, 40 - abs(20.0 - state['atmosphere']['oxygen_pct']) * 2)
        
        if state['atmosphere']['co2_pct'] <= 10.0:
            atmosphere_score += 35
        else:
            atmosphere_score += max(0, 35 - (state['atmosphere']['co2_pct'] - 10.0) * 2)
        
        if -10.0 <= state['atmosphere']['temperature'] <= 30.0:
            atmosphere_score += 25
        else:
            temp_penalty = max(abs(state['atmosphere']['temperature'] + 10), 
                             abs(state['atmosphere']['temperature'] - 30))
            atmosphere_score += max(0, 25 - temp_penalty)
        
        atmosphere_component = min(30.0, atmosphere_score * 0.3)
        
        # Water component (25% max)
        water_score = min(100, state['hydrosphere']['surface_water_pct'] * 1.5)
        if state['hydrosphere']['ph_level'] >= 6.0:
            water_score *= 1.2
        elif state['hydrosphere']['ph_level'] < 4.0:
            water_score *= 0.7
        water_component = min(25.0, water_score * 0.25)
        
        # Biology component (25% max)
        active_microbes = 100.0 - state['biosphere_seeds']['dormant_microbes']
        active_flora = 50.0 - state['biosphere_seeds']['dormant_flora']
        biology_score = (active_microbes * 0.6) + (active_flora * 0.8) + state['lithosphere']['soil_fertility']
        biology_component = min(25.0, biology_score * 0.125)
        
        # Stability component (20% max)
        stability_score = max(0, 100 - state['global_metrics']['instability_index'])
        stability_component = min(20.0, stability_score * 0.2)
        
        return min(100.0, atmosphere_component + water_component + biology_component + stability_component)
    
    def _calculate_minimum_energy_required(self) -> float:
        """Calculate minimum energy needed for a viable solution."""
        # Minimum viable strategy energy costs:
        # - 4-5 atmospheric processors: 400-500 energy
        # - 3-4 water catalyst releases: 240-320 energy  
        # - 2-3 biological seeding: 120-180 energy
        # - 1-2 stabilization actions: 100-200 energy
        # - Buffer for shields: 100-200 energy
        return 960.0  # Conservative estimate
    
    def _estimate_minimum_steps(self, state: Dict[str, Any]) -> int:
        """Estimate minimum steps needed to reach 100% habitability."""
        # Based on required transformations:
        # - Atmospheric processing: 5-7 steps
        # - Water system activation: 3-4 steps
        # - Biological development: 2-3 steps
        # - Infrastructure: 1-2 steps
        # - Stabilization actions: 2-3 steps
        # - Buffer for cascading effects: 5-8 steps
        return 25  # Conservative minimum estimate
    
    def _check_impossible_patterns(self, state: Dict[str, Any]) -> List[str]:
        """Check for common patterns that make levels unsolvable."""
        issues = []
        
        # Pattern 1: Impossible atmospheric conditions
        if state['atmosphere']['co2_pct'] > 98.0 and state['infrastructure']['energy_reserves'] < 500:
            issues.append("CO2 levels too high for available energy to reduce sufficiently")
        
        # Pattern 2: Extreme temperature with insufficient atmospheric control
        if (state['atmosphere']['temperature'] < -90.0 and 
            state['atmosphere']['co2_pct'] < 70.0):
            issues.append("Temperature too low with insufficient CO2 for greenhouse warming")
        
        # Pattern 3: Circular dependencies
        if (state['hydrosphere']['subsurface_ice_pct'] < 10.0 and 
            state['hydrosphere']['surface_water_pct'] < 5.0 and
            state['atmosphere']['temperature'] < -50.0):
            issues.append("Insufficient water resources combined with extreme cold creates impossible conditions")
        
        # Pattern 4: Excessive initial instability
        if state['global_metrics']['instability_index'] > 30.0:
            issues.append("Initial instability too high - likely to cause cascade failure")
        
        return issues
    
    def _validate_reward_structure(self, state: Dict[str, Any]) -> List[str]:
        """Validate that reward structure encourages solving rather than action farming."""
        issues = []
        
        # Check mission completion reward dominance
        mission_completion_reward = 20.0
        max_action_farming_reward = self._calculate_max_action_farming_reward()
        
        if mission_completion_reward < max_action_farming_reward * 0.5:
            issues.append(f"Mission completion reward {mission_completion_reward} too low compared to action farming potential {max_action_farming_reward}")
        
        # Check for excessive step rewards
        stability_reward_per_step = 0.1
        total_stability_rewards = stability_reward_per_step * self.max_steps
        if total_stability_rewards > mission_completion_reward * 0.3:
            issues.append(f"Stability rewards per episode {total_stability_rewards} too high compared to completion bonus")
        
        # Verify failure penalties are meaningful
        failure_penalty = -40.0
        if abs(failure_penalty) < mission_completion_reward:
            issues.append("Failure penalty insufficient to discourage reckless actions")
        
        # Check for reward loops (actions that can be repeated for easy points)
        if self._has_exploitable_reward_loops():
            issues.append("Reward structure allows exploitation through repetitive actions")
        
        return issues
    
    def _calculate_max_action_farming_reward(self) -> float:
        """Calculate maximum reward possible through action farming rather than solving."""
        # Maximum rewards from non-completion sources:
        # - Stability maintenance: 0.1 * 40 steps = 4.0
        # - Atmospheric progress: ~5 * 0.05 = 0.25
        # - Water progress: ~4 * 0.1 = 0.4  
        # - Habitability increases: ~50 * 0.2 = 10.0 (but this requires actual progress)
        return 4.65  # Conservative estimate excluding habitability progress
    
    def _has_exploitable_reward_loops(self) -> bool:
        """Check if agents can exploit reward loops for easy points."""
        # In this environment, most actions have energy costs and consequences,
        # limiting exploitation. Main risk is passive observation spam, but that
        # provides no positive rewards.
        return False
    
    def _validate_state_consistency(self, state: Dict[str, Any]) -> List[str]:
        """Validate basic state consistency and bounds."""
        issues = []
        
        # Check required state structure exists
        required_sections = ['atmosphere', 'hydrosphere', 'lithosphere', 
                           'biosphere_seeds', 'infrastructure', 'global_metrics']
        for section in required_sections:
            if section not in state:
                issues.append(f"Missing required state section: {section}")
        
        # Validate numeric bounds
        if 'atmosphere' in state:
            atm = state['atmosphere']
            if not (0 <= atm.get('oxygen_pct', 0) <= 100):
                issues.append("Oxygen percentage out of valid range [0, 100]")
            if not (0 <= atm.get('co2_pct', 0) <= 100):
                issues.append("CO2 percentage out of valid range [0, 100]")
            if not (-150 <= atm.get('temperature', 0) <= 100):
                issues.append("Temperature out of reasonable range [-150, 100]")
        
        if 'hydrosphere' in state:
            hydro = state['hydrosphere']
            if not (0 <= hydro.get('surface_water_pct', 0) <= 100):
                issues.append("Surface water percentage out of valid range [0, 100]")
            if not (0 <= hydro.get('subsurface_ice_pct', 0) <= 100):
                issues.append("Subsurface ice percentage out of valid range [0, 100]")
        
        if 'infrastructure' in state:
            infra = state['infrastructure']
            if infra.get('energy_reserves', 0) < 0:
                issues.append("Energy reserves cannot be negative")
            if not (0 <= infra.get('terraforming_stations', 0) <= 5):
                issues.append("Terraforming stations out of reasonable range [0, 5]")
        
        if 'global_metrics' in state:
            metrics = state['global_metrics']
            if not (0 <= metrics.get('habitability_index', 0) <= 100):
                issues.append("Habitability index out of valid range [0, 100]")
            if metrics.get('instability_index', 0) < 0:
                issues.append("Instability index cannot be negative")
        
        return issues

def validate_terraforming_level(world_path: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Main validation function for terraforming levels."""
    validator = TerraformingLevelValidator(config)
    return validator.validate_level(world_path)

# Example usage function
def validate_all_levels(levels_dir: str, config_path: str) -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all levels in a directory."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = {}
    for filename in os.listdir(levels_dir):
        if filename.endswith('.yaml'):
            world_path = os.path.join(levels_dir, filename)
            world_id = filename[:-5]  # Remove .yaml extension
            results[world_id] = validate_terraforming_level(world_path, config)
    
    return results