import yaml
import statistics
from typing import Dict, Any, List, Tuple, Optional
import os

class HiveMindLevelValidator:
    """
    Validates generated Hive Mind Consensus levels for solvability and reward structure alignment.
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """
        Main validation entry point.
        Returns: (is_valid, list_of_issues)
        """
        try:
            with open(level_path, 'r') as f:
                level_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level file: {e}"]
        
        issues = []
        
        # 1. Level Solvability Analysis
        solvability_issues = self._check_level_solvability(level_state)
        issues.extend(solvability_issues)
        
        # 2. Reward Structure Validation
        reward_issues = self._validate_reward_structure(level_state)
        issues.extend(reward_issues)
        
        # 3. Basic structural validation
        structure_issues = self._validate_structure(level_state)
        issues.extend(structure_issues)
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for impossible puzzles - ensures levels are actually solvable.
        """
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(level_state)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(level_state)
        issues.extend(reachability_issues)
        
        # RESOURCE AVAILABILITY
        resource_issues = self._check_resource_availability(level_state)
        issues.extend(resource_issues)
        
        # STEP BUDGET ANALYSIS
        step_issues = self._check_step_budget(level_state)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Understand environment's fundamental limitations and action preconditions.
        """
        issues = []
        
        # Check if initial state allows for basic actions
        sub_streams = level_state.get('sub_streams', [])
        
        if len(sub_streams) < 2:
            issues.append("SOLVABILITY: Need at least 2 sub-streams for merge operations")
        
        # Check if streams have reasonable properties for modifications
        for stream in sub_streams:
            coherence = stream.get('coherence', 0)
            novelty = stream.get('novelty', 0)
            size = stream.get('size', 0)
            
            if coherence < 0 or coherence > 100:
                issues.append(f"SOLVABILITY: Stream {stream['id']} has invalid coherence: {coherence}")
            
            if novelty < 0 or novelty > 100:
                issues.append(f"SOLVABILITY: Stream {stream['id']} has invalid novelty: {novelty}")
            
            if size <= 0:
                issues.append(f"SOLVABILITY: Stream {stream['id']} has invalid size: {size}")
        
        # Check if cognitive energy is sufficient for basic operations
        cognitive_energy = level_state.get('globals', {}).get('cognitive_energy', 0)
        if cognitive_energy < 20:
            issues.append("SOLVABILITY: Insufficient initial cognitive energy for meaningful actions")
        
        return issues
    
    def _check_target_reachability(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Verify target state (200 knowledge, unity>=70%, diversity>=50%) is achievable.
        """
        issues = []
        
        globals_state = level_state.get('globals', {})
        sub_streams = level_state.get('sub_streams', [])
        
        initial_unity = globals_state.get('unity', 0)
        initial_diversity = globals_state.get('diversity', 0)
        initial_knowledge = globals_state.get('knowledge_score', 0)
        
        # Check unity reachability
        if initial_unity < 70:
            max_possible_unity = self._calculate_max_possible_unity(sub_streams)
            if max_possible_unity < 70:
                issues.append(f"SOLVABILITY: Maximum achievable unity ({max_possible_unity:.1f}) below required 70%")
        
        # Check diversity reachability  
        if initial_diversity < 50:
            max_possible_diversity = self._calculate_max_possible_diversity(sub_streams)
            if max_possible_diversity < 50:
                issues.append(f"SOLVABILITY: Maximum achievable diversity ({max_possible_diversity:.1f}) below required 50%")
        
        # Check knowledge production potential
        max_knowledge_potential = self._estimate_knowledge_potential(sub_streams)
        if max_knowledge_potential < 200:
            issues.append(f"SOLVABILITY: Estimated max knowledge production ({max_knowledge_potential:.1f}) below target 200")
        
        return issues
    
    def _calculate_max_possible_unity(self, sub_streams: List[Dict[str, Any]]) -> float:
        """
        Calculate theoretical maximum unity achievable through coherence improvements.
        """
        total_size = sum(stream['size'] for stream in sub_streams)
        if total_size == 0:
            return 0
        
        # Assume we can boost coherence to 100 for all streams through meditation
        max_unity = sum(100 * stream['size'] for stream in sub_streams) / total_size
        return max_unity
    
    def _calculate_max_possible_diversity(self, sub_streams: List[Dict[str, Any]]) -> float:
        """
        Calculate theoretical maximum diversity through novelty spread optimization.
        """
        if len(sub_streams) < 2:
            return 0
        
        # Maximum diversity achieved when novelties are maximally spread
        # With n streams, optimal spread is 0, 100/(n-1), 2*100/(n-1), ..., 100
        n = len(sub_streams)
        optimal_novelties = [i * 100 / (n - 1) for i in range(n)]
        variance = statistics.variance(optimal_novelties)
        max_diversity = 100 - variance
        
        return max(0, max_diversity)
    
    def _estimate_knowledge_potential(self, sub_streams: List[Dict[str, Any]]) -> float:
        """
        Estimate maximum knowledge production over 40 steps.
        """
        max_steps = 40
        total_potential = 0
        
        for stream in sub_streams:
            # Maximum per-turn production: size * (100 + 100) / 200 = size
            max_per_turn = stream['size']
            # Over 40 steps
            stream_potential = max_per_turn * max_steps
            # Plus archiving potential (assume 5 archive operations at max novelty)
            archive_potential = 5 * 100 * stream['size'] / 10
            total_potential += stream_potential + archive_potential
        
        return total_potential
    
    def _check_resource_availability(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Check if required resources are available or obtainable.
        """
        issues = []
        
        # Cognitive energy regenerates to 100 each step, so this is always available
        # Main constraint is time and action efficiency
        
        sub_streams = level_state.get('sub_streams', [])
        total_size = sum(stream['size'] for stream in sub_streams)
        
        if abs(total_size - 100.0) > 1.0:
            issues.append(f"SOLVABILITY: Sub-stream sizes don't sum to 100% (total: {total_size:.1f})")
        
        return issues
    
    def _check_step_budget(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Verify solution is achievable within 40 step limit.
        """
        issues = []
        
        max_steps = self.config.get('termination', {}).get('max_steps', 40)
        target_knowledge = 200
        
        sub_streams = level_state.get('sub_streams', [])
        
        # Estimate minimum steps needed for knowledge production
        total_production_potential = sum(
            stream['size'] * (stream['coherence'] + stream['novelty']) / 200 
            for stream in sub_streams
        )
        
        if total_production_potential > 0:
            steps_needed_for_knowledge = target_knowledge / total_production_potential
            
            if steps_needed_for_knowledge > max_steps * 0.8:  # Leave buffer for balance maintenance
                issues.append(f"SOLVABILITY: Estimated {steps_needed_for_knowledge:.1f} steps needed for knowledge target, but only {max_steps} available")
        
        return issues
    
    def _validate_reward_structure(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Critical check for incentive alignment - ensure rewards prioritize problem-solving.
        """
        issues = []
        
        # Check if continuous rewards are balanced
        reward_issues = self._check_reward_balance()
        issues.extend(reward_issues)
        
        # Check for potential reward exploitation
        exploitation_issues = self._check_reward_exploitation(level_state)
        issues.extend(exploitation_issues)
        
        return issues
    
    def _check_reward_balance(self) -> List[str]:
        """
        Verify reward structure prioritizes goal achievement over action usage.
        """
        issues = []
        
        reward_config = self.config.get('reward', {})
        
        # Check continuous reward factors
        unity_factor = reward_config.get('continuous', {}).get('unity_factor', 0)
        diversity_factor = reward_config.get('continuous', {}).get('diversity_factor', 0)
        knowledge_factor = reward_config.get('continuous', {}).get('knowledge_production_factor', 0)
        
        # Goal achievement rewards should be higher than maintenance
        milestone_reward = reward_config.get('milestone_rewards', {}).get('knowledge_50', 0)
        completion_bonus = reward_config.get('completion_value', {}).get('target_reached', 0)
        
        if milestone_reward < 5:
            issues.append("REWARD: Milestone rewards too low - may not incentivize goal achievement")
        
        if completion_bonus < 30:
            issues.append("REWARD: Completion bonus too low - agents may not prioritize target achievement")
        
        # Check that knowledge production is rewarded more than just maintaining balance
        max_maintenance_reward = max(unity_factor * 100, diversity_factor * 100)  # Max per step
        if knowledge_factor * 10 < max_maintenance_reward:  # 10 = reasonable knowledge gain per step
            issues.append("REWARD: Knowledge production undervalued compared to balance maintenance")
        
        return issues
    
    def _check_reward_exploitation(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Check for potential reward loops or exploitation strategies.
        """
        issues = []
        
        # In this environment, main exploitation risks are:
        # 1. Grinding unity/diversity without progress toward knowledge target
        # 2. Focusing only on continuous rewards while ignoring milestones
        
        # The current reward structure is well-designed:
        # - Continuous rewards are small (0.05 per %)
        # - Milestone rewards are significant (10 points)
        # - Completion bonus is large (50 points)
        # - No negative penalties prevent exploration
        
        # Check if actions could be used for "reward farming"
        sub_streams = level_state.get('sub_streams', [])
        if len(sub_streams) > 6:
            issues.append("REWARD: Too many sub-streams may enable merge/split farming for diversity rewards")
        
        return issues
    
    def _validate_structure(self, level_state: Dict[str, Any]) -> List[str]:
        """
        Basic structural validation of the level format.
        """
        issues = []
        
        required_global_keys = ['unity', 'diversity', 'knowledge_score', 'cognitive_energy', 'step_count']
        globals_state = level_state.get('globals', {})
        
        for key in required_global_keys:
            if key not in globals_state:
                issues.append(f"STRUCTURE: Missing required global key: {key}")
        
        sub_streams = level_state.get('sub_streams', [])
        if not sub_streams:
            issues.append("STRUCTURE: No sub-streams defined")
        
        required_stream_keys = ['id', 'size', 'coherence', 'novelty', 'knowledge_per_turn']
        for i, stream in enumerate(sub_streams):
            for key in required_stream_keys:
                if key not in stream:
                    issues.append(f"STRUCTURE: Sub-stream {i} missing required key: {key}")
        
        # Check for duplicate stream IDs
        stream_ids = [stream.get('id') for stream in sub_streams if 'id' in stream]
        if len(stream_ids) != len(set(stream_ids)):
            issues.append("STRUCTURE: Duplicate sub-stream IDs found")
        
        return issues

def validate_level_file(level_path: str) -> Tuple[bool, List[str]]:
    """
    Standalone function to validate a single level file.
    """
    validator = HiveMindLevelValidator()
    return validator.validate_level(level_path)

def validate_all_levels(levels_dir: str = "./levels/") -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate all level files in the specified directory.
    """
    validator = HiveMindLevelValidator()
    results = {}
    
    if not os.path.exists(levels_dir):
        return {"ERROR": (False, [f"Levels directory {levels_dir} does not exist"])}
    
    for filename in os.listdir(levels_dir):
        if filename.endswith('.yaml'):
            level_path = os.path.join(levels_dir, filename)
            results[filename] = validator.validate_level(level_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    results = validate_all_levels()
    for level_name, (is_valid, issues) in results.items():
        print(f"\n{level_name}: {'✅ VALID' if is_valid else '❌ INVALID'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")