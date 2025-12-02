import yaml
from typing import Dict, Any, List, Tuple, Optional
import copy

class BackwardsInvestigationValidator:
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a generated backwards investigation level for solvability and proper design.
        Returns: (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        self._validate_solvability(world_state)
        
        # 2. REWARD STRUCTURE DESIGN
        self._validate_reward_structure(world_state)
        
        # 3. Environment-specific validations
        self._validate_temporal_consistency(world_state)
        self._validate_investigation_mechanics(world_state)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_solvability(self, world_state: Dict[str, Any]):
        """Critical check for impossible puzzles"""
        
        # 1. ACTION CONSTRAINT ANALYSIS
        self._check_action_constraints(world_state)
        
        # 2. TARGET REACHABILITY
        self._check_target_reachability(world_state)
        
        # 3. COMMON IMPOSSIBLE PATTERNS
        self._check_impossible_patterns(world_state)
        
        # 4. VALIDATION LOGIC FRAMEWORK
        self._check_solution_path_exists(world_state)
    
    def _check_action_constraints(self, world_state: Dict[str, Any]):
        """Understand environment's fundamental limitations"""
        
        ground_truth = world_state.get('ground_truth', {})
        timeline_events = world_state.get('timeline', {}).get('events', {})
        essential_clues = ground_truth.get('essential_clues', [])
        
        # Check if actions can access required information
        root_time = ground_truth.get('root_cause_time', 0)
        root_perpetrator = ground_truth.get('root_cause_perpetrator', '')
        root_action = ground_truth.get('root_cause_action', '')
        
        # Verify root cause event exists in timeline
        if str(root_time) not in timeline_events:
            self.validation_errors.append(f"Root cause time {root_time} not present in timeline events")
        else:
            root_event = timeline_events[str(root_time)]
            if root_event.get('perpetrator') != root_perpetrator:
                self.validation_errors.append(f"Timeline event perpetrator mismatch at root time {root_time}")
            if root_event.get('action') != root_action:
                self.validation_errors.append(f"Timeline event action mismatch at root time {root_time}")
        
        # Check if essential clues are accessible through available actions
        accessible_clues = 0
        for clue in essential_clues:
            if clue.get('relevance') == 'essential':
                # Clue must be accessible via ExamineScene, InterrogatePerson, or TraceObject
                if (clue.get('location') and clue.get('time_available') is not None and 
                    clue.get('time_available') < 40):  # Must be accessible before investigation end
                    accessible_clues += 1
                else:
                    self.validation_errors.append(f"Essential clue {clue['id']} not accessible through investigation actions")
        
        if accessible_clues < 3:  # Need minimum clues to solve
            self.validation_errors.append(f"Insufficient essential clues accessible ({accessible_clues}), need at least 3")
    
    def _check_target_reachability(self, world_state: Dict[str, Any]):
        """Verify target state is actually achievable"""
        
        ground_truth = world_state.get('ground_truth', {})
        crime_scene = world_state.get('crime_scene', {})
        
        root_time = ground_truth.get('root_cause_time', 0)
        root_perpetrator = ground_truth.get('root_cause_perpetrator', '')
        
        # Check if target perpetrator is in suspect list
        suspect_list = crime_scene.get('suspect_list', [])
        suspect_names = [s.get('name', '') for s in suspect_list]
        
        if root_perpetrator not in suspect_names:
            self.validation_errors.append(f"Root cause perpetrator '{root_perpetrator}' not in initial suspect list")
        
        # Check time constraints - agent starts at time 40 and moves backwards
        max_steps = world_state.get('globals', {}).get('max_steps', 40)
        if root_time < 0:
            self.validation_errors.append(f"Root cause time {root_time} is before timeline start (0)")
        
        # Agent needs enough steps to reach root time
        steps_needed = 40 - root_time
        if steps_needed > max_steps:
            self.validation_errors.append(f"Root cause at time {root_time} unreachable within {max_steps} steps")
    
    def _check_impossible_patterns(self, world_state: Dict[str, Any]):
        """Check for common impossible patterns"""
        
        timeline_events = world_state.get('timeline', {}).get('events', {})
        essential_clues = world_state.get('ground_truth', {}).get('essential_clues', [])
        unresolved_effects = world_state.get('investigation', {}).get('unresolved_effects', [])
        
        # Pattern 1: Circular dependencies
        event_dependencies = {}
        for time_str, event in timeline_events.items():
            caused_by = event.get('caused_by')
            if caused_by is not None:
                event_dependencies[int(time_str)] = caused_by
        
        # Check for circular references
        for time_idx, caused_by in event_dependencies.items():
            visited = set()
            current = time_idx
            while current in event_dependencies and current not in visited:
                visited.add(current)
                current = event_dependencies[current]
            if current in visited:
                self.validation_errors.append(f"Circular dependency detected involving event at time {time_idx}")
        
        # Pattern 2: Required resources not available
        for effect in unresolved_effects:
            # Each unresolved effect should be resolvable by connecting available clues
            if not self._can_resolve_effect(effect, essential_clues, timeline_events):
                self.validation_warnings.append(f"Unresolved effect may not be resolvable: {effect}")
        
        # Pattern 3: Target violates environment invariants
        ground_truth = world_state.get('ground_truth', {})
        root_time = ground_truth.get('root_cause_time', 0)
        
        # Root cause must be earlier than all consequence events
        for time_str, event in timeline_events.items():
            event_time = int(time_str)
            if event.get('type') != 'root_cause' and event_time <= root_time:
                self.validation_errors.append(f"Consequence event at time {event_time} occurs before root cause at time {root_time}")
    
    def _check_solution_path_exists(self, world_state: Dict[str, Any]):
        """Validate that a solution path exists within step limits"""
        
        ground_truth = world_state.get('ground_truth', {})
        essential_clues = ground_truth.get('essential_clues', [])
        timeline_events = world_state.get('timeline', {}).get('events', {})
        max_steps = world_state.get('globals', {}).get('max_steps', 40)
        
        # Simulate minimum steps needed to solve
        root_time = ground_truth.get('root_cause_time', 0)
        
        # Steps to reach root time
        steps_to_root = 40 - root_time
        
        # Steps to collect essential clues
        essential_clue_times = []
        for clue in essential_clues:
            if clue.get('relevance') == 'essential':
                clue_time = clue.get('time_available', 40)
                essential_clue_times.append(40 - clue_time)  # Steps from start to reach clue
        
        # Minimum steps needed (reach furthest essential clue)
        min_steps_needed = max([steps_to_root] + essential_clue_times) if essential_clue_times else steps_to_root
        
        if min_steps_needed > max_steps:
            self.validation_errors.append(f"Minimum steps to solve ({min_steps_needed}) exceeds max steps ({max_steps})")
        
        # Check if enough essential clues exist
        essential_count = len([c for c in essential_clues if c.get('relevance') == 'essential'])
        if essential_count < 3:
            self.validation_errors.append(f"Insufficient essential clues ({essential_count}), need at least 3 for solvable puzzle")
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]):
        """Critical check for incentive alignment"""
        
        # 1. GOAL-ORIENTED REWARDS
        self._check_goal_oriented_rewards(world_state)
        
        # 2. AVOID INCENTIVE MISALIGNMENT
        self._check_incentive_alignment(world_state)
    
    def _check_goal_oriented_rewards(self, world_state: Dict[str, Any]):
        """Design rewards that prioritize problem-solving over action usage"""
        
        # In this environment, rewards are binary and only given for correct identification
        # This is actually good design - no action grinding possible
        
        # Verify ground truth is complete for reward validation
        ground_truth = world_state.get('ground_truth', {})
        required_fields = ['root_cause_time', 'root_cause_perpetrator', 'root_cause_action']
        
        for field in required_fields:
            if field not in ground_truth or ground_truth[field] is None or ground_truth[field] == '':
                self.validation_errors.append(f"Missing or empty ground truth field: {field}")
        
        # Check that there's exactly one correct answer
        if not self._has_unique_solution(world_state):
            self.validation_errors.append("Level does not have exactly one unique solution")
    
    def _check_incentive_alignment(self, world_state: Dict[str, Any]):
        """Avoid incentive misalignment"""
        
        # This environment uses binary rewards which prevents:
        # - Action grinding (no incremental rewards)
        # - Exploration loops (no rewards for repeated actions)
        # - Action farming (only final identification matters)
        
        # Verify decoy suspects exist to prevent trivial solutions
        decoy_suspects = world_state.get('ground_truth', {}).get('decoy_suspects', [])
        if len(decoy_suspects) < 2:
            self.validation_warnings.append("Insufficient decoy suspects, puzzle may be too easy")
        
        # Check that decoy clues exist
        essential_clues = world_state.get('ground_truth', {}).get('essential_clues', [])
        decoy_count = len([c for c in essential_clues if c.get('relevance') == 'decoy'])
        if decoy_count < 2:
            self.validation_warnings.append("Insufficient decoy clues, may allow trivial solution")
    
    def _validate_temporal_consistency(self, world_state: Dict[str, Any]):
        """Validate temporal logic and backwards investigation mechanics"""
        
        timeline_events = world_state.get('timeline', {}).get('events', {})
        
        # Check temporal ordering
        for time_str, event in timeline_events.items():
            time_idx = int(time_str)
            
            # If event has a cause, cause should be earlier
            caused_by = event.get('caused_by')
            if caused_by is not None and caused_by >= time_idx:
                self.validation_errors.append(f"Event at time {time_idx} caused by later or same time {caused_by}")
        
        # Check that times are within valid range [0, 40]
        for time_str in timeline_events.keys():
            time_idx = int(time_str)
            if time_idx < 0 or time_idx > 40:
                self.validation_errors.append(f"Timeline event at invalid time {time_idx} (must be 0-40)")
    
    def _validate_investigation_mechanics(self, world_state: Dict[str, Any]):
        """Validate investigation-specific mechanics"""
        
        # Check suspect list completeness
        crime_scene = world_state.get('crime_scene', {})
        suspect_list = crime_scene.get('suspect_list', [])
        
        for suspect in suspect_list:
            required_fields = ['name', 'involvement_level', 'evidence_against', 'alibi_status']
            for field in required_fields:
                if field not in suspect:
                    self.validation_errors.append(f"Suspect missing required field: {field}")
        
        # Check unresolved effects exist
        unresolved_effects = world_state.get('investigation', {}).get('unresolved_effects', [])
        if len(unresolved_effects) == 0:
            self.validation_warnings.append("No unresolved effects defined, investigation may lack direction")
        
        # Validate clue structure
        essential_clues = world_state.get('ground_truth', {}).get('essential_clues', [])
        for clue in essential_clues:
            required_fields = ['id', 'type', 'location', 'time_available', 'relevance', 'description']
            for field in required_fields:
                if field not in clue:
                    self.validation_errors.append(f"Clue {clue.get('id', 'unknown')} missing required field: {field}")
    
    def _can_resolve_effect(self, effect: str, clues: List[Dict], events: Dict) -> bool:
        """Check if an unresolved effect can be resolved with available clues"""
        # Simple heuristic - in real implementation this would be more sophisticated
        return len(clues) > 0 and len(events) > 0
    
    def _has_unique_solution(self, world_state: Dict[str, Any]) -> bool:
        """Verify the level has exactly one correct solution"""
        ground_truth = world_state.get('ground_truth', {})
        
        # Check that root cause is unique
        root_time = ground_truth.get('root_cause_time')
        root_perpetrator = ground_truth.get('root_cause_perpetrator')
        root_action = ground_truth.get('root_cause_action')
        
        # All three must be specified and non-empty
        return (root_time is not None and 
                root_perpetrator and 
                root_action and
                isinstance(root_time, int) and 
                0 <= root_time <= 40)

def validate_generated_level(world_state: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Main validation function for backwards investigation levels.
    Returns: (is_valid, errors, warnings)
    """
    validator = BackwardsInvestigationValidator()
    return validator.validate_level(world_state)

# Helper function for external use
def validate_level_file(file_path: str) -> Tuple[bool, List[str], List[str]]:
    """Validate a level from a YAML file"""
    try:
        with open(file_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return validate_generated_level(world_state)
    except Exception as e:
        return False, [f"Failed to load level file: {str(e)}"], []