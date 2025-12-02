from typing import Dict, Any, List, Tuple, Optional
import copy
import yaml

class BizarroLabValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inverted_reaction_table = {
            ("Acid", "Base"): {"product": "Xylene", "pH_change": 2.0, "temp_change": -10},
            ("Solvent", "Acid"): {"product": "Bizarrolene", "pH_change": -1.5, "temp_change": 8},
            ("Base", "Solvent"): {"product": "InvertedAcetate", "pH_change": 1.0, "temp_change": -5}
        }
        self.available_reagents = ["Acid", "Base", "Solvent"]
        self.target_compounds = ["Xylene", "Bizarrolene", "InvertedAcetate"]
        
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """
        Validates a generated level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        try:
            with open(level_path, 'r') as f:
                level_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load level: {str(e)}"]
        
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(level_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(level_state)
        issues.extend(reward_issues)
        
        # 3. BASIC CONFIGURATION VALIDATION
        config_issues = self._validate_basic_config(level_state)
        issues.extend(config_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, level_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        target_compound = level_state.get("globals", {}).get("target_compound")
        if not target_compound:
            issues.append("No target compound specified")
            return issues
        
        if target_compound not in self.target_compounds:
            issues.append(f"Invalid target compound: {target_compound}. Must be one of {self.target_compounds}")
            return issues
        
        # TARGET REACHABILITY - Check if target compound can be synthesized
        synthesis_path = self._find_synthesis_path(target_compound)
        if not synthesis_path:
            issues.append(f"Target compound {target_compound} cannot be synthesized with available reagents and reactions")
        else:
            # Check if required reagents are available
            required_reagents = synthesis_path["reagents"]
            available_reagents = [r["name"] for r in level_state.get("reagents_catalog", [])]
            missing_reagents = [r for r in required_reagents if r not in available_reagents]
            if missing_reagents:
                issues.append(f"Missing required reagents for {target_compound}: {missing_reagents}")
        
        # STEP COUNTING - Estimate minimum steps needed
        if synthesis_path:
            min_steps = self._estimate_minimum_steps(synthesis_path, level_state)
            max_steps = level_state.get("globals", {}).get("step_remaining", 40)
            if min_steps > max_steps:
                issues.append(f"Insufficient steps: need at least {min_steps}, but only {max_steps} available")
        
        # RESOURCE AVAILABILITY - Check beaker capacity and volumes
        capacity_issues = self._check_capacity_constraints(level_state, synthesis_path)
        issues.extend(capacity_issues)
        
        return issues
    
    def _find_synthesis_path(self, target_compound: str) -> Optional[Dict[str, Any]]:
        """Find the synthesis path for the target compound"""
        synthesis_paths = {
            "Xylene": {
                "reagents": ["Acid", "Base"],
                "reaction_key": ("Acid", "Base"),
                "min_volume_each": 50  # ml needed of each reagent
            },
            "Bizarrolene": {
                "reagents": ["Solvent", "Acid"], 
                "reaction_key": ("Solvent", "Acid"),
                "min_volume_each": 50
            },
            "InvertedAcetate": {
                "reagents": ["Base", "Solvent"],
                "reaction_key": ("Base", "Solvent"), 
                "min_volume_each": 50
            }
        }
        return synthesis_paths.get(target_compound)
    
    def _estimate_minimum_steps(self, synthesis_path: Dict[str, Any], level_state: Dict[str, Any]) -> int:
        """Estimate minimum steps needed to complete synthesis"""
        min_steps = 0
        
        # Steps to add reagents (2 reagents minimum)
        min_steps += 2
        
        # Steps to set up equipment (stir, temperature control)
        min_steps += 2  # Set stirrer + toggle heating/cooling
        
        # Wait time for reaction to occur (based on reaction kinetics)
        # At minimum stirrer speed (1), reaction rate is 0.2, need significant conversion
        min_steps += 10  # Conservative estimate for reaction time
        
        # Submit step
        min_steps += 1
        
        # Add buffer for potential failed attempts or optimization
        min_steps += 5
        
        return min_steps
    
    def _check_capacity_constraints(self, level_state: Dict[str, Any], synthesis_path: Optional[Dict[str, Any]]) -> List[str]:
        """Check if beaker capacities allow for proper synthesis"""
        issues = []
        
        if not synthesis_path:
            return issues
        
        beakers = level_state.get("beakers", [])
        if not beakers:
            issues.append("No beakers available")
            return issues
        
        max_capacity = max(beaker.get("capacity_ml", 0) for beaker in beakers)
        min_volume_needed = synthesis_path["min_volume_each"] * 2  # Need space for both reagents
        
        if max_capacity < min_volume_needed:
            issues.append(f"Beaker capacity ({max_capacity}ml) insufficient for synthesis (need {min_volume_needed}ml)")
        
        return issues
    
    def _validate_reward_structure(self, level_state: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Check target purity requirement
        target_purity = level_state.get("globals", {}).get("target_purity", 0.95)
        if target_purity < 0.8 or target_purity > 1.0:
            issues.append(f"Invalid target purity: {target_purity}. Should be between 0.8 and 1.0")
        
        # The reward structure is defined in the environment code, but we can validate
        # that the level setup doesn't create reward exploitation opportunities
        
        # GOAL-ORIENTED REWARDS - Check that success is most rewarding
        # Based on environment code: success gives 1.0 + 0.5 * unused_fraction
        # Dense rewards are purity improvements, max possible per step is small
        max_dense_reward_per_step = 0.1  # Conservative estimate
        max_steps = level_state.get("globals", {}).get("step_remaining", 40)
        max_possible_dense_reward = max_dense_reward_per_step * max_steps
        min_success_reward = 1.0  # Base success bonus
        
        if max_possible_dense_reward > min_success_reward * 2:
            issues.append("Potential reward misalignment: dense rewards might exceed success rewards")
        
        # AVOID INCENTIVE MISALIGNMENT - Check for action grinding opportunities
        # The environment doesn't reward meaningless actions, which is good
        # But we should ensure the level doesn't create infinite loops
        
        return issues
    
    def _validate_basic_config(self, level_state: Dict[str, Any]) -> List[str]:
        """Validate basic configuration requirements"""
        issues = []
        
        # Check required globals
        globals_section = level_state.get("globals", {})
        required_globals = ["target_compound", "target_purity", "step_remaining", "ambient_temperature_c"]
        for key in required_globals:
            if key not in globals_section:
                issues.append(f"Missing required global: {key}")
        
        # Check beakers setup
        beakers = level_state.get("beakers", [])
        if len(beakers) != 5:
            issues.append(f"Expected 5 beakers, found {len(beakers)}")
        
        for i, beaker in enumerate(beakers):
            if beaker.get("id") != i:
                issues.append(f"Beaker {i} has incorrect id: {beaker.get('id')}")
            if beaker.get("capacity_ml", 0) <= 0:
                issues.append(f"Beaker {i} has invalid capacity: {beaker.get('capacity_ml')}")
            if beaker.get("volume_ml", 0) != 0:
                issues.append(f"Beaker {i} should start empty but has volume: {beaker.get('volume_ml')}")
        
        # Check equipment setup
        equipment = level_state.get("equipment", {})
        required_equipment = ["hot_plates", "cooling_coils", "stir_speeds"]
        for key in required_equipment:
            if key not in equipment:
                issues.append(f"Missing equipment section: {key}")
            elif len(equipment[key]) != 5:
                issues.append(f"Equipment {key} should have 5 entries, found {len(equipment[key])}")
        
        # Check reagents catalog
        reagents = level_state.get("reagents_catalog", [])
        expected_reagents = {"Acid", "Base", "Solvent"}
        available_reagents = {r.get("name") for r in reagents if r.get("name")}
        if not expected_reagents.issubset(available_reagents):
            missing = expected_reagents - available_reagents
            issues.append(f"Missing reagents in catalog: {missing}")
        
        return issues
    
    def simulate_optimal_solution(self, level_state: Dict[str, Any]) -> Tuple[bool, int, List[str]]:
        """
        Simulate an optimal solution attempt to verify solvability.
        Returns (success, steps_used, action_log)
        """
        target_compound = level_state.get("globals", {}).get("target_compound")
        if not target_compound:
            return False, 0, ["No target compound specified"]
        
        synthesis_path = self._find_synthesis_path(target_compound)
        if not synthesis_path:
            return False, 0, ["No synthesis path found"]
        
        action_log = []
        steps_used = 0
        
        # Simulate optimal action sequence
        # 1. Add first reagent
        reagents = synthesis_path["reagents"]
        action_log.append(f"AddReagent(beaker_id=0, reagent_name='{reagents[0]}', volume_ml=100)")
        steps_used += 1
        
        # 2. Add second reagent  
        action_log.append(f"AddReagent(beaker_id=0, reagent_name='{reagents[1]}', volume_ml=100)")
        steps_used += 1
        
        # 3. Set stirrer
        action_log.append("SetStirSpeed(beaker_id=0, speed=3)")
        steps_used += 1
        
        # 4. Wait for reaction (conservative estimate)
        wait_steps = 15
        for i in range(wait_steps):
            action_log.append("Wait()")
            steps_used += 1
        
        # 5. Submit for analysis
        action_log.append("SubmitForAnalysis(beaker_id=0)")
        steps_used += 1
        
        max_steps = level_state.get("globals", {}).get("step_remaining", 40)
        success = steps_used <= max_steps
        
        return success, steps_used, action_log

def validate_bizarro_lab_level(level_path: str, config_path: str = "./config.yaml") -> Tuple[bool, List[str]]:
    """
    Main validation function for Bizarro Lab levels.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load config: {str(e)}"]
    
    validator = BizarroLabValidator(config)
    return validator.validate_level(level_path)