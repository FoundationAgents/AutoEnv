import yaml
import random
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy
import numpy as np

class LightSpectrumValidator:
    """Validator for Light Spectrum Analysis Environment levels"""
    
    def __init__(self):
        self.max_steps = 40
        self.num_materials = 10
        self.num_wavelengths = 5
        self.illumination_actions = ['EmitUV', 'EmitBlue', 'EmitGreen', 'EmitRed', 'EmitIR']
        self.declaration_actions = [f'Declare_{i}' for i in range(10)]
        
    def validate_level(self, level_path: str) -> Tuple[bool, List[str]]:
        """Main validation function that checks both solvability and reward structure"""
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
        
        # 3. STRUCTURAL VALIDATION
        structure_issues = self._validate_level_structure(level_data)
        issues.extend(structure_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, level_data: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(level_data)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY
        reachability_issues = self._check_target_reachability(level_data)
        issues.extend(reachability_issues)
        
        # COMMON IMPOSSIBLE PATTERNS
        pattern_issues = self._check_impossible_patterns(level_data)
        issues.extend(pattern_issues)
        
        return issues
    
    def _analyze_action_constraints(self, level_data: Dict[str, Any]) -> List[str]:
        """Understand environment's fundamental limitations"""
        issues = []
        
        # Check if reference library exists and is accessible
        if 'reference_library' not in level_data:
            issues.append("CRITICAL: No reference library found - agent cannot compare observations")
            return issues
            
        ref_lib = level_data['reference_library'].get('material_signatures', [])
        if len(ref_lib) != self.num_materials:
            issues.append(f"CRITICAL: Reference library has {len(ref_lib)} materials, expected {self.num_materials}")
        
        # Verify each material signature is valid
        for i, signature in enumerate(ref_lib):
            if len(signature) != 10:  # 5 fluorescence + 5 reflection
                issues.append(f"CRITICAL: Material {i} signature has {len(signature)} values, expected 10")
            
            for j, value in enumerate(signature):
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                    issues.append(f"CRITICAL: Material {i} signature[{j}] = {value}, must be float in [0.0, 1.0]")
        
        # Check action space completeness
        required_actions = set(self.illumination_actions + self.declaration_actions)
        # Note: We assume action space is complete based on environment description
        
        return issues
    
    def _check_target_reachability(self, level_data: Dict[str, Any]) -> List[str]:
        """Verify target state is achievable"""
        issues = []
        
        # Check if target material ID is valid
        target_material = level_data.get('sample', {}).get('true_material_id', -1)
        if not (0 <= target_material < self.num_materials):
            issues.append(f"CRITICAL: Invalid target material ID {target_material}, must be in [0, {self.num_materials-1}]")
            return issues
        
        # Verify target material has distinguishable signature
        ref_lib = level_data.get('reference_library', {}).get('material_signatures', [])
        if len(ref_lib) <= target_material:
            issues.append(f"CRITICAL: Target material {target_material} not found in reference library")
            return issues
        
        target_signature = ref_lib[target_material]
        
        # Check if target material is distinguishable from others
        distinguishability_issues = self._check_material_distinguishability(ref_lib, target_material)
        issues.extend(distinguishability_issues)
        
        # Verify step budget is sufficient
        # Minimum steps: 5 illuminations + 1 declaration = 6 steps
        min_steps_needed = 6  # Conservative estimate for systematic approach
        if self.max_steps < min_steps_needed:
            issues.append(f"CRITICAL: Step budget {self.max_steps} < minimum needed {min_steps_needed}")
        
        # Check if perfect information gathering is possible within budget
        max_info_steps = len(self.illumination_actions) + 1  # All illuminations + declaration
        if self.max_steps < max_info_steps:
            issues.append(f"WARNING: Cannot gather complete spectral data and declare within {self.max_steps} steps")
        
        return issues
    
    def _check_material_distinguishability(self, ref_lib: List[List[float]], target_material: int) -> List[str]:
        """Check if target material can be distinguished from others"""
        issues = []
        
        if len(ref_lib) <= target_material:
            return [f"CRITICAL: Target material {target_material} not in reference library"]
        
        target_sig = ref_lib[target_material]
        
        # Check for identical signatures (impossible to distinguish)
        for i, other_sig in enumerate(ref_lib):
            if i != target_material:
                if np.allclose(target_sig, other_sig, atol=1e-6):
                    issues.append(f"CRITICAL: Target material {target_material} identical to material {i} - impossible to distinguish")
        
        # Check if target can be distinguished with partial illumination
        # This is a sophisticated analysis - we check if there exists a subset of wavelengths
        # that uniquely identifies the target material
        distinguishable = False
        
        # Check all possible illumination combinations (2^5 = 32 combinations)
        for combination in range(1, 2**self.num_wavelengths):  # Skip empty set
            if self._is_distinguishable_with_combination(ref_lib, target_material, combination):
                distinguishable = True
                break
        
        if not distinguishable:
            issues.append(f"CRITICAL: Target material {target_material} cannot be distinguished from others with any illumination combination")
        
        return issues
    
    def _is_distinguishable_with_combination(self, ref_lib: List[List[float]], target_material: int, combination: int) -> bool:
        """Check if target material is distinguishable with specific illumination combination"""
        target_sig = ref_lib[target_material]
        
        # Generate observed spectrum for target material with this illumination combination
        target_observed = self._simulate_observation(target_sig, combination)
        
        # Check if any other material could produce the same observation
        for i, other_sig in enumerate(ref_lib):
            if i != target_material:
                other_observed = self._simulate_observation(other_sig, combination)
                if np.allclose(target_observed, other_observed, atol=1e-6):
                    return False  # Not distinguishable - another material produces same observation
        
        return True  # Target material produces unique observation
    
    def _simulate_observation(self, material_signature: List[float], illumination_combination: int) -> List[float]:
        """Simulate observed spectrum for given material and illumination combination"""
        fluorescence = [0.0] * self.num_wavelengths
        reflection = [0.0] * self.num_wavelengths
        
        # Apply illumination combination (bit mask)
        for i in range(self.num_wavelengths):
            if illumination_combination & (1 << i):  # If this wavelength is illuminated
                # Use element-wise maximum as per environment rules
                fluorescence[i] = max(fluorescence[i], material_signature[i])
                reflection[i] = max(reflection[i], material_signature[i + self.num_wavelengths])
        
        return fluorescence + reflection
    
    def _check_impossible_patterns(self, level_data: Dict[str, Any]) -> List[str]:
        """Check for common impossible patterns"""
        issues = []
        
        # Pattern 1: Invalid initial state
        initial_illuminated = level_data.get('sample', {}).get('illuminated_bands', [])
        if len(initial_illuminated) != self.num_wavelengths:
            issues.append(f"CRITICAL: Initial illuminated_bands has {len(initial_illuminated)} elements, expected {self.num_wavelengths}")
        
        if any(initial_illuminated):  # Should start with no illumination
            issues.append("WARNING: Level starts with pre-illuminated bands - may affect difficulty")
        
        # Pattern 2: Invalid observed spectrum initialization
        observed = level_data.get('sample', {}).get('observed_spectrum', {})
        fluor = observed.get('fluorescence', [])
        refl = observed.get('reflection', [])
        
        if len(fluor) != self.num_wavelengths or len(refl) != self.num_wavelengths:
            issues.append("CRITICAL: Initial observed spectrum has wrong dimensions")
        
        if any(fluor) or any(refl):  # Should start with zeros
            issues.append("WARNING: Level starts with non-zero observed spectrum")
        
        # Pattern 3: Agent state consistency
        agent_state = level_data.get('agent', {})
        if agent_state.get('current_step', 0) != 0:
            issues.append("CRITICAL: Level should start at step 0")
        
        if agent_state.get('has_declared', False):
            issues.append("CRITICAL: Level should start with has_declared = False")
        
        return issues
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Validate reward structure for incentive alignment"""
        issues = []
        
        # The environment uses binary rewards: +1 for correct declaration, 0 otherwise
        # This is actually good design - let's validate it's properly set up
        
        # Check that the reward structure promotes goal achievement over action usage
        # Since this environment has binary rewards, we mainly need to ensure:
        # 1. No intermediate rewards that could cause action farming
        # 2. The single positive reward is only for correct identification
        
        # In this environment, reward structure is hardcoded in the environment class,
        # not in the level data, so we validate the level supports proper reward calculation
        
        # Ensure target material is valid for reward calculation
        target_material = level_data.get('sample', {}).get('true_material_id', -1)
        if not (0 <= target_material < self.num_materials):
            issues.append("REWARD: Invalid target material prevents proper reward calculation")
        
        # Check that there are enough declaration actions
        # Agent must be able to declare any material as the answer
        # This is handled by the action space, not level data
        
        # Validate that the binary reward structure is maintained
        # (This environment's reward structure is actually well-designed)
        
        return issues
    
    def _validate_level_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Validate basic level structure and required components"""
        issues = []
        
        # Check required top-level keys
        required_keys = ['globals', 'agent', 'sample', 'reference_library']
        for key in required_keys:
            if key not in level_data:
                issues.append(f"STRUCTURE: Missing required section '{key}'")
        
        # Validate globals section
        globals_section = level_data.get('globals', {})
        if globals_section.get('max_steps') != self.max_steps:
            issues.append(f"STRUCTURE: max_steps should be {self.max_steps}")
        if globals_section.get('num_materials') != self.num_materials:
            issues.append(f"STRUCTURE: num_materials should be {self.num_materials}")
        if globals_section.get('num_wavelengths') != self.num_wavelengths:
            issues.append(f"STRUCTURE: num_wavelengths should be {self.num_wavelengths}")
        
        # Validate agent section
        agent_section = level_data.get('agent', {})
        required_agent_keys = ['current_step', 'has_declared', 'declared_material']
        for key in required_agent_keys:
            if key not in agent_section:
                issues.append(f"STRUCTURE: Missing agent.{key}")
        
        # Validate sample section
        sample_section = level_data.get('sample', {})
        required_sample_keys = ['true_material_id', 'illuminated_bands', 'observed_spectrum']
        for key in required_sample_keys:
            if key not in sample_section:
                issues.append(f"STRUCTURE: Missing sample.{key}")
        
        return issues

def validate_generated_level(level_path: str) -> Tuple[bool, List[str]]:
    """Main entry point for level validation"""
    validator = LightSpectrumValidator()
    return validator.validate_level(level_path)

# Example usage:
if __name__ == "__main__":
    # Test validation on a level file
    is_valid, issues = validate_generated_level("./levels/test_level.yaml")
    
    if is_valid:
        print("✅ Level is valid and solvable!")
    else:
        print("❌ Level validation failed:")
        for issue in issues:
            print(f"  - {issue}")