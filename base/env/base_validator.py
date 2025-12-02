# ============================================================
# Base class for environment level validators.
# Used to verify whether a generated level instance is reasonable by design, ensuring a reachable reward path exists.
# ============================================================


from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Reward type enumeration"""
    BINARY = "binary"        # Binary: success/failure
    CUMULATIVE = "cumulative"  # Cumulative: reward accumulation


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    reward_type: RewardType
    issues: List[str]
    suggestions: List[str]
    theoretical_max_reward: Optional[float] = None
    success_probability: Optional[float] = None
    
    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = f"{status} ({self.reward_type.value})\n"
        
        if self.theoretical_max_reward is not None:
            result += f"Max Possible Reward: {self.theoretical_max_reward}\n"
        if self.success_probability is not None:
            result += f"Success Probability: {self.success_probability:.2%}\n"
            
        if self.issues:
            result += "Issues:\n" + "\n".join(f"  - {issue}" for issue in self.issues)
        if self.suggestions:
            result += "Suggestions:\n" + "\n".join(f"  - {suggestion}" for suggestion in self.suggestions)
            
        return result


class BaseValidator(ABC):
    """Base class for environment validators"""
    
    def __init__(self):
        self.reward_type = self.get_reward_type()
    
    @abstractmethod
    def get_reward_type(self) -> RewardType:
        """Return the reward type supported by this validator"""
        pass
    
    @abstractmethod
    def validate(self, level_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate the rationality of a level instance's design
        
        Args:
            level_data: Complete level data (dictionary parsed from YAML)
        
        Returns:
            ValidationResult: Validation result
        """
        pass
    
    def quick_check(self, level_data: Dict[str, Any]) -> bool:
        """Quickly check basic validity of the level"""
        result = self.validate(level_data)
        return result.is_valid


class BinaryRewardValidator(BaseValidator):
    """Base validator for binary (0/1 class) rewards"""
    
    def get_reward_type(self) -> RewardType:
        return RewardType.BINARY
    
    def validate(self, level_data: Dict[str, Any]) -> ValidationResult:
        """Check if there exists a success path"""
        issues = []
        suggestions = []
        
        # Basic data integrity check
        basic_issues = self._check_basic_integrity(level_data)
        issues.extend(basic_issues)
        
        # Check for existence of success path
        success_possible, path_issues = self._check_success_path_exists(level_data)
        if not success_possible:
            issues.extend(path_issues)
            suggestions.append("Review termination conditions and ensure success is reachable")
        
        # Check for reasonable termination conditions
        termination_issues = self._check_termination_conditions(level_data)
        issues.extend(termination_issues)
        
        is_valid = len(issues) == 0 and success_possible
        
        return ValidationResult(
            is_valid=is_valid,
            reward_type=self.reward_type,
            issues=issues,
            suggestions=suggestions,
            success_probability=1.0 if success_possible else 0.0
        )
    
    @abstractmethod
    def _check_success_path_exists(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if there is at least one success path
        
        Returns:
            (success_possible, issues)
        """
        pass
    
    def _check_basic_integrity(self, level_data: Dict[str, Any]) -> List[str]:
        """Check basic data integrity"""
        issues = []
        
        # Check required fields
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in level_data:
                issues.append(f"Missing required field: {field}")
        
        return issues
    
    def _check_termination_conditions(self, level_data: Dict[str, Any]) -> List[str]:
        """Check termination conditions"""
        issues = []
        
        # Check if explicit termination conditions exist
        termination = level_data.get("termination", {})
        if not termination:
            issues.append("No termination conditions defined")
        
        return issues
    
    def get_required_fields(self) -> List[str]:
        """Return list of required fields (subclasses may override)"""
        return []


class CumulativeRewardValidator(BaseValidator):
    """Base validator for cumulative reward classes"""
    
    def get_reward_type(self) -> RewardType:
        return RewardType.CUMULATIVE
    
    def validate(self, level_data: Dict[str, Any]) -> ValidationResult:
        """Check if a positive outcome strategy exists"""
        issues = []
        suggestions = []
        
        # Basic data integrity check
        basic_issues = self._check_basic_integrity(level_data)
        issues.extend(basic_issues)
        
        # Check for positive reward possibility
        max_reward, reward_issues = self._calculate_theoretical_max_reward(level_data)
        if max_reward <= 0:
            issues.append("No positive reward strategy possible")
            suggestions.append("Review reward structure and ensure positive outcomes are achievable")
        
        issues.extend(reward_issues)
        
        # Check reward structure validity
        structure_issues = self._check_reward_structure(level_data)
        issues.extend(structure_issues)
        
        is_valid = len(issues) == 0 and max_reward > 0
        
        return ValidationResult(
            is_valid=is_valid,
            reward_type=self.reward_type,
            issues=issues,
            suggestions=suggestions,
            theoretical_max_reward=max_reward
        )
    
    @abstractmethod
    def _calculate_theoretical_max_reward(self, level_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate theoretical maximum reward
        
        Returns:
            (max_possible_reward, issues)
        """
        pass
    
    def _check_basic_integrity(self, level_data: Dict[str, Any]) -> List[str]:
        """Check basic data integrity"""
        issues = []
        
        # Check reward config
        reward_config = level_data.get("reward", {})
        if not reward_config:
            issues.append("No reward configuration found")
        
        return issues
    
    def _check_reward_structure(self, level_data: Dict[str, Any]) -> List[str]:
        """Check reward structure validity"""
        issues = []
        
        # Check if positive rewards exist
        reward_config = level_data.get("reward", {})
        if reward_config:
            positive_rewards = self._extract_positive_rewards(reward_config)
            if not positive_rewards:
                issues.append("No positive rewards defined in reward structure")
        
        return issues
    
    def _extract_positive_rewards(self, reward_config: Dict[str, Any]) -> List[float]:
        """Extract all positive reward values in the reward configuration"""
        positive_rewards = []
        
        def extract_rewards(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_rewards(value)
            elif isinstance(obj, (int, float)) and obj > 0:
                positive_rewards.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    extract_rewards(item)
        
        extract_rewards(reward_config)
        return positive_rewards


# Factory function
def create_validator(reward_type: str, env_specific_class=None) -> BaseValidator:
    """
    Create a validator instance
    
    Args:
        reward_type: 'binary' or 'cumulative'
        env_specific_class: environment-specific validator class (optional)
    
    Returns:
        Validator instance
    """
    if env_specific_class:
        return env_specific_class()
    
    reward_type_enum = RewardType(reward_type)
    
    if reward_type_enum == RewardType.BINARY:
        # Return default binary validator (requires subclass implementation for concrete logic)
        class DefaultBinaryValidator(BinaryRewardValidator):
            def _check_success_path_exists(self, level_data):
                return True, []  # Default implementation
        
        return DefaultBinaryValidator()
    
    elif reward_type_enum == RewardType.CUMULATIVE:
        # Return default cumulative validator (requires subclass implementation for concrete logic)
        class DefaultCumulativeValidator(CumulativeRewardValidator):
            def _calculate_theoretical_max_reward(self, level_data):
                return 1.0, []  # Default implementation
        
        return DefaultCumulativeValidator()
    
    else:
        raise ValueError(f"Unsupported reward type: {reward_type}")
