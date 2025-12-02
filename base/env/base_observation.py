# ============================================================
# OBSERVATION POLICY BASE CLASS
# Purpose: Abstract base class for implementing observation policies
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, Any

class ObservationPolicy(ABC):
    """Semantic observation extractor."""
    @abstractmethod
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        You can implement custom observe methods here. 
        """
        pass

