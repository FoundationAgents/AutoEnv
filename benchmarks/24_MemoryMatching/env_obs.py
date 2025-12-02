from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
import numpy as np

class ChaosSlideObservation(ObservationPolicy):
    """Observation policy for Chaos Slide Puzzle environment."""
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Extract observation from environment state.
        
        Args:
            env_state: Complete environment state
            t: Current timestep
            
        Returns:
            Observation dictionary for agent
        """
        observation = {
            'board': np.array(env_state['board']['grid'], dtype=int),
            'steps_remaining': env_state['agent']['steps_remaining'],
            'chaos_pattern': np.array(env_state['targets']['chaos_pattern'], dtype=int),
            't': t + 1
        }
        
        return observation