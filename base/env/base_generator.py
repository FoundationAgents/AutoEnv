# ============================================================
# WORLD GENERATOR BASE CLASS
# Purpose: Abstract base class for implementing world generators
# ============================================================


from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class WorldGenerator(ABC):
    """
    World generator that creates complete world instances and saves them to files.
    """
    
    def __init__(self, env_id: str, config: Dict[str, Any]):
        """
        Initialize generator with environment configuration.
        
        Args:
            env_id: Environment identifier for file paths
            config: Generator configuration from DSL
        """
        self.env_id = env_id
        self.config = config
        
    @abstractmethod
    def generate(self, 
                 seed: Optional[int] = None,
                 save_path: Optional[str] = None) -> str:
        """
        Generate complete world instance and save to file.
        
        Args:
            seed: Random seed for reproducible generation
            save_path: Custom save path (optional, will auto-generate if None)
            
        Returns:
            world_id: Identifier of the generated and saved world
        """
        pass
    
    @abstractmethod
    def _execute_pipeline(self, 
                         base_state: Dict[str, Any], 
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the generation pipeline to create complete world state.
        
        Args:
            base_state: Initial state from state_template
            seed: Random seed for generation
            
        Returns:
            Complete generated world state
        """
        pass
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        """
        Save generated world to file.
        
        Args:
            world_state: Complete world state to save
            world_id: Identifier for the world file
            
        Returns:
            world_id: Confirmed identifier of saved world
        """
        # Implementation would use config from world_loading section
        # to determine file path, format, etc.
        pass
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        """
        Generate unique world identifier.
        
        Args:
            seed: Seed used for generation (for ID creation)
            
        Returns:
            Unique world identifier
        """
        pass