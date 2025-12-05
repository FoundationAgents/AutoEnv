# ============================================================
# BASE ENVIRONMENT CLASSES
# Purpose: Abstract base classes for implementing agentic environments
# ============================================================

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from base.env.base_observation import ObservationPolicy

class BaseEnv(ABC):
    """Defines the true state, transition, and reward."""
    def __init__(self, env_id: int):
        self.env_id = env_id # env_id means the id of this class env. 
        self._t = 0
        self._history: List = [] # past state 
        self._state = None # current state
        self.configs = None
        # Optional: store latest action side-effect/result for UI/agent feedback
        self._last_action_result: Any = None

    @abstractmethod
    def _dsl_config(self): 
        """
        Load DSL configuration from YAML file.
        Expected path: worlds/{env_id}/config.yaml
        """
        pass

    @abstractmethod
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        """
        Reset environment by either loading an existing world or generating a new one.

        Args:
            mode: "load" to load from file, "generate" to generate a new world
            world_id: Used only in "load" mode. Load the world with this id.
            seed: Used only in "generate" mode. Generate a new world with this seed.

        Behavior:
            - If mode == "load": Load world state from file using world_id.
            - If mode == "generate": Generate new world using seed, then load it.
        """
        pass

    @abstractmethod
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """
        Load world state from file.
        
        Args:
            world_id: Identifier of the world file to load
            
        Returns:
            Complete world state dictionary
        """
        pass
        
    @abstractmethod  
    def _generate_world(self, seed: Optional[int] = None) -> str:
        """
        Generate complete world using generator pipeline and save to file.
        
        Args:
            seed: Random seed for reproducible generation
            
        Returns:
            world_id: Identifier of the generated world file
        """
        pass

    @abstractmethod
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        State transition function.
        Input an action dict with two key:
        - action: str, the name of action
        - params: dict, the parameters of action
        And then apply the transition to self.state
        """
        pass

    @abstractmethod
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        """
        Reward Function.
        It define agent how to get a reward.
        The state can be obtained from self.state, and past state can be gained from self.history. 
        """
        pass


class ObsEnv(BaseEnv):
    """Adds observation interface: output semantic observation from true state."""

    def __init__(self, env_id, obs_policy: ObservationPolicy):
        super().__init__(env_id)
        self.obs_policy = obs_policy

    @abstractmethod
    def observe_semantic(self):
        """
        Semantic-level observation.
        The observation policy refer to the observation state, such as full, partial, radius. 
        And this function is used to transfer state to semantic obs.
        """
        pass


class SkinEnv(ObsEnv):
    """Adds rendering interface: semantic observation -> final input (X)."""

    @abstractmethod
    def render_skin(self, omega) -> Any:
        """Render the final input from semantic observation."""
        pass

    def done(self) -> bool:
        # Default: only step count; override/add conditions if needed
        return self._t >= self.configs["termination"]["max_steps"]

    def step(self, action: Dict[str, Any]):
        """
        Basic step logic for an environment
        You can modify it in anywhere you want.
        """
        # Reset last action result; transition can set it
        self._last_action_result = None
        s_next = self.transition(action)
        reward, events, rinfo = self.reward(action)
        self._t += 1
        raw_obs = self.observe_semantic()
        agent_obs = self.render_skin(raw_obs)
        if_done = self.done()
        info = {
            "raw_obs": raw_obs,
            "skinned": agent_obs,
            "events": events,
            "reward_info": rinfo,
            "last_action_result": self._last_action_result,
        }
        return s_next, reward, if_done, info

