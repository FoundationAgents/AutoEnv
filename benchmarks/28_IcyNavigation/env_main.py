from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import LocalWindowObservation
from env_generate import ReverseLakeGenerator
from typing import Dict, Any, Optional, Tuple, List
import yaml
import os

class ReverseLakeNavEnv(SkinEnv):
    def __init__(self, env_id: str):
        # Initialize observation policy
        obs_policy = LocalWindowObservation(window_size=3)
        super().__init__(env_id, obs_policy)
        self.terminated = False
        self.termination_reason = None
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        self.terminated = False
        self.termination_reason = None
        
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        else:
            raise ValueError("mode must be either 'load' or 'generate'")
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World file not found: {world_path}")
        
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = ReverseLakeGenerator(self.env_id, self.configs)
        world_id = generator.generate(seed=seed)
        return world_id
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.terminated:
            return self._state
        
        action_name = action.get('action')
        params = action.get('params', {})
        
        # Store previous state in history
        self._history.append(self._state.copy())
        
        current_pos = self._state['agent']['pos']
        new_pos = current_pos.copy()
        
        # Handle movement actions
        if action_name == "MoveNorth":
            new_pos[1] -= 1
        elif action_name == "MoveSouth":
            new_pos[1] += 1
        elif action_name == "MoveEast":
            new_pos[0] += 1
        elif action_name == "MoveWest":
            new_pos[0] -= 1
        elif action_name == "Wait":
            # Stay in same position
            pass
        else:
            # Invalid action, treat as wait
            pass
        
        # Check boundaries
        grid_size = self._state['globals']['grid_size']
        if (new_pos[0] >= 0 and new_pos[0] < grid_size[0] and 
            new_pos[1] >= 0 and new_pos[1] < grid_size[1]):
            
            # Check if stepping on ice
            for ice_tile in self._state['objects']['ice_tiles']:
                if ice_tile['pos'] == new_pos:
                    self._state['agent']['pos'] = new_pos
                    self.terminated = True
                    self.termination_reason = "stepped_on_ice"
                    return self._state
            
            # Check if reaching goal
            if self._state['objects']['goal_flag']['pos'] == new_pos:
                self._state['agent']['pos'] = new_pos
                self._state['objects']['goal_flag']['collected'] = True
                self.terminated = True
                self.termination_reason = "goal_reached"
                return self._state
            
            # Safe move
            self._state['agent']['pos'] = new_pos
        
        # If we reach max steps, terminate
        if self._t + 1 >= self._state['globals']['max_steps']:
            self.terminated = True
            self.termination_reason = "timeout"
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_value = 0.0
        
        if self.terminated and self.termination_reason == "goal_reached":
            reward_value = self.configs['reward']['goal_values']['success']
            events.append("goal_reached")
        elif self.terminated and self.termination_reason == "stepped_on_ice":
            reward_value = self.configs['reward']['goal_values']['failure']
            events.append("stepped_on_ice")
        elif self.terminated and self.termination_reason == "timeout":
            reward_value = self.configs['reward']['goal_values']['timeout']
            events.append("timeout")
        
        reward_info = {
            "termination_reason": self.termination_reason,
            "terminated": self.terminated
        }
        
        return reward_value, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> Any:
        # Format local grid for display
        local_grid_display = ""
        for row in omega['local_grid']:
            local_grid_display += " ".join(row) + "\n"
        local_grid_display = local_grid_display.strip()
        
        max_steps = self._state['globals']['max_steps']
        
        rendered = f"""Step {omega['t']}/{max_steps} | Remaining: {omega['remaining_steps']}
Local View (3x3 centered on agent):
{local_grid_display}

Legend: A=Agent, H=Hole(safe), I=Ice(danger), G=Goal, #=Boundary
Position: {omega['agent_pos']}
Actions: MoveNorth, MoveSouth, MoveEast, MoveWest, Wait"""
        
        return rendered
    
    def done(self, state=None) -> bool:
        return self.terminated or self._t >= self._state['globals']['max_steps']