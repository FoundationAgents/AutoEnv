from base.env.base_env import SkinEnv
from env_obs import MemoryObservationPolicy
from env_generate import MemoryGenerator
import yaml
import random
from typing import Dict, Any, Optional, Tuple, List

class MemoryPairEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = MemoryObservationPolicy()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        with open("./config.yaml", 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        elif mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
            
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        with open(f"./levels/{world_id}.yaml", 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = MemoryGenerator(str(self.env_id), self.configs.get('generator', {}))
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        
        action_name = action.get('action')
        params = action.get('params', {})
        
        if action_name != 'flip':
            self._last_action_result = "invalid_action"
            return self._state
            
        position = params.get('position')
        if position is None or position < 0 or position > 15:
            self._last_action_result = "invalid_position"
            return self._state
        
        if self._state['game']['card_states'][position] == 2:
            self._last_action_result = "illegal_move"
            self._state['agent']['steps_remaining'] -= 1
            return self._state
        
        if self._state['game']['card_states'][position] == 0:  # Face-down card
            self._state['game']['card_states'][position] = 1  # Reveal it
            self._state['game']['revealed_positions'].append(position)
            self._state['game']['current_revealed_symbol'] = self._state['game']['cards'][position]
            
            if position not in self._state['game']['explored_positions']:
                self._state['game']['explored_positions'].append(position)
        
        revealed_positions = self._state['game']['revealed_positions']
        if len(revealed_positions) == 2:
            pos1, pos2 = revealed_positions
            symbol1 = self._state['game']['cards'][pos1]
            symbol2 = self._state['game']['cards'][pos2]
            
            if symbol1 == symbol2:
                self._state['game']['card_states'][pos1] = 2
                self._state['game']['card_states'][pos2] = 2
                self._state['game']['cleared_pairs'] += 1
                self._last_action_result = "pair_cleared"
            else:
                self._state['game']['card_states'][pos1] = 0
                self._state['game']['card_states'][pos2] = 0
                self._last_action_result = "no_match"
            
            self._state['game']['revealed_positions'] = []
        
        self._state['agent']['steps_remaining'] -= 1
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if self._last_action_result == "pair_cleared":
            return 1.0, ["pair_cleared"], {"reason": "Successfully matched a pair"}
        
        position = action.get('params', {}).get('position')
        if (position is not None and 
            len(self._history) > 0 and 
            position not in self._history[-1]['game']['explored_positions']):
            return 0.05, ["first_exploration"], {"reason": "First time exploring this position"}
        
        return 0.0, ["no_reward"], {"reason": "No reward conditions met"}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> Any:
        grid_str = ""
        card_states = omega['card_states']
        for i in range(4):
            row = []
            for j in range(4):
                pos = i * 4 + j
                row.append(str(card_states[pos]))
            grid_str += " ".join(row) + "\n"
        
        template = f"""Memory Pair Matching - Step {omega['t']}/{self.configs['termination']['max_steps']}
Steps remaining: {omega['steps_remaining']}
Cleared pairs: {self._state['game']['cleared_pairs']}/8
Current revealed symbol: {omega['current_revealed_symbol']}

Grid (0=face-down, 1=revealed, 2=cleared):
{grid_str.strip()}

Available action: flip(position) where position is 0-15"""
        
        return template
    
    def done(self, state=None) -> bool:
        max_steps = self._state.get('globals', {}).get('max_steps', self.configs["termination"]["max_steps"])
        return (self._state['game']['cleared_pairs'] >= 8 or 
                self._state['agent']['steps_remaining'] <= 0)