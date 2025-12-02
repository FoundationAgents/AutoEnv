from base.env.base_env import SkinEnv
from env_obs import PartialCardVisibilityPolicy
from env_generate import MemoryWorldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class MismatchedMemoryEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = PartialCardVisibilityPolicy(show_face_up_only=True)
        super().__init__(env_id, obs_policy)
        self._dsl_config()  # Initialize configs
    
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            world_state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            world_state = self._load_world(world_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self._state = world_state
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = MemoryWorldGenerator(str(self.env_id), self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action.get("action")
        params = action.get("params", {})
        
        if action_name == "FLIP":
            x = params.get("x")
            y = params.get("y")
            
            if x is not None and y is not None:
                if 0 <= x < 4 and 0 <= y < 4:
                    card_states = self._state["board"]["card_states"]
                    if card_states[x][y] == 0:
                        card_states[x][y] = 1
                        self._last_action_result = f"Flipped card at ({x},{y})"
                    else:
                        self._last_action_result = f"Card at ({x},{y}) already revealed or solved"
                else:
                    self._last_action_result = "Invalid coordinates"
            else:
                self._last_action_result = "Missing coordinates"
        
        elif action_name == "WAIT":
            self._last_action_result = "Waited"
        
        self._check_and_resolve_pairs()
        self._auto_flip_unpaired_cards()
        
        self._state["game"]["step_count"] += 1
        self._t += 1
        
        return self._state
    
    def _check_and_resolve_pairs(self):
        card_states = self._state["board"]["card_states"]
        cards = self._state["board"]["cards"]
        symbol_pairs = self._state["game"]["symbol_pairs"]
        
        face_up_positions = []
        for i in range(4):
            for j in range(4):
                if card_states[i][j] == 1:
                    face_up_positions.append((i, j))
        
        if len(face_up_positions) == 2:
            pos1, pos2 = face_up_positions
            symbol1 = cards[pos1[0]][pos1[1]]
            symbol2 = cards[pos2[0]][pos2[1]]
            
            if symbol_pairs.get(symbol1) == symbol2:
                card_states[pos1[0]][pos1[1]] = 2
                card_states[pos2[0]][pos2[1]] = 2
                self._state["game"]["discovered_pairs"] += 1
                self._last_action_result = f"Pair found: {symbol1}-{symbol2}!"
    
    def _auto_flip_unpaired_cards(self):
        card_states = self._state["board"]["card_states"]
        
        # Count face-up cards (state 1)
        face_up_count = 0
        for i in range(4):
            for j in range(4):
                if card_states[i][j] == 1:
                    face_up_count += 1
        
        # Only flip back if there are exactly 2 face-up cards
        # (meaning they didn't match, since matched pairs would be state 2)
        if face_up_count == 2:
            for i in range(4):
                for j in range(4):
                    if card_states[i][j] == 1:
                        card_states[i][j] = 0
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        reward = 0.0
        events = []
        rinfo = {}
        
        if len(self._history) > 0:
            prev_pairs = self._history[-1]["game"]["discovered_pairs"]
            curr_pairs = self._state["game"]["discovered_pairs"]
            
            if curr_pairs > prev_pairs:
                reward += 1.0
                events.append("pair_discovered")
                rinfo["pair_rewards"] = 1.0
        
        action_name = action.get("action")
        if action_name == "FLIP":
            params = action.get("params", {})
            x, y = params.get("x"), params.get("y")
            
            if x is not None and y is not None and 0 <= x < 4 and 0 <= y < 4:
                cards = self._state["board"]["cards"]
                symbol = cards[x][y]
                seen_symbols = self._state["game"]["seen_symbols"]
                
                if symbol not in seen_symbols:
                    reward += 0.05
                    events.append("new_symbol_seen")
                    rinfo["exploration_rewards"] = 0.05
                    seen_symbols.append(symbol)
        
        self._state["game"]["cumulative_reward"] += reward
        
        return reward, events, rinfo
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        step_count = omega.get("step_count", 0)
        max_steps = omega.get("max_steps", 40)
        discovered_pairs = omega.get("discovered_pairs", 0)
        total_pairs = omega.get("total_pairs", 8)
        cumulative_reward = omega.get("cumulative_reward", 0.0)
        card_states = omega.get("card_states", [])
        visible_symbols = omega.get("visible_symbols", {})
        
        output = f"Step {step_count}/{max_steps} | Pairs Found: {discovered_pairs}/{total_pairs} | Score: {cumulative_reward:.2f}\n\n"
        
        output += "Board State (0=face-down, 1=face-up, 2=solved):\n"
        if card_states:
            for row in card_states:
                output += " ".join(str(cell) for cell in row) + "\n"
        output += "\n"
        
        output += "Currently Visible Symbols:\n"
        if visible_symbols:
            for pos, symbol in visible_symbols.items():
                output += f"Position {pos}: {symbol}\n"
        else:
            output += "None\n"
        output += "\n"
        
        output += "Available Actions: FLIP(x,y) where x,y in [0,3], WAIT()"
        
        return output
    
    def done(self, state=None) -> bool:
        current_state = state if state is not None else self._state
        
        discovered_pairs = current_state["game"]["discovered_pairs"]
        total_pairs = current_state["globals"]["total_pairs"]
        step_count = current_state["game"]["step_count"]
        
        world_max_steps = current_state.get("globals", {}).get("max_steps")
        config_max_steps = self.configs["termination"]["max_steps"]
        max_steps = world_max_steps if world_max_steps is not None else config_max_steps
        
        return discovered_pairs >= total_pairs or step_count >= max_steps
