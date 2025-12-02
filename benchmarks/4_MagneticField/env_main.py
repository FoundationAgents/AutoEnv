from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import FullGridObserver
from env_generate import MagneticFieldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class MagneticFieldEnv(SkinEnv):
    def __init__(self, env_id: str):
        obs_policy = FullGridObserver()
        super().__init__(env_id, obs_policy)
        self.hex_chars = '0123456789ABCDEF'
    
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = MagneticFieldGenerator(self.env_id, self.configs)
        world_id = generator.generate(seed=seed)
        return world_id
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            generated_world_id = self._generate_world(seed)
            self._state = self._load_world(generated_world_id)
        elif mode == "load":
            self._state = self._load_world(world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        self._t = 0
        self._history = []
        return self._state
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action')
        params = action.get('params', {})
        
        if action_name == "INPUT_HEX":
            hex_value = params.get('hex_value')
            if hex_value is not None and 0 <= hex_value <= 15:
                self._state['agent']['answer_slots'][self._state['agent']['cursor_pos']] = self.hex_chars[hex_value]
            self._state['step_count'] += 1
            
        elif action_name == "MOVE_CURSOR_RIGHT":
            self._state['agent']['cursor_pos'] = (self._state['agent']['cursor_pos'] + 1) % 4
            self._state['step_count'] += 1
            
        elif action_name == "MOVE_CURSOR_LEFT":
            self._state['agent']['cursor_pos'] = (self._state['agent']['cursor_pos'] - 1) % 4
            self._state['step_count'] += 1
            
        elif action_name == "SUBMIT_ANSWER":
            self._state['agent']['submitted'] = True
            self._state['step_count'] += 1
        
        return self._state
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if self._state['agent']['submitted'] or self._state['step_count'] >= self._state['globals']['max_steps']:
            submitted_answer = ''.join(self._state['agent']['answer_slots'])
            correct_answer = self._state['grid']['encoded_message']
            
            # Calculate partial rewards for correct individual characters
            correct_chars = sum(1 for i in range(4) if i < len(submitted_answer) and 
                              i < len(correct_answer) and submitted_answer[i] == correct_answer[i])
            
            # Enhanced reward structure: more points for using the free hints correctly
            reward_value = 0.0
            char_rewards = []
            
            for i in range(4):
                if i < len(submitted_answer) and i < len(correct_answer):
                    if submitted_answer[i] == correct_answer[i]:
                        # First two characters are "free" - give bonus points for using them
                        if i < 2:
                            char_reward = 0.3  # Bonus for using free hints
                        else:
                            char_reward = 0.2  # Regular points for figuring out the hard ones
                        reward_value += char_reward
                        char_rewards.append(char_reward)
                    else:
                        char_rewards.append(0.0)
            
            # Perfect answer bonus
            is_perfect = submitted_answer == correct_answer
            if is_perfect:
                reward_value += 0.2  # Bonus for perfect answer (total = 1.2 max)
            
            events = ["answer_submitted"]
            metadata = {
                "correct": is_perfect,
                "correct_characters": correct_chars,
                "submitted": submitted_answer,
                "expected": correct_answer,
                "character_rewards": char_rewards,
                "using_free_hints_correctly": correct_chars >= 2  # Did they use the free first 2 chars?
            }
            
            return reward_value, events, metadata
        
        return 0.0, [], {}
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        grid_display = ""
        for row in omega['grid_pattern']:
            grid_display += " ".join(str(cell) for cell in row) + "\n"
        
        answer_display = ""
        for i, slot in enumerate(omega['answer_slots']):
            if i == omega['cursor_pos']:
                answer_display += f"[{slot if slot else '_'}] "
            else:
                answer_display += f"{slot if slot else '_'} "
        
        # Include the encoding hints in the agent's observation!
        encoding_hints = omega.get('encoding_hints', {})
        
        # Create a user-friendly display of the most important hints
        hint_display = ""
        if 'MAJOR_HINT_first_two_characters' in encoding_hints:
            first_two = encoding_hints['MAJOR_HINT_first_two_characters']
            hint_display += f"\n🎯 FREE CHARACTERS: The first two characters are '{first_two}' (guaranteed points!)"
        
        if 'learning_tip' in encoding_hints:
            hint_display += f"\n💡 {encoding_hints['learning_tip']}"
        
        if 'step_by_step_example' in encoding_hints and encoding_hints['step_by_step_example']:
            example = encoding_hints['step_by_step_example'][0]
            hint_display += f"\n📋 EXAMPLE: Pattern {example['pattern']} → Binary {example['binary_value']} → Hex '{example['hex_character']}'"
        
        if 'example_2x2_from_grid' in encoding_hints:
            examples = encoding_hints['example_2x2_from_grid'][:3]  # Show first 3 examples
            hint_display += f"\n🔍 CURRENT GRID PATTERNS:"
            for ex in examples:
                hint_display += f"\n   {ex['explanation']}"
        
        return f"""=== MAGNETIC FIELD DECODER - STEP {omega['t'] + 1}/{omega['max_steps']} ===

GRID PATTERN:
{grid_display.strip()}

ANSWER INPUT:
{answer_display.strip()}

HINTS & GUIDANCE:{hint_display}

AVAILABLE ACTIONS:
- INPUT_HEX(0-15): Input hexadecimal digit at cursor position
- MOVE_CURSOR_RIGHT: Move cursor to next position
- MOVE_CURSOR_LEFT: Move cursor to previous position  
- SUBMIT_ANSWER: Submit your complete 4-character answer"""
    
    def done(self, state=None) -> bool:
        return (self._state['step_count'] >= self._state['globals']['max_steps'] or 
                self._state['agent']['submitted'])
