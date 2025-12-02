from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class PartialCardVisibilityPolicy(ObservationPolicy):
    def __init__(self, show_face_up_only=True):
        self.show_face_up_only = show_face_up_only
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        board = env_state.get("board", {})
        game = env_state.get("game", {})
        globals_data = env_state.get("globals", {})
        
        card_states = board.get("card_states", [])
        cards = board.get("cards", [])
        
        visible_symbols = {}
        
        if self.show_face_up_only and len(card_states) > 0 and len(cards) > 0:
            for i in range(len(card_states)):
                for j in range(len(card_states[0])):
                    if card_states[i][j] == 1:
                        visible_symbols[f"{i},{j}"] = cards[i][j]
        
        return {
            "card_states": card_states,
            "visible_symbols": visible_symbols,
            "step_count": game.get("step_count", 0),
            "max_steps": globals_data.get("max_steps", 40),
            "discovered_pairs": game.get("discovered_pairs", 0),
            "total_pairs": globals_data.get("total_pairs", 8),
            "cumulative_reward": game.get("cumulative_reward", 0.0),
            "t": t + 1
        }