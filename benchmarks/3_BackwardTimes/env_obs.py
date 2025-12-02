from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class FullTimelineAccessPolicy(ObservationPolicy):
    def __init__(self, show_future_events=False, show_unvalidated_connections=True):
        self.show_future_events = show_future_events
        self.show_unvalidated_connections = show_unvalidated_connections
    
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        current_time = env_state.get('agent', {}).get('current_time_index', 40)
        
        # Filter timeline ledger to hide future events
        timeline_ledger = env_state.get('investigation', {}).get('timeline_ledger', {})
        filtered_ledger = {}
        if not self.show_future_events:
            for time_idx, events in timeline_ledger.items():
                if isinstance(time_idx, (int, str)) and int(time_idx) <= current_time:
                    filtered_ledger[time_idx] = events
        else:
            filtered_ledger = timeline_ledger
        
        observation = {
            'current_time_index': current_time,
            'timeline_ledger': filtered_ledger,
            'unresolved_effects': env_state.get('investigation', {}).get('unresolved_effects', []),
            'collected_clues': env_state.get('investigation', {}).get('collected_clues', []),
            'suspect_list': env_state.get('crime_scene', {}).get('suspect_list', []),
            'validated_connections': env_state.get('timeline', {}).get('validated_connections', {}),
            'max_steps': env_state.get('globals', {}).get('max_steps', 40),
            't': t + 1  # Display step counter starting from 1 instead of 0
        }
        
        if self.show_unvalidated_connections:
            observation['proposed_connections'] = env_state.get('investigation', {}).get('proposed_connections', [])
        
        return observation