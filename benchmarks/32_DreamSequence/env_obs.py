from base.env.base_observation import ObservationPolicy
from typing import Dict, Any

class DreamObservationPolicy(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        agent = env_state['agent']
        world = env_state['world']
        globals_info = env_state['globals']
        
        current_room = agent['current_room']
        has_key = agent['has_key']
        
        # Get available exits from current room
        room_connections = world['connections'].get(current_room, {})
        available_exits = list(room_connections.keys())
        
        # Get room type and apply effects to available exits
        room_properties = world['rooms'].get(current_room, {})
        room_type = room_properties.get('type', 'Normal')
        
        # Check if key is in current room
        key_in_room = (world['key_location'] == current_room and not has_key)
        
        # Calculate steps remaining
        steps_remaining = globals_info['max_steps'] - t

        return {
            'current_room': current_room,
            'has_key': has_key,
            'available_exits': available_exits,
            'room_type': room_type,
            'key_in_room': key_in_room,
            'steps_remaining': steps_remaining,
            't': t + 1
        }
