from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import uuid
from collections import deque

class DreamGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Load state template from config
        base_state = self.config['state_template']
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID and save
        world_id = self._generate_world_id(seed)
        if save_path is None:
            save_path = f"./levels/{world_id}.yaml"
        
        self._save_world(world_state, save_path)
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        # Deep copy base state
        state = {
            'globals': base_state['globals'].copy(),
            'agent': base_state['agent'].copy(),
            'world': {
                'rooms': {},
                'key_location': 0,
                'portal_room': 5,
                'connections': {}
            },
            'episode': base_state['episode'].copy()
        }
        
        # Setup difficulty and rooms
        self._setup_difficulty_rooms(state)
        
        # Generate room connections with guaranteed connectivity
        self._generate_room_connections(state)
        
        # Assign room properties
        self._assign_room_properties(state)
        
        # Place key and set portal
        self._place_key_and_portal(state)
        
        # Select start room
        self._select_start_room(state)
        
        # Validate and fix connectivity if needed
        self._ensure_solvability(state)
        
        return state
    
    def _setup_difficulty_rooms(self, state: Dict[str, Any]):
        difficulty = state['globals']['difficulty']
        
        if difficulty == "Easy":
            num_rooms = 6
            portal_room = 5
        elif difficulty == "Medium":
            num_rooms = 8
            portal_room = 7
        else:  # Hard
            num_rooms = 10
            portal_room = 9
        
        # Initialize rooms
        for i in range(num_rooms):
            state['world']['rooms'][i] = {'type': 'Normal'}
        
        state['world']['portal_room'] = portal_room
    
    def _generate_room_connections(self, state: Dict[str, Any]):
        rooms = state['world']['rooms']
        connections = {}
        colors = ['red', 'blue', 'green']
        
        # Initialize connections
        for room_id in rooms.keys():
            connections[room_id] = {}
        
        # Create a strongly connected graph using circular approach
        room_list = list(rooms.keys())
        
        # Create circular connectivity (each room connects to next)
        for i in range(len(room_list)):
            current_room = room_list[i]
            next_room = room_list[(i + 1) % len(room_list)]  # Circular
            
            # Forward connection
            color1 = random.choice(colors)
            connections[current_room][color1] = next_room
            
            # Backward connection with different color
            available_colors = [c for c in colors if c not in connections[next_room]]
            if available_colors:
                color2 = random.choice(available_colors)
                connections[next_room][color2] = current_room
        
        # Add additional random connections to increase connectivity
        for room_id in rooms.keys():
            current_doors = len(connections[room_id])
            max_additional = min(3 - current_doors, 1)  # More conservative
            additional_doors = random.randint(0, max_additional)
            
            for _ in range(additional_doors):
                available_colors = [c for c in colors if c not in connections[room_id]]
                if not available_colors:
                    break
                
                color = random.choice(available_colors)
                possible_destinations = [r for r in rooms.keys() if r != room_id]
                if possible_destinations:
                    destination = random.choice(possible_destinations)
                    connections[room_id][color] = destination
        
        state['world']['connections'] = connections
    
    def _assign_room_properties(self, state: Dict[str, Any]):
        rooms = state['world']['rooms']
        room_types = ['Normal', 'Anti-Gravity', 'Time-Slow']
        
        # Ensure at least one normal room (the starting room will be normal)
        normal_room = random.choice(list(rooms.keys()))
        rooms[normal_room]['type'] = 'Normal'
        
        # Assign random types to other rooms (but not too many special rooms)
        special_room_count = 0
        max_special_rooms = max(1, len(rooms) // 3)  # At most 1/3 special rooms
        
        for room_id in rooms.keys():
            if room_id != normal_room:
                if special_room_count < max_special_rooms and random.random() < 0.4:
                    rooms[room_id]['type'] = random.choice(['Anti-Gravity', 'Time-Slow'])
                    special_room_count += 1
                else:
                    rooms[room_id]['type'] = 'Normal'
    
    def _place_key_and_portal(self, state: Dict[str, Any]):
        rooms = list(state['world']['rooms'].keys())
        portal_room = state['world']['portal_room']
        
        # Place key in random room (excluding portal room)
        available_rooms = [r for r in rooms if r != portal_room]
        key_location = random.choice(available_rooms)
        state['world']['key_location'] = key_location
    
    def _select_start_room(self, state: Dict[str, Any]):
        rooms = list(state['world']['rooms'].keys())
        start_room = random.choice(rooms)
        state['agent']['current_room'] = start_room
        state['episode']['start_room'] = start_room
    
    def _ensure_solvability(self, state: Dict[str, Any]):
        """Ensure the generated level is solvable by fixing connectivity issues"""
        start_room = state['agent']['current_room']
        key_location = state['world']['key_location']
        portal_room = state['world']['portal_room']
        connections = state['world']['connections']
        
        # Check if key is reachable from start
        if not self._is_reachable(start_room, key_location, connections):
            self._fix_connectivity(start_room, key_location, connections)
        
        # Check if portal is reachable from key location
        if not self._is_reachable(key_location, portal_room, connections):
            self._fix_connectivity(key_location, portal_room, connections)
    
    def _is_reachable(self, start: int, target: int, connections: Dict[int, Dict[str, int]]) -> bool:
        """Check if target is reachable from start"""
        if start == target:
            return True
        
        visited = set()
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target:
                return True
            
            # Add all destinations from current room
            for dest in connections.get(current, {}).values():
                if dest not in visited:
                    queue.append(dest)
        
        return False
    
    def _fix_connectivity(self, start: int, target: int, connections: Dict[int, Dict[str, int]]):
        """Fix connectivity by adding a direct connection"""
        colors = ['red', 'blue', 'green']
        
        # Find available color for start room
        available_colors = [c for c in colors if c not in connections[start]]
        if available_colors:
            color = available_colors[0]
            connections[start][color] = target
        else:
            # Replace a random connection
            color = random.choice(colors)
            connections[start][color] = target
    
    def _save_world(self, world_state: Dict[str, Any], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"world_seed_{seed}"
        else:
            return f"world_{str(uuid.uuid4())[:8]}"
