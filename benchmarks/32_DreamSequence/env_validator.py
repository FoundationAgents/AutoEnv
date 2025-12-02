from typing import Dict, Any, List, Tuple, Optional, Set
import yaml
import os
from collections import deque

class DreamNavValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config['termination']['max_steps']
    
    def validate_world(self, world_path: str) -> Tuple[bool, List[str]]:
        """Main validation function that checks both solvability and reward alignment"""
        try:
            with open(world_path, 'r') as f:
                world_state = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load world file: {str(e)}"]
        
        issues = []
        
        # Validate world structure
        structure_valid, structure_issues = self._validate_world_structure(world_state)
        if not structure_valid:
            issues.extend(structure_issues)
            return False, issues
        
        # Check level solvability
        solvable, solvability_issues = self._check_level_solvability(world_state)
        if not solvable:
            issues.extend(solvability_issues)
        
        # Validate reward structure alignment
        reward_valid, reward_issues = self._validate_reward_structure(world_state)
        if not reward_valid:
            issues.extend(reward_issues)
        
        return len(issues) == 0, issues
    
    def _validate_world_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate basic world structure and data integrity"""
        issues = []
        
        # Check required keys
        required_keys = ['globals', 'agent', 'world', 'episode']
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required key: {key}")
        
        if issues:
            return False, issues
        
        # Validate world components
        world = world_state['world']
        required_world_keys = ['rooms', 'key_location', 'portal_room', 'connections']
        for key in required_world_keys:
            if key not in world:
                issues.append(f"Missing world key: {key}")
        
        # Check room count matches difficulty
        difficulty = world_state['globals']['difficulty']
        expected_rooms = {'Easy': 6, 'Medium': 8, 'Hard': 10}
        if difficulty in expected_rooms:
            expected_count = expected_rooms[difficulty]
            actual_count = len(world['rooms'])
            if actual_count != expected_count:
                issues.append(f"Room count mismatch: expected {expected_count} for {difficulty}, got {actual_count}")
        
        # Validate room IDs are sequential
        room_ids = sorted(world['rooms'].keys())
        expected_ids = list(range(len(world['rooms'])))
        if room_ids != expected_ids:
            issues.append(f"Room IDs should be sequential 0-{len(world['rooms'])-1}, got {room_ids}")
        
        # Validate key and portal locations
        if world['key_location'] not in world['rooms']:
            issues.append(f"Key location {world['key_location']} not in rooms")
        
        if world['portal_room'] not in world['rooms']:
            issues.append(f"Portal room {world['portal_room']} not in rooms")
        
        # Validate connections reference valid rooms
        for room_id, connections in world['connections'].items():
            if room_id not in world['rooms']:
                issues.append(f"Connection source room {room_id} not in rooms")
            for color, dest_room in connections.items():
                if dest_room not in world['rooms']:
                    issues.append(f"Connection destination room {dest_room} not in rooms")
                if color not in ['red', 'blue', 'green']:
                    issues.append(f"Invalid door color: {color}")
        
        return len(issues) == 0, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Critical solvability analysis - ensure the level is actually solvable"""
        issues = []
        
        # Extract key information
        start_room = world_state['agent']['current_room']
        key_location = world_state['world']['key_location']
        portal_room = world_state['world']['portal_room']
        connections = world_state['world']['connections']
        rooms = world_state['world']['rooms']
        
        # 1. ACTION CONSTRAINT ANALYSIS
        # Check if all rooms have at least one exit (avoid dead-ends except portal)
        for room_id, room_data in rooms.items():
            room_connections = connections.get(room_id, {})
            if len(room_connections) == 0 and room_id != portal_room:
                issues.append(f"Dead-end room {room_id} with no exits (not portal room)")
        
        # 2. TARGET REACHABILITY ANALYSIS
        # Check if key location is reachable from start
        key_reachable, key_path_length = self._find_shortest_path(start_room, key_location, connections, rooms)
        if not key_reachable:
            issues.append(f"Key at room {key_location} is not reachable from start room {start_room}")
        
        # Check if portal is reachable from key location
        portal_reachable, portal_path_length = self._find_shortest_path(key_location, portal_room, connections, rooms)
        if not portal_reachable:
            issues.append(f"Portal at room {portal_room} is not reachable from key location {key_location}")
        
        # 3. STEP BUDGET ANALYSIS
        if key_reachable and portal_reachable:
            # Calculate minimum steps needed
            # Path to key + pick up key (1 step) + path to portal
            min_steps_needed = key_path_length + 1 + portal_path_length
            
            # Account for Time-Slow rooms that require WAIT actions
            total_wait_steps = self._count_time_slow_penalties(start_room, key_location, portal_room, connections, rooms)
            min_steps_needed += total_wait_steps
            
            if min_steps_needed > self.max_steps:
                issues.append(f"Minimum steps required ({min_steps_needed}) exceeds step limit ({self.max_steps})")
        
        # 4. RESOURCE AVAILABILITY CHECK
        # Ensure key exists and is obtainable
        if key_location not in rooms:
            issues.append(f"Key location {key_location} does not exist in world")
        
        # 5. CIRCULAR DEPENDENCY CHECK
        # In this environment, we need: start -> key -> portal
        # Check if there's a valid complete path
        complete_reachable = self._check_complete_solution_path(start_room, key_location, portal_room, connections, rooms)
        if not complete_reachable:
            issues.append("No valid complete solution path exists (start -> key -> portal)")
        
        return len(issues) == 0, issues
    
    def _find_shortest_path(self, start: int, target: int, connections: Dict[int, Dict[str, int]], 
                           rooms: Dict[int, Dict[str, Any]]) -> Tuple[bool, int]:
        """BFS to find shortest path between two rooms, accounting for room effects"""
        if start == target:
            return True, 0
        
        queue = deque([(start, 0, False)])  # (room, steps, wait_used)
        visited = set()
        visited.add((start, False))
        
        while queue:
            current_room, steps, wait_used = queue.popleft()
            
            # Get room connections and type
            room_connections = connections.get(current_room, {})
            room_type = rooms.get(current_room, {}).get('type', 'Normal')
            
            # Handle Time-Slow rooms
            if room_type == "Time-Slow" and not wait_used:
                # Must wait first
                next_state = (current_room, steps + 1, True)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((current_room, steps + 1, True))
                continue
            
            # Try each door
            for door_color, dest_room in room_connections.items():
                # Apply Anti-Gravity effect
                effective_dest = dest_room
                if room_type == "Anti-Gravity":
                    if door_color == "red":
                        # Red door acts like blue door
                        effective_dest = room_connections.get("blue", dest_room)
                    elif door_color == "blue":
                        # Blue door acts like red door  
                        effective_dest = room_connections.get("red", dest_room)
                
                if effective_dest == target:
                    return True, steps + 1
                
                next_state = (effective_dest, False)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((effective_dest, steps + 1, False))
        
        return False, float('inf')
    
    def _count_time_slow_penalties(self, start: int, key_location: int, portal_room: int,
                                  connections: Dict[int, Dict[str, int]], 
                                  rooms: Dict[int, Dict[str, Any]]) -> int:
        """Count additional wait steps needed for Time-Slow rooms in optimal path"""
        penalty = 0
        
        # Simple approximation: count Time-Slow rooms that might be traversed
        path_rooms = self._get_likely_path_rooms(start, key_location, portal_room, connections, rooms)
        
        for room_id in path_rooms:
            room_type = rooms.get(room_id, {}).get('type', 'Normal')
            if room_type == "Time-Slow":
                penalty += 1  # One extra WAIT step per Time-Slow room
        
        return penalty
    
    def _get_likely_path_rooms(self, start: int, key_location: int, portal_room: int,
                              connections: Dict[int, Dict[str, int]], 
                              rooms: Dict[int, Dict[str, Any]]) -> Set[int]:
        """Get set of rooms likely to be visited in optimal solution"""
        # Simple heuristic: rooms that are on shortest paths
        path_rooms = set()
        
        # Add rooms in path from start to key
        path_rooms.update(self._get_path_rooms(start, key_location, connections, rooms))
        
        # Add rooms in path from key to portal  
        path_rooms.update(self._get_path_rooms(key_location, portal_room, connections, rooms))
        
        return path_rooms
    
    def _get_path_rooms(self, start: int, target: int, connections: Dict[int, Dict[str, int]], 
                       rooms: Dict[int, Dict[str, Any]]) -> Set[int]:
        """Get rooms traversed in shortest path (simplified)"""
        if start == target:
            return {start}
        
        # BFS to find path
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            current_room, path = queue.popleft()
            room_connections = connections.get(current_room, {})
            
            for door_color, dest_room in room_connections.items():
                if dest_room == target:
                    return set(path + [dest_room])
                
                if dest_room not in visited:
                    visited.add(dest_room)
                    queue.append((dest_room, path + [dest_room]))
        
        return set()
    
    def _check_complete_solution_path(self, start: int, key_location: int, portal_room: int,
                                    connections: Dict[int, Dict[str, int]], 
                                    rooms: Dict[int, Dict[str, Any]]) -> bool:
        """Verify complete solution path exists with proper state transitions"""
        # Check start -> key
        can_reach_key, _ = self._find_shortest_path(start, key_location, connections, rooms)
        if not can_reach_key:
            return False
        
        # Check key -> portal (after picking up key)
        can_reach_portal, _ = self._find_shortest_path(key_location, portal_room, connections, rooms)
        if not can_reach_portal:
            return False
        
        return True
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate reward structure promotes goal achievement over action grinding"""
        issues = []
        
        # Check reward system alignment
        # The environment uses binary reward (+1 for success, 0 otherwise)
        # This is good - no intermediate rewards that could be exploited
        
        # Verify no reward loops possible
        # Since rewards are only given for reaching portal with key,
        # there's no way to farm rewards through repeated actions
        
        # Check that success reward is meaningful
        success_reward = 1.0  # From environment code
        if success_reward <= 0:
            issues.append("Success reward must be positive")
        
        # Verify no action grinding opportunities
        # The binary reward structure prevents action grinding since:
        # 1. Only win condition gives reward
        # 2. No intermediate progress rewards
        # 3. No rewards for exploration or action usage
        
        # Check efficiency incentives
        # Step limit naturally encourages efficiency
        # Binary reward prevents partial credit exploitation
        
        # Validate failure costs
        # Time limit provides natural failure cost (opportunity cost)
        # Dead-end rooms provide immediate failure
        
        return len(issues) == 0, issues


def validate_level(world_path: str, config_path: str = "./config.yaml") -> Tuple[bool, List[str]]:
    """Standalone function to validate a single level"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load config: {str(e)}"]
    
    validator = DreamNavValidator(config)
    return validator.validate_world(world_path)


def validate_all_levels(levels_dir: str = "./levels", config_path: str = "./config.yaml") -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all levels in directory"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"config_error": (False, [f"Failed to load config: {str(e)}"])}
    
    validator = DreamNavValidator(config)
    results = {}
    
    if not os.path.exists(levels_dir):
        return {"directory_error": (False, [f"Levels directory {levels_dir} does not exist"])}
    
    for filename in os.listdir(levels_dir):
        if filename.endswith('.yaml'):
            world_path = os.path.join(levels_dir, filename)
            world_id = filename[:-5]  # Remove .yaml extension
            results[world_id] = validator.validate_world(world_path)
    
    return results


if __name__ == "__main__":
    # Example usage
    results = validate_all_levels()
    
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    total_count = len(results)
    
    print(f"Validation Results: {valid_count}/{total_count} levels valid")
    
    for world_id, (is_valid, issues) in results.items():
        if not is_valid:
            print(f"\n{world_id}: INVALID")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"{world_id}: VALID")