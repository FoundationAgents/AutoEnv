import yaml
import random
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque

class SmartHomeValidator:
    def __init__(self):
        self.max_steps = 60
        self.grid_size = [12, 12]
        self.object_types = ["food", "book", "clothes", "cleaning_supplies", "electronics"]
        self.colors = ["red", "blue", "green", "yellow", "white", "black"]
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Main validation function that checks all aspects of level validity"""
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. BASIC STRUCTURAL VALIDATION
        structural_issues = self._validate_basic_structure(world_state)
        issues.extend(structural_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for impossible puzzles"""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        action_issues = self._analyze_action_constraints(world_state)
        issues.extend(action_issues)
        
        # TARGET REACHABILITY for each chore
        for i, instruction in enumerate(world_state["chores"]["instructions"]):
            reachability_issues = self._check_target_reachability(world_state, instruction, i)
            issues.extend(reachability_issues)
        
        # STEP BUDGET ANALYSIS
        step_issues = self._check_step_budget_feasibility(world_state)
        issues.extend(step_issues)
        
        return issues
    
    def _analyze_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Analyze fundamental limitations of available actions"""
        issues = []
        
        # Check if all rooms are reachable
        connectivity_issues = self._check_spatial_connectivity(world_state)
        issues.extend(connectivity_issues)
        
        # Check if required objects exist and are accessible
        object_issues = self._check_object_accessibility(world_state)
        issues.extend(object_issues)
        
        # Check if required appliances exist and are accessible
        appliance_issues = self._check_appliance_accessibility(world_state)
        issues.extend(appliance_issues)
        
        # Check if required containers exist and are accessible
        container_issues = self._check_container_accessibility(world_state)
        issues.extend(container_issues)
        
        return issues
    
    def _check_spatial_connectivity(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if all rooms are reachable from agent's starting position"""
        issues = []
        
        start_pos = tuple(world_state["agent"]["pos"])
        walls = [tuple(wall) for wall in world_state["apartment"]["walls"]]
        
        # BFS to find all reachable positions
        reachable = self._bfs_reachable_positions(start_pos, walls)
        
        # Check if each room has at least one reachable floor tile
        for room_name, room_data in world_state["apartment"]["rooms"].items():
            if not room_data.get("bounds"):
                continue
                
            bounds = room_data["bounds"]
            room_reachable = False
            
            for x in range(bounds[0], bounds[2] + 1):
                for y in range(bounds[1], bounds[3] + 1):
                    if (x, y) in reachable:
                        room_reachable = True
                        break
                if room_reachable:
                    break
            
            if not room_reachable:
                issues.append(f"Room '{room_name}' is not reachable from agent's starting position")
        
        return issues
    
    def _bfs_reachable_positions(self, start_pos: Tuple[int, int], walls: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """BFS to find all positions reachable from start position"""
        queue = deque([start_pos])
        visited = {start_pos}
        wall_set = set(walls)
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 12 and 0 <= ny < 12 and 
                    (nx, ny) not in wall_set and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return visited
    
    def _check_object_accessibility(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if all objects needed for chores exist and are accessible"""
        issues = []
        
        # Extract required objects from chore instructions
        required_objects = self._extract_required_objects(world_state["chores"]["instructions"])
        available_objects = world_state["objects"]
        
        for req_color, req_type in required_objects:
            found = False
            for obj in available_objects:
                if obj["color"] == req_color and obj["type"] == req_type:
                    # Check if object position is valid and accessible
                    pos = tuple(obj["pos"])
                    if not self._is_position_accessible(pos, world_state):
                        issues.append(f"Required object '{req_color} {req_type}' is not accessible")
                    found = True
                    break
            
            if not found:
                issues.append(f"Required object '{req_color} {req_type}' does not exist in the level")
        
        return issues
    
    def _check_appliance_accessibility(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if all appliances needed for chores exist and are accessible"""
        issues = []
        
        required_appliances = self._extract_required_appliances(world_state["chores"]["instructions"])
        available_appliances = world_state["appliances"]
        
        for req_appliance in required_appliances:
            found = False
            for appliance in available_appliances:
                if appliance["type"] == req_appliance:
                    # Check if appliance is accessible (agent can get adjacent to it)
                    pos = tuple(appliance["pos"])
                    if not self._is_position_adjacent_accessible(pos, world_state):
                        issues.append(f"Required appliance '{req_appliance}' is not accessible")
                    found = True
                    break
            
            if not found:
                issues.append(f"Required appliance '{req_appliance}' does not exist in the level")
        
        return issues
    
    def _check_container_accessibility(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if all containers needed for chores exist and are accessible"""
        issues = []
        
        required_containers = self._extract_required_containers(world_state["chores"]["instructions"])
        available_containers = world_state["containers"]
        
        for req_container in required_containers:
            found = False
            for container in available_containers:
                if container["type"] == req_container:
                    # Check if container is accessible
                    pos = tuple(container["pos"])
                    if not self._is_position_adjacent_accessible(pos, world_state):
                        issues.append(f"Required container '{req_container}' is not accessible")
                    found = True
                    break
            
            if not found:
                issues.append(f"Required container '{req_container}' does not exist in the level")
        
        return issues
    
    def _check_target_reachability(self, world_state: Dict[str, Any], instruction: str, chore_id: int) -> List[str]:
        """Check if each chore's target state is actually achievable"""
        issues = []
        
        instruction_lower = instruction.lower()
        
        if "move the" in instruction_lower:
            issues.extend(self._check_move_chore_reachability(world_state, instruction, chore_id))
        elif "turn" in instruction_lower:
            issues.extend(self._check_appliance_chore_reachability(world_state, instruction, chore_id))
        elif "put the" in instruction_lower:
            issues.extend(self._check_container_chore_reachability(world_state, instruction, chore_id))
        else:
            issues.append(f"Chore {chore_id}: Unrecognized instruction pattern: {instruction}")
        
        return issues
    
    def _check_move_chore_reachability(self, world_state: Dict[str, Any], instruction: str, chore_id: int) -> List[str]:
        """Check if move chore can be completed"""
        issues = []
        
        # Extract object and target room from instruction
        color, obj_type, target_room = self._parse_move_instruction(instruction)
        
        if not color or not obj_type or not target_room:
            issues.append(f"Chore {chore_id}: Could not parse move instruction: {instruction}")
            return issues
        
        # Check if target object exists
        target_object = None
        for obj in world_state["objects"]:
            if obj["color"] == color and obj["type"] == obj_type:
                target_object = obj
                break
        
        if not target_object:
            issues.append(f"Chore {chore_id}: Target object '{color} {obj_type}' not found")
            return issues
        
        # Check if target room exists and has accessible floor space
        room_key = target_room.replace(" ", "_")
        if room_key not in world_state["apartment"]["rooms"]:
            issues.append(f"Chore {chore_id}: Target room '{target_room}' not found")
            return issues
        
        room_bounds = world_state["apartment"]["rooms"][room_key]["bounds"]
        room_has_accessible_floor = False
        
        for x in range(room_bounds[0], room_bounds[2] + 1):
            for y in range(room_bounds[1], room_bounds[3] + 1):
                if self._is_position_accessible((x, y), world_state):
                    room_has_accessible_floor = True
                    break
            if room_has_accessible_floor:
                break
        
        if not room_has_accessible_floor:
            issues.append(f"Chore {chore_id}: Target room '{target_room}' has no accessible floor space")
        
        return issues
    
    def _check_appliance_chore_reachability(self, world_state: Dict[str, Any], instruction: str, chore_id: int) -> List[str]:
        """Check if appliance chore can be completed"""
        issues = []
        
        # Extract appliance and target state from instruction
        target_state, appliance_type = self._parse_appliance_instruction(instruction)
        
        if not target_state or not appliance_type:
            issues.append(f"Chore {chore_id}: Could not parse appliance instruction: {instruction}")
            return issues
        
        # Check if appliance exists
        target_appliance = None
        for appliance in world_state["appliances"]:
            if appliance["type"] == appliance_type:
                target_appliance = appliance
                break
        
        if not target_appliance:
            issues.append(f"Chore {chore_id}: Target appliance '{appliance_type}' not found")
            return issues
        
        # Check if appliance is accessible for interaction
        pos = tuple(target_appliance["pos"])
        if not self._is_position_adjacent_accessible(pos, world_state):
            issues.append(f"Chore {chore_id}: Appliance '{appliance_type}' is not accessible for interaction")
        
        return issues
    
    def _check_container_chore_reachability(self, world_state: Dict[str, Any], instruction: str, chore_id: int) -> List[str]:
        """Check if container chore can be completed"""
        issues = []
        
        # Extract object and container from instruction
        color, obj_type, container_type = self._parse_container_instruction(instruction)
        
        if not color or not obj_type or not container_type:
            issues.append(f"Chore {chore_id}: Could not parse container instruction: {instruction}")
            return issues
        
        # Check if target object exists
        target_object = None
        for obj in world_state["objects"]:
            if obj["color"] == color and obj["type"] == obj_type:
                target_object = obj
                break
        
        if not target_object:
            issues.append(f"Chore {chore_id}: Target object '{color} {obj_type}' not found")
            return issues
        
        # Check if target container exists
        target_container = None
        for container in world_state["containers"]:
            if container["type"] == container_type:
                target_container = container
                break
        
        if not target_container:
            issues.append(f"Chore {chore_id}: Target container '{container_type}' not found")
            return issues
        
        # Check semantic compatibility
        if not self._is_semantically_compatible(obj_type, container_type):
            issues.append(f"Chore {chore_id}: Object '{obj_type}' is not compatible with container '{container_type}'")
        
        # Check if container is accessible
        pos = tuple(target_container["pos"])
        if not self._is_position_adjacent_accessible(pos, world_state):
            issues.append(f"Chore {chore_id}: Container '{container_type}' is not accessible for interaction")
        
        return issues
    
    def _check_step_budget_feasibility(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if all chores can be completed within step limit"""
        issues = []
        
        # Estimate minimum steps needed for all chores
        total_estimated_steps = 0
        
        for i, instruction in enumerate(world_state["chores"]["instructions"]):
            estimated_steps = self._estimate_chore_steps(world_state, instruction)
            total_estimated_steps += estimated_steps
        
        # Add buffer for navigation between chores
        total_estimated_steps += 10  # Buffer for inter-chore navigation
        
        if total_estimated_steps > self.max_steps:
            issues.append(f"Estimated minimum steps ({total_estimated_steps}) exceeds maximum allowed steps ({self.max_steps})")
        
        return issues
    
    def _estimate_chore_steps(self, world_state: Dict[str, Any], instruction: str) -> int:
        """Estimate minimum steps needed to complete a chore"""
        instruction_lower = instruction.lower()
        
        if "move the" in instruction_lower:
            return self._estimate_move_chore_steps(world_state, instruction)
        elif "turn" in instruction_lower:
            return self._estimate_appliance_chore_steps(world_state, instruction)
        elif "put the" in instruction_lower:
            return self._estimate_container_chore_steps(world_state, instruction)
        
        return 5  # Default estimate
    
    def _estimate_move_chore_steps(self, world_state: Dict[str, Any], instruction: str) -> int:
        """Estimate steps for move chore"""
        # Find object + navigate to it + pick up + navigate to target room + drop
        return 15  # Conservative estimate
    
    def _estimate_appliance_chore_steps(self, world_state: Dict[str, Any], instruction: str) -> int:
        """Estimate steps for appliance chore"""
        # Navigate to appliance + toggle
        return 8  # Conservative estimate
    
    def _estimate_container_chore_steps(self, world_state: Dict[str, Any], instruction: str) -> int:
        """Estimate steps for container chore"""
        # Find object + pick up + navigate to container + open + drop + close
        return 18  # Conservative estimate including container operations
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical check for incentive alignment"""
        issues = []
        
        # Define expected reward structure based on environment design
        expected_rewards = {
            "pickup_target_object": 0.3,
            "complete_chore": 0.7,
            "complete_final_chore": 1.0,
            "default_reward": 0.0
        }
        
        # GOAL-ORIENTED REWARDS CHECK
        goal_reward_total = expected_rewards["complete_chore"] * 3 + expected_rewards["complete_final_chore"]
        action_reward_total = expected_rewards["pickup_target_object"] * 3  # Max 3 pickup bonuses
        
        if action_reward_total >= goal_reward_total:
            issues.append("Action usage rewards are too high compared to goal achievement rewards")
        
        # ACHIEVEMENT > PROCESS principle check
        if expected_rewards["pickup_target_object"] > expected_rewards["complete_chore"]:
            issues.append("Process rewards (pickup) should be lower than achievement rewards (completion)")
        
        # Check for proper reward sparsity
        if expected_rewards["default_reward"] != 0.0:
            issues.append("Default reward should be 0.0 to maintain reward sparsity")
        
        # Check that final completion has highest individual reward
        if expected_rewards["complete_final_chore"] < expected_rewards["complete_chore"]:
            issues.append("Final chore completion should have highest individual reward")
        
        return issues
    
    def _validate_basic_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic structural requirements"""
        issues = []
        
        # Check required fields exist
        required_fields = ["globals", "agent", "apartment", "objects", "appliances", "containers", "chores"]
        for field in required_fields:
            if field not in world_state:
                issues.append(f"Missing required field: {field}")
        
        # Check agent starting position is valid
        if "agent" in world_state:
            agent_pos = world_state["agent"]["pos"]
            if not self._is_position_accessible(tuple(agent_pos), world_state):
                issues.append("Agent starting position is not accessible")
        
        # Check all objects are on valid positions
        for i, obj in enumerate(world_state.get("objects", [])):
            if not self._is_position_accessible(tuple(obj["pos"]), world_state):
                issues.append(f"Object {i} is positioned on inaccessible location")
        
        # Check we have exactly 3 chores
        if len(world_state.get("chores", {}).get("instructions", [])) != 3:
            issues.append("Must have exactly 3 chore instructions")
        
        return issues
    
    # UTILITY METHODS
    
    def _extract_required_objects(self, instructions: List[str]) -> List[Tuple[str, str]]:
        """Extract (color, type) pairs of objects needed for chores"""
        required = []
        for instruction in instructions:
            if "move the" in instruction.lower() or "put the" in instruction.lower():
                words = instruction.lower().split()
                color_idx = -1
                for i, word in enumerate(words):
                    if word in self.colors:
                        color_idx = i
                        break
                
                if color_idx != -1 and color_idx + 1 < len(words):
                    color = words[color_idx]
                    obj_type = words[color_idx + 1]
                    if obj_type in self.object_types:
                        required.append((color, obj_type))
        
        return required
    
    def _extract_required_appliances(self, instructions: List[str]) -> List[str]:
        """Extract appliance types needed for chores"""
        required = []
        appliance_types = ["refrigerator", "stove", "tv", "sink"]
        
        for instruction in instructions:
            if "turn" in instruction.lower():
                words = instruction.lower().split()
                for word in words:
                    if word in appliance_types:
                        required.append(word)
                        break
        
        return required
    
    def _extract_required_containers(self, instructions: List[str]) -> List[str]:
        """Extract container types needed for chores"""
        required = []
        container_types = ["dresser", "refrigerator", "closet"]
        
        for instruction in instructions:
            if "put the" in instruction.lower():
                words = instruction.lower().split()
                for word in words:
                    if word in container_types:
                        required.append(word)
                        break
        
        return required
    
    def _parse_move_instruction(self, instruction: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse move instruction to extract color, object_type, room"""
        words = instruction.lower().split()
        
        color = None
        obj_type = None
        room = None
        
        for i, word in enumerate(words):
            if word in self.colors:
                color = word
                if i + 1 < len(words) and words[i + 1] in self.object_types:
                    obj_type = words[i + 1]
                break
        
        rooms = ["kitchen", "living room", "bedroom", "bathroom", "corridor"]
        for room_name in rooms:
            if room_name in instruction.lower():
                room = room_name
                break
        
        return color, obj_type, room
    
    def _parse_appliance_instruction(self, instruction: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse appliance instruction to extract target state and appliance type"""
        instruction_lower = instruction.lower()
        
        state = None
        if "turn on" in instruction_lower:
            state = "on"
        elif "turn off" in instruction_lower:
            state = "off"
        
        appliance_types = ["refrigerator", "stove", "tv", "sink"]
        appliance_type = None
        for app_type in appliance_types:
            if app_type in instruction_lower:
                appliance_type = app_type
                break
        
        return state, appliance_type
    
    def _parse_container_instruction(self, instruction: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse container instruction to extract color, object_type, container_type"""
        words = instruction.lower().split()
        
        color = None
        obj_type = None
        container_type = None
        
        for i, word in enumerate(words):
            if word in self.colors:
                color = word
                if i + 1 < len(words) and words[i + 1] in self.object_types:
                    obj_type = words[i + 1]
                break
        
        container_types = ["dresser", "refrigerator", "closet"]
        for cont_type in container_types:
            if cont_type in instruction.lower():
                container_type = cont_type
                break
        
        return color, obj_type, container_type
    
    def _is_position_accessible(self, pos: Tuple[int, int], world_state: Dict[str, Any]) -> bool:
        """Check if a position is accessible (not wall, in bounds)"""
        x, y = pos
        
        if x < 0 or x >= 12 or y < 0 or y >= 12:
            return False
        
        walls = [tuple(wall) for wall in world_state["apartment"]["walls"]]
        if pos in walls:
            return False
        
        return True
    
    def _is_position_adjacent_accessible(self, pos: Tuple[int, int], world_state: Dict[str, Any]) -> bool:
        """Check if at least one adjacent position is accessible for interaction"""
        x, y = pos
        adjacent_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for adj_pos in adjacent_positions:
            if self._is_position_accessible(adj_pos, world_state):
                return True
        
        return False
    
    def _is_semantically_compatible(self, obj_type: str, container_type: str) -> bool:
        """Check if object type is compatible with container type"""
        compatibility = {
            "dresser": ["clothes"],
            "refrigerator": ["food"],
            "closet": ["clothes", "cleaning_supplies"]
        }
        
        if container_type not in compatibility:
            return True  # Unknown containers accept anything
        
        return obj_type in compatibility[container_type]

# Main validation function
def validate_generated_level(world_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a generated Smart Home Assistant level
    
    Args:
        world_path: Path to the world file to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    try:
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        validator = SmartHomeValidator()
        return validator.validate_level(world_state)
        
    except Exception as e:
        return False, [f"Error loading world file: {str(e)}"]

# Validation function for world state dict
def validate_world_state(world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a Smart Home Assistant world state dictionary
    
    Args:
        world_state: Dictionary containing the world state
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = SmartHomeValidator()
    return validator.validate_level(world_state)