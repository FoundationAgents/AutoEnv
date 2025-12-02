import sys
sys.path.append('../../../')
import yaml
import os
import random
import uuid
from typing import Dict, Any, Optional
from base.env.base_generator import WorldGenerator

class SmartHomeGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        self.object_types = ["food", "book", "clothes", "cleaning_supplies", "electronics"]
        self.colors = ["red", "blue", "green", "yellow", "white", "black"]
        self.chore_templates = [
            "Move the {color} {object} to the {room}",
            "Turn {state} the {appliance} in the {room}",
            "Put the {color} {object} in the {container}"
        ]
    
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        base_state = {
            "globals": {"max_steps": 60, "grid_size": [12, 12], "num_chores": 3},
            "agent": {"pos": [0, 0], "facing": "north", "inventory": None},
            "apartment": {
                "walls": [],
                "doors": [],
                "rooms": {
                    "kitchen": {"bounds": [], "furniture": [], "appliances": []},
                    "living_room": {"bounds": [], "furniture": [], "appliances": []},
                    "bedroom": {"bounds": [], "furniture": [], "appliances": []},
                    "bathroom": {"bounds": [], "furniture": [], "appliances": []},
                    "corridor": {"bounds": [], "furniture": [], "appliances": []}
                }
            },
            "objects": [],
            "appliances": [],
            "containers": [],
            "chores": {"instructions": [], "completed": [False, False, False]},
            "current_room": "corridor"
        }
        
        world_state = self._execute_pipeline(base_state, seed)
        world_id = self._generate_world_id(seed)
        
        os.makedirs("./levels", exist_ok=True)
        save_path = f"./levels/world_{world_id}.yaml"
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = base_state.copy()
        
        state = self._generate_apartment_layout(state)
        state = self._place_furniture_appliances(state)
        state = self._populate_objects(state)
        state = self._generate_chore_instructions(state)
        state = self._place_agent(state)
        
        return state
    
    def _generate_apartment_layout(self, state):
        rooms = {
            "kitchen": [0, 0, 5, 4],
            "living_room": [6, 0, 11, 4], 
            "bedroom": [0, 5, 5, 8],
            "bathroom": [6, 5, 11, 8],  # Extended to cover the gap
            "corridor": [0, 9, 11, 11]
        }
        
        walls = []
        doors = []
        
        for room_name, bounds in rooms.items():
            x1, y1, x2, y2 = bounds
            state["apartment"]["rooms"][room_name]["bounds"] = bounds
            
            # Only add exterior walls, not internal room boundaries
            if room_name == "kitchen":
                # Kitchen: left, top walls only
                for x in range(x1, x2 + 1):
                    walls.append([x, y1])  # top wall
                for y in range(y1, y2 + 1):
                    walls.append([x1, y])  # left wall
            elif room_name == "living_room":
                # Living room: right, top walls only  
                for x in range(x1, x2 + 1):
                    walls.append([x, y1])  # top wall
                for y in range(y1, y2 + 1):
                    walls.append([x2, y])  # right wall
            elif room_name == "bedroom":
                # Bedroom: left wall only
                for y in range(y1, y2 + 1):
                    walls.append([x1, y])  # left wall
            elif room_name == "bathroom":
                # Bathroom: right wall only
                for y in range(y1, y2 + 1):
                    walls.append([x2, y])  # right wall
            elif room_name == "corridor":
                # Corridor: left, right, bottom walls
                for y in range(y1, y2 + 1):
                    walls.extend([[x1, y], [x2, y]])  # left and right walls
                for x in range(x1, x2 + 1):
                    walls.append([x, y2])  # bottom wall
        
        doors.extend([[3, 4], [8, 4], [3, 8], [7, 8], [5, 9]])
        
        for door in doors:
            if door in walls:
                walls.remove(door)
        
        state["apartment"]["walls"] = walls
        state["apartment"]["doors"] = doors
        return state
    
    def _place_furniture_appliances(self, state):
        room_items = {
            "kitchen": {
                "appliances": [
                    {"type": "refrigerator", "pos": [1, 1], "state": "off"},
                    {"type": "stove", "pos": [2, 1], "state": "off"}
                ],
                "furniture": [
                    {"type": "counter", "pos": [3, 1]},
                    {"type": "sink", "pos": [4, 1]}
                ],
                "containers": [
                    {"type": "refrigerator", "pos": [1, 1], "open": False, "contents": []}
                ]
            },
            "living_room": {
                "appliances": [
                    {"type": "tv", "pos": [9, 1], "state": "off"}
                ],
                "furniture": [
                    {"type": "sofa", "pos": [7, 2]},
                    {"type": "table", "pos": [8, 2]}
                ]
            },
            "bedroom": {
                "furniture": [
                    {"type": "bed", "pos": [1, 6]},
                    {"type": "dresser", "pos": [4, 6]}
                ],
                "containers": [
                    {"type": "dresser", "pos": [4, 6], "open": False, "contents": []},
                    {"type": "closet", "pos": [4, 7], "open": False, "contents": []}
                ]
            },
            "bathroom": {
                "furniture": [
                    {"type": "toilet", "pos": [7, 6]},
                    {"type": "sink", "pos": [6, 7]}
                ]
            }
        }
        
        for room_name, items in room_items.items():
            if "appliances" in items:
                state["apartment"]["rooms"][room_name]["appliances"] = items["appliances"]
                state["appliances"].extend(items["appliances"])
            
            if "furniture" in items:
                state["apartment"]["rooms"][room_name]["furniture"] = items["furniture"]
            
            if "containers" in items:
                state["containers"].extend(items["containers"])
        
        return state
    
    def _populate_objects(self, state):
        floor_positions = []
        for x in range(12):
            for y in range(12):
                if ([x, y] not in state["apartment"]["walls"] and 
                    not self._is_occupied([x, y], state)):
                    floor_positions.append([x, y])
        
        for i in range(8):
            if not floor_positions:
                break
            
            pos = random.choice(floor_positions)
            floor_positions.remove(pos)
            
            obj_type = random.choice(self.object_types)
            color = random.choice(self.colors)
            
            obj = {
                "type": obj_type,
                "color": color,
                "pos": pos,
                "id": f"obj_{i}"
            }
            state["objects"].append(obj)
        
        return state
    
    def _generate_chore_instructions(self, state):
        instructions = []
        
        for i in range(3):
            template_idx = i % len(self.chore_templates)
            template = self.chore_templates[template_idx]
            
            if "Move the" in template:
                obj = random.choice(state["objects"])
                room = random.choice(["kitchen", "living room", "bedroom", "bathroom"])
                instruction = template.format(
                    color=obj["color"],
                    object=obj["type"],
                    room=room
                )
            elif "Turn" in template:
                appliance = random.choice(state["appliances"])
                room_name = self._get_room_for_appliance(appliance["pos"], state)
                state_word = random.choice(["on", "off"])
                instruction = template.format(
                    state=state_word,
                    appliance=appliance["type"],
                    room=room_name.replace("_", " ")
                )
            elif "Put the" in template:
                # Ensure compatible object-container pairs
                compatible_pairs = []
                for obj in state["objects"]:
                    for container in state["containers"]:
                        if self._is_compatible_pair(obj, container):
                            compatible_pairs.append((obj, container))
                
                if compatible_pairs:
                    obj, container = random.choice(compatible_pairs)
                    instruction = template.format(
                        color=obj["color"],
                        object=obj["type"],
                        container=container["type"]
                    )
                else:
                    # Fallback to move instruction if no compatible pairs
                    obj = random.choice(state["objects"])
                    room = random.choice(["kitchen", "living room", "bedroom", "bathroom"])
                    instruction = "Move the {color} {object} to the {room}".format(
                        color=obj["color"],
                        object=obj["type"],
                        room=room
                    )
            
            instructions.append(instruction)
        
        state["chores"]["instructions"] = instructions
        return state
        
    def _is_compatible_pair(self, obj, container):
        """Check if object can be placed in container"""
        if container["type"] == "dresser":
            return obj["type"] == "clothes"
        elif container["type"] == "refrigerator":
            return obj["type"] == "food"
        elif container["type"] == "closet":
            return obj["type"] == "clothes"
        return True  # Default: any object fits in generic containers
    
    def _place_agent(self, state):
        floor_positions = []
        for x in range(12):
            for y in range(12):
                if ([x, y] not in state["apartment"]["walls"] and 
                    not self._is_occupied([x, y], state)):
                    floor_positions.append([x, y])
        
        if floor_positions:
            agent_pos = random.choice(floor_positions)
            state["agent"]["pos"] = agent_pos
        
        state["agent"]["facing"] = random.choice(["north", "south", "east", "west"])
        return state
    
    def _is_occupied(self, pos, state):
        for obj in state["objects"]:
            if obj["pos"] == pos:
                return True
        for appliance in state["appliances"]:
            if appliance["pos"] == pos:
                return True
        for container in state["containers"]:
            if container["pos"] == pos:
                return True
        for room_data in state["apartment"]["rooms"].values():
            for furniture in room_data.get("furniture", []):
                if furniture["pos"] == pos:
                    return True
        return False
    
    def _get_room_for_appliance(self, pos, state):
        for room_name, room_data in state["apartment"]["rooms"].items():
            bounds = room_data["bounds"]
            if (bounds[0] <= pos[0] <= bounds[2] and 
                bounds[1] <= pos[1] <= bounds[3]):
                return room_name
        return "corridor"
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            return f"seed_{seed}_{uuid.uuid4().hex[:8]}"
        else:
            return uuid.uuid4().hex[:12]