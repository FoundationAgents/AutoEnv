import sys
sys.path.append('../../../')
import yaml
import os
import random
from typing import Dict, Any, Optional, Tuple, List
from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import LocalGridObservation
from env_generate import SmartHomeGenerator

class SmartHomeEnv(SkinEnv):
    def __init__(self, env_id: int = 85):
        obs_policy = LocalGridObservation()
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = SmartHomeGenerator("smart_home_assistant", self.configs["generator"])
        world_id = generator.generate(seed)
        return world_id
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        action_name = action.get("action")
        params = action.get("params", {})
        
        if action_name == "MoveForward":
            self._move_forward()
        elif action_name == "TurnLeft":
            self._turn_left()
        elif action_name == "TurnRight":
            self._turn_right()
        elif action_name == "PickUp":
            self._pickup()
        elif action_name == "Drop":
            self._drop()
        elif action_name == "ToggleAppliance":
            self._toggle_appliance()
        elif action_name == "OpenCloseContainer":
            self._open_close_container()
        elif action_name == "Wait":
            pass
        
        self._update_current_room()
        return self._state
    
    def _move_forward(self):
        facing = self._state["agent"]["facing"]
        pos = self._state["agent"]["pos"]
        new_pos = pos.copy()
        
        if facing == "north":
            new_pos[1] -= 1
        elif facing == "south":
            new_pos[1] += 1
        elif facing == "east":
            new_pos[0] += 1
        elif facing == "west":
            new_pos[0] -= 1
        
        if self._is_valid_position(new_pos):
            self._state["agent"]["pos"] = new_pos
    
    def _turn_left(self):
        facing_order = ["north", "west", "south", "east"]
        current_idx = facing_order.index(self._state["agent"]["facing"])
        self._state["agent"]["facing"] = facing_order[(current_idx + 1) % 4]
    
    def _turn_right(self):
        facing_order = ["north", "east", "south", "west"]
        current_idx = facing_order.index(self._state["agent"]["facing"])
        self._state["agent"]["facing"] = facing_order[(current_idx + 1) % 4]
    
    def _pickup(self):
        if self._state["agent"]["inventory"] is not None:
            return
        
        adjacent_pos = self._get_front_position()
        for obj in self._state["objects"]:
            if obj["pos"] == adjacent_pos:
                self._state["agent"]["inventory"] = obj
                self._state["objects"].remove(obj)
                break
    
    def _drop(self):
        if self._state["agent"]["inventory"] is None:
            return
        
        front_pos = self._get_front_position()
        inventory_item = self._state["agent"]["inventory"]
        
        for container in self._state["containers"]:
            if container["pos"] == front_pos and container["open"]:
                if self._is_compatible_container(inventory_item, container):
                    container["contents"].append(inventory_item)
                    self._state["agent"]["inventory"] = None
                    return
        
        if self._is_valid_position(front_pos) and not self._is_occupied(front_pos):
            inventory_item["pos"] = front_pos
            self._state["objects"].append(inventory_item)
            self._state["agent"]["inventory"] = None
    
    def _toggle_appliance(self):
        adjacent_positions = self._get_adjacent_positions()
        for appliance in self._state["appliances"]:
            if appliance["pos"] in adjacent_positions:
                appliance["state"] = "off" if appliance["state"] == "on" else "on"
                break
    
    def _open_close_container(self):
        front_pos = self._get_front_position()
        for container in self._state["containers"]:
            if container["pos"] == front_pos:
                container["open"] = not container["open"]
                break
    
    def _get_front_position(self):
        pos = self._state["agent"]["pos"]
        facing = self._state["agent"]["facing"]
        
        if facing == "north":
            return [pos[0], pos[1] - 1]
        elif facing == "south":
            return [pos[0], pos[1] + 1]
        elif facing == "east":
            return [pos[0] + 1, pos[1]]
        elif facing == "west":
            return [pos[0] - 1, pos[1]]
    
    def _get_adjacent_positions(self):
        pos = self._state["agent"]["pos"]
        return [
            [pos[0], pos[1] - 1],
            [pos[0], pos[1] + 1],
            [pos[0] + 1, pos[1]],
            [pos[0] - 1, pos[1]]
        ]
    
    def _is_valid_position(self, pos):
        if pos[0] < 0 or pos[0] >= 12 or pos[1] < 0 or pos[1] >= 12:
            return False
        return pos not in self._state["apartment"]["walls"]
    
    def _is_occupied(self, pos):
        for obj in self._state["objects"]:
            if obj["pos"] == pos:
                return True
        for appliance in self._state["appliances"]:
            if appliance["pos"] == pos:
                return True
        for container in self._state["containers"]:
            if container["pos"] == pos:
                return True
        for room_data in self._state["apartment"]["rooms"].values():
            for furniture in room_data["furniture"]:
                if furniture["pos"] == pos:
                    return True
        return False
    
    def _is_compatible_container(self, obj, container):
        if container["type"] == "dresser":
            return obj["type"] == "clothes"
        elif container["type"] == "refrigerator":
            return obj["type"] == "food"
        return True
    
    def _update_current_room(self):
        agent_pos = self._state["agent"]["pos"]
        for room_name, room_data in self._state["apartment"]["rooms"].items():
            bounds = room_data["bounds"]
            if (bounds[0] <= agent_pos[0] <= bounds[2] and 
                bounds[1] <= agent_pos[1] <= bounds[3]):
                self._state["current_room"] = room_name
                return
        self._state["current_room"] = "corridor"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        total_reward = 0.0
        events = []
        reward_info = {}
        
        if self._pickup_target_object():
            total_reward += 0.3
            events.append("pickup_target_object")
            reward_info["pickup_bonus"] = 0.3
        
        completed_chores = self._check_completed_chores()
        for i, completed in enumerate(completed_chores):
            if completed and not self._state["chores"]["completed"][i]:
                total_reward += 0.7
                events.append("complete_chore")
                reward_info[f"chore_{i}_completion"] = 0.7
                self._state["chores"]["completed"][i] = True
                
                if all(self._state["chores"]["completed"]):
                    total_reward += 1.0
                    events.append("complete_final_chore")
                    reward_info["final_chore_bonus"] = 1.0
        
        return total_reward, events, reward_info
    
    def _pickup_target_object(self):
        if len(self._history) == 0:
            return False
            
        prev_inventory = self._history[-1]["agent"]["inventory"]
        curr_inventory = self._state["agent"]["inventory"]
        
        if prev_inventory is None and curr_inventory is not None:
            for i, instruction in enumerate(self._state["chores"]["instructions"]):
                if not self._state["chores"]["completed"][i]:
                    if self._object_needed_for_instruction(curr_inventory, instruction):
                        return True
        return False
    
    def _object_needed_for_instruction(self, obj, instruction):
        return obj["type"] in instruction.lower() and obj["color"] in instruction.lower()
    
    def _check_completed_chores(self):
        completed = []
        for instruction in self._state["chores"]["instructions"]:
            if "move" in instruction.lower():
                completed.append(self._check_move_chore(instruction))
            elif "turn" in instruction.lower():
                completed.append(self._check_appliance_chore(instruction))
            elif "put" in instruction.lower():
                completed.append(self._check_container_chore(instruction))
            else:
                completed.append(False)
        return completed
    
    def _check_move_chore(self, instruction):
        words = instruction.lower().split()
        for i, obj in enumerate(self._state["objects"]):
            if obj["color"] in instruction.lower() and obj["type"] in instruction.lower():
                target_room = None
                for room in ["kitchen", "living room", "bedroom", "bathroom", "corridor"]:
                    if room in instruction.lower():
                        target_room = room.replace(" ", "_")
                        break
                
                if target_room:
                    room_bounds = self._state["apartment"]["rooms"][target_room]["bounds"]
                    return (room_bounds[0] <= obj["pos"][0] <= room_bounds[2] and 
                           room_bounds[1] <= obj["pos"][1] <= room_bounds[3])
        return False
    
    def _check_appliance_chore(self, instruction):
        target_state = "on" if "on" in instruction.lower() else "off"
        for appliance in self._state["appliances"]:
            if appliance["type"] in instruction.lower():
                return appliance["state"] == target_state
        return False
    
    def _check_container_chore(self, instruction):
        for container in self._state["containers"]:
            if container["type"] in instruction.lower():
                for item in container["contents"]:
                    if item["type"] in instruction.lower() and item["color"] in instruction.lower():
                        return True
        return False
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        max_steps = self._state.get("globals", {}).get("max_steps", 40)
        current_room = omega.get("current_room", "unknown")
        facing = omega.get("agent", {}).get("facing", "north")
        inventory = omega.get("agent", {}).get("inventory")
        
        inventory_str = f"{inventory['color']} {inventory['type']}" if inventory else "empty"
        
        chores_display = ""
        instructions = omega.get("chores", {}).get("instructions", [])
        completed = omega.get("chores", {}).get("completed", [])
        
        for i, (instruction, done) in enumerate(zip(instructions, completed)):
            status = "✓" if done else "○"
            chores_display += f"{status} {instruction}\n"
        
        vision_grid = self._format_vision_grid(omega.get("visible_grid", []))
        
        visible_appliances = []
        for app in omega.get("visible_appliances", []):
            visible_appliances.append(f"{app['type']}({app['state']})")
        appliances_str = ", ".join(visible_appliances) if visible_appliances else "none"
        
        return f"""Step {omega['t']}/{max_steps} | Room: {current_room} | Facing: {facing}
Inventory: {inventory_str}

Chores:
{chores_display}
Vision Grid (5x5, A=Agent):
{vision_grid}

Visible Appliances: {appliances_str}
Available Actions: MoveForward, TurnLeft, TurnRight, PickUp, Drop, ToggleAppliance, OpenCloseContainer, Wait"""
    
    def _format_vision_grid(self, grid):
        if not grid or len(grid) != 5:
            return "Grid unavailable"
        
        result = ""
        for row in grid:
            if len(row) != 5:
                result += "Invalid row\n"
                continue
            row_str = ""
            for cell in row:
                if cell == "agent":
                    row_str += "A "
                elif cell == "wall":
                    row_str += "# "
                elif cell == "floor":
                    row_str += ". "
                elif cell == "door":
                    row_str += "D "
                elif cell == "unknown":
                    row_str += "? "
                elif isinstance(cell, dict):
                    if cell.get("type") == "object":
                        row_str += cell.get("name", "O")[0].upper() + " "
                    else:
                        row_str += "X "
                else:
                    row_str += str(cell)[0] + " "
            result += row_str.strip() + "\n"
        return result.strip()
    
    def done(self, state=None) -> bool:
        max_steps = self._state.get("globals", {}).get("max_steps", self.configs["termination"]["max_steps"])
        return all(self._state["chores"]["completed"]) or self._t >= max_steps
