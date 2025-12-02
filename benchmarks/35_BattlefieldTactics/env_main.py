from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import TacticalObservationPolicy
from env_generate import SquadWorldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy
import random

class SquadReconEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = TacticalObservationPolicy(sensor_radius=3)
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        with open("./config.yaml", 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        self.generator = SquadWorldGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode is 'load'")
            self._state = self._load_world(world_id)
        else:
            raise ValueError("mode must be 'load' or 'generate'")
        
        self._t = 0
        self._history = []
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.load(f, Loader=yaml.FullLoader)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed=seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action["action"]
        params = action.get("params", {})
        
        if action_name == "HOLD_POSITION":
            squad_id = params.get("squad_id")
            # Squad stays in position, no state change
            
        elif action_name in ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]:
            squad_id = params.get("squad_id")
            self._move_squad(squad_id, action_name)
            
        elif action_name == "ATTACK_ENEMY_CAMP":
            squad_id = params.get("squad_id")
            camp_id = params.get("camp_id")
            self._attack_enemy_camp(squad_id, camp_id)
        
        # Update visibility based on new squad positions
        self._update_visibility()
        
        return self._state
    
    def _move_squad(self, squad_id: int, direction: str):
        squad = next((s for s in self._state["squads"] if s["id"] == squad_id and s["active"]), None)
        if squad is None:
            return
        
        current_pos = squad["pos"]
        new_pos = current_pos.copy()
        
        if direction == "MOVE_NORTH":
            new_pos[1] -= 1
        elif direction == "MOVE_SOUTH":
            new_pos[1] += 1
        elif direction == "MOVE_EAST":
            new_pos[0] += 1
        elif direction == "MOVE_WEST":
            new_pos[0] -= 1
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            squad["pos"] = new_pos
    
    def _is_valid_position(self, pos: List[int]) -> bool:
        grid_size = self._state["globals"]["grid_size"]
        x, y = pos
        
        # Check grid bounds
        if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
            return False
        
        # Check for walls and forests
        walls = self._state["terrain"]["walls"]
        forests = self._state["terrain"]["forests"]
        
        if pos in walls or pos in forests:
            return False
        
        return True
    
    def _attack_enemy_camp(self, squad_id: int, camp_id: int):
        squad = next((s for s in self._state["squads"] if s["id"] == squad_id and s["active"]), None)
        camp = next((c for c in self._state["enemy_camps"] if c["id"] == camp_id), None)
        
        if squad is None or camp is None:
            return
        
        # Check if squad is adjacent to camp
        if not self._is_adjacent(squad["pos"], camp["pos"]):
            return
        
        # Get all squads adjacent to the camp
        adjacent_squads = self._get_adjacent_squads(camp["pos"])
        total_attack_strength = sum(s["strength"] for s in adjacent_squads)
        
        if total_attack_strength > camp["strength"]:
            # Successful attack - eliminate camp
            self._state["enemy_camps"] = [c for c in self._state["enemy_camps"] if c["id"] != camp_id]
            self._state["globals"]["eliminated_camps"] += 1
            self._last_action_result = "success"
        else:
            # Failed attack - destroy all attacking squads
            attacking_ids = [s["id"] for s in adjacent_squads]
            for squad in self._state["squads"]:
                if squad["id"] in attacking_ids:
                    squad["active"] = False
            self._last_action_result = "failure"
    
    def _is_adjacent(self, pos1: List[int], pos2: List[int]) -> bool:
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _get_adjacent_squads(self, camp_pos: List[int]) -> List[Dict[str, Any]]:
        adjacent = []
        for squad in self._state["squads"]:
            if squad["active"] and self._is_adjacent(squad["pos"], camp_pos):
                adjacent.append(squad)
        return adjacent
    
    def _update_visibility(self):
        # Visibility is handled by observation policy
        pass
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_value = 0.0
        reward_info = {}
        
        # Check if any camps were eliminated this step
        prev_eliminated = 0
        if self._history:
            prev_eliminated = self._history[-1]["globals"]["eliminated_camps"]
        
        current_eliminated = self._state["globals"]["eliminated_camps"]
        
        if current_eliminated > prev_eliminated:
            events.append("enemy_camp_eliminated")
            reward_value = 0.5  # Each camp worth 0.5 points
            reward_info["camps_eliminated"] = current_eliminated - prev_eliminated
        
        return reward_value, events, reward_info
    
    def observe_semantic(self) -> str:
        omega = self.obs_policy(self._state, self._t)
        return self.render_skin(omega)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        
        # Handle case where omega is already a string (from base class)
        if isinstance(omega, str):
            return omega
        grid_size = self._state["globals"]["grid_size"]
        tactical_map = omega["tactical_map"]
        ascii_map = [['?' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        
        # Fill explored areas
        walls = self._state["terrain"]["walls"]
        forests = self._state["terrain"]["forests"]
        
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if tactical_map[x][y]:  # Explored
                    if [x, y] in walls:
                        ascii_map[x][y] = '#'
                    elif [x, y] in forests:
                        ascii_map[x][y] = 'T'
                    else:
                        ascii_map[x][y] = '.'
        
        # Add squads
        for squad in omega["squads"]:
            pos = squad["pos"]
            ascii_map[pos[0]][pos[1]] = f'S{squad["id"]}'
        
        # Add discovered enemy camps
        for camp in omega["discovered_enemy_camps"]:
            pos = camp["pos"]
            ascii_map[pos[0]][pos[1]] = f'E{camp["id"]}'
        
        # Convert to string
        map_str = '\n'.join(''.join(f'{cell:>3}' for cell in row) for row in ascii_map)
        
        # Create squad status table
        squad_status = "ID | Position | Strength | Status\n"
        squad_status += "-" * 30 + "\n"
        for squad in omega["squads"]:
            pos_str = f"({squad['pos'][0]}, {squad['pos'][1]})"
            status = "Active" if squad["active"] else "Destroyed"
            squad_status += f"{squad['id']:2} | {pos_str:8} | {squad['strength']:8} | {status}\n"
        
        # Format final output
        output = f"""=== TACTICAL SITUATION REPORT ===
Step: {omega['t']}/{omega['max_steps']}
Mission Progress: {omega['eliminated_camps']}/{omega['total_enemy_camps']} enemy camps eliminated

SQUAD STATUS:
{squad_status}

TACTICAL MAP (15x15):
{map_str}

LEGEND: S=Squad, E=Enemy Camp, #=Wall, T=Forest, .=Open, ?=Unexplored

Total Friendly Strength: {omega['total_friendly_strength']}

Available Actions: HOLD_POSITION(squad_id), MOVE_NORTH/SOUTH/EAST/WEST(squad_id), ATTACK_ENEMY_CAMP(squad_id, camp_id)"""
        
        return output
    
    def done(self, state=None) -> bool:
        state = state if state is not None else self._state
        
        # Victory condition
        if state["globals"]["eliminated_camps"] >= state["globals"]["total_enemy_camps"]:
            return True
        
        # Time limit (check both loaded world config and environment config)
        max_steps = state["globals"]["max_steps"]
        if self._t >= max_steps:
            return True
        
        # All squads destroyed
        active_squads = [s for s in state["squads"] if s["active"]]
        if len(active_squads) == 0:
            return True
        
        return False