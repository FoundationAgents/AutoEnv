import sys
import os
sys.path.append('../../../')

from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from base.env.base_observation import ObservationPolicy
from base.env.base_generator import WorldGenerator
from env_obs import FullGridObservation
from env_generate import InvertedBoxEscapeGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class InvertedBoxEscapeEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = FullGridObservation()
        super().__init__(env_id, obs_policy)
        self.generator = None

    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        
        self.generator = InvertedBoxEscapeGenerator(
            env_id=str(self.env_id),
            config=self.configs.get('generator', {})
        )

    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        
        if world_id is None:
            # Generate a default world
            world_id = self._generate_world(seed)
        
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self.observe_semantic()

    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.load(f, Loader=yaml.FullLoader)
        return world_state

    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed=seed)

    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_name = action.get('action', '')
        
        # Store previous state
        prev_state = self._state.copy()
        self._history.append(prev_state)
        
        agent_pos = self._state['agent']['pos']
        H, W = self._state['grid']['size']
        
        # Calculate new position based on action
        new_pos = agent_pos[:]
        if action_name == "MOVE_NORTH":
            new_pos[0] = max(0, agent_pos[0] - 1)
        elif action_name == "MOVE_SOUTH":
            new_pos[0] = min(H - 1, agent_pos[0] + 1)
        elif action_name == "MOVE_EAST":
            new_pos[1] = min(W - 1, agent_pos[1] + 1)
        elif action_name == "MOVE_WEST":
            new_pos[1] = max(0, agent_pos[1] - 1)
        elif action_name == "WAIT":
            # Do nothing, just consume time
            return self._state
        else:
            # Invalid action, do nothing
            return self._state
        
        # Check if move is blocked by boundary
        if new_pos == agent_pos and action_name != "WAIT":
            # Hit boundary, no movement
            return self._state
        
        # Check if target cell has wall
        layout = self._state['grid']['layout']
        if new_pos[0] < len(layout) and new_pos[1] < len(layout[0]):
            if layout[new_pos[0]][new_pos[1]] == 'E':
                # Hit wall, no movement
                return self._state
        
        # Check if target cell has crate - attempt push
        crate_positions = self._state['objects']['crates']
        crate_at_target = None
        for i, crate_pos in enumerate(crate_positions):
            if crate_pos == new_pos:
                crate_at_target = i
                break
        
        if crate_at_target is not None:
            # Calculate crate's new position
            crate_new_pos = new_pos[:]
            if action_name == "MOVE_NORTH":
                crate_new_pos[0] = max(0, new_pos[0] - 1)
            elif action_name == "MOVE_SOUTH":
                crate_new_pos[0] = min(H - 1, new_pos[0] + 1)
            elif action_name == "MOVE_EAST":
                crate_new_pos[1] = min(W - 1, new_pos[1] + 1)
            elif action_name == "MOVE_WEST":
                crate_new_pos[1] = max(0, new_pos[1] - 1)
            
            # Check if crate push is valid
            push_blocked = False
            
            # Check boundary
            if crate_new_pos == new_pos:
                push_blocked = True
            
            # Check wall
            if not push_blocked and crate_new_pos[0] < len(layout) and crate_new_pos[1] < len(layout[0]):
                if layout[crate_new_pos[0]][crate_new_pos[1]] == 'E':
                    push_blocked = True
            
            # Check other crates
            if not push_blocked:
                for other_crate in crate_positions:
                    if other_crate == crate_new_pos:
                        push_blocked = True
                        break
            
            if push_blocked:
                # Can't push crate, no movement
                return self._state
            else:
                # Push crate and move agent
                self._state['objects']['crates'][crate_at_target] = crate_new_pos
                self._state['agent']['pos'] = new_pos
                
                # Check if crate landed on storage tile
                for storage_pos in self._state['objects']['storage_tiles']:
                    if storage_pos == crate_new_pos:
                        # Cover this storage tile
                        if storage_pos not in self._state['objects']['covered_tiles']:
                            self._state['objects']['covered_tiles'].append(storage_pos)
                        break
                
                return self._state
        else:
            # No crate at target, just move agent if safe
            self._state['agent']['pos'] = new_pos
            return self._state

    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward = 0.0
        
        # Check if all storage tiles are covered
        all_covered = len(self._state['objects']['covered_tiles']) == len(self._state['objects']['storage_tiles'])
        
        # Check if agent is on exit
        agent_pos = self._state['agent']['pos']
        exit_pos = self._state['objects']['exit_pos']
        on_exit = agent_pos == exit_pos
        
        if all_covered and on_exit:
            reward = 1.0
            events.append("escape_success")
        
        reward_info = {
            "all_storage_covered": all_covered,
            "agent_on_exit": on_exit,
            "success": reward > 0
        }
        
        return reward, events, reward_info

    def done(self, state=None) -> bool:
        # Check max steps (from level override or config)
        max_steps = self._state.get('globals', {}).get('max_steps', self.configs["termination"]["max_steps"])
        if self._t >= max_steps:
            return True
        
        # Check hazard contact
        agent_pos = self._state['agent']['pos']
        
        # Check if agent touched crate (lethal)
        for crate_pos in self._state['objects']['crates']:
            if agent_pos == crate_pos:
                return True
        
        # Check if agent touched uncovered storage tile (dangerous)
        covered_tiles = set(tuple(pos) for pos in self._state['objects']['covered_tiles'])
        for storage_pos in self._state['objects']['storage_tiles']:
            if agent_pos == storage_pos and tuple(storage_pos) not in covered_tiles:
                return True
        
        # Check successful escape
        all_covered = len(self._state['objects']['covered_tiles']) == len(self._state['objects']['storage_tiles'])
        on_exit = agent_pos == self._state['objects']['exit_pos']
        if all_covered and on_exit:
            return True
        
        return False

    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)

    def render_skin(self, omega: Dict[str, Any]) -> Any:
        grid = omega['grid']
        H, W = len(grid), len(grid[0])
        
        # Create grid display string
        grid_lines = []
        for row in grid:
            grid_lines.append(' '.join(row))
        grid_display = '\n'.join(grid_lines)
        
        # Get current step info
        max_steps = omega['max_steps']
        current_step = omega['step_count']
        agent_pos = omega['agent_pos']
        covered_count = omega['covered_count']
        total_storage = omega['total_storage']
        
        # Format output
        output = f"Step {current_step}/{max_steps}\n"
        output += f"Agent at ({agent_pos[0]}, {agent_pos[1]})\n\n"
        output += "Grid Layout:\n"
        output += grid_display + "\n\n"
        output += "Legend: P=Agent, B=Crate(lethal), C=Storage(dangerous), D=Exit, E=Wall, A=Safe floor\n"
        output += f"Covered storage tiles: {covered_count}/{total_storage}\n\n"
        output += "Actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST, WAIT"
        
        return output
