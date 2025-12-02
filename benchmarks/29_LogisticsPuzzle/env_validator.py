from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import copy

class InvertedBoxEscapeValidator:
    def __init__(self):
        self.max_steps = 40
    
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a generated level for solvability and proper reward structure.
        Returns (is_valid, issues_list)
        """
        issues = []
        
        # 1. Basic structure validation
        structure_issues = self._validate_structure(world_state)
        issues.extend(structure_issues)
        
        # 2. Solvability analysis
        solvability_issues = self._validate_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 3. Reward structure validation
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        return len(issues) == 0, issues
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic level structure and constraints"""
        issues = []
        
        # Check required components exist
        if 'grid' not in world_state or 'size' not in world_state['grid']:
            issues.append("Missing grid size information")
            return issues
        
        H, W = world_state['grid']['size']
        layout = world_state['grid']['layout']
        
        # Validate grid dimensions
        if H < 6 or H > 10 or W < 6 or W > 10:
            issues.append(f"Invalid grid dimensions {H}x{W}, must be between 6x6 and 10x10")
        
        if len(layout) != H:
            issues.append(f"Layout height {len(layout)} doesn't match declared height {H}")
        
        for i, row in enumerate(layout):
            if len(row) != W:
                issues.append(f"Layout row {i} width {len(row)} doesn't match declared width {W}")
        
        # Check object counts
        crates = world_state['objects']['crates']
        storage_tiles = world_state['objects']['storage_tiles']
        
        if len(crates) != len(storage_tiles):
            issues.append(f"Crate count {len(crates)} doesn't match storage tile count {len(storage_tiles)}")
        
        if not (3 <= len(crates) <= 5):
            issues.append(f"Invalid crate count {len(crates)}, must be between 3 and 5")
        
        # Check positions are within bounds
        for i, pos in enumerate(crates):
            if not (0 <= pos[0] < H and 0 <= pos[1] < W):
                issues.append(f"Crate {i} position {pos} is out of bounds")
        
        for i, pos in enumerate(storage_tiles):
            if not (0 <= pos[0] < H and 0 <= pos[1] < W):
                issues.append(f"Storage tile {i} position {pos} is out of bounds")
        
        agent_pos = world_state['agent']['pos']
        if not (0 <= agent_pos[0] < H and 0 <= agent_pos[1] < W):
            issues.append(f"Agent position {agent_pos} is out of bounds")
        
        exit_pos = world_state['objects']['exit_pos']
        if not (0 <= exit_pos[0] < H and 0 <= exit_pos[1] < W):
            issues.append(f"Exit position {exit_pos} is out of bounds")
        
        # Check for position conflicts
        all_positions = [tuple(pos) for pos in crates + storage_tiles + [agent_pos, exit_pos]]
        unique_positions = set(all_positions)
        if len(all_positions) != len(unique_positions):
            issues.append("Multiple objects occupy the same position")
        
        return issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Critical solvability analysis using BFS pathfinding"""
        issues = []
        
        try:
            # 1. ACTION CONSTRAINT ANALYSIS
            H, W = world_state['grid']['size']
            layout = world_state['grid']['layout']
            
            # Check if crates can be pushed to storage tiles
            crates = [tuple(pos) for pos in world_state['objects']['crates']]
            storage_tiles = [tuple(pos) for pos in world_state['objects']['storage_tiles']]
            agent_start = tuple(world_state['agent']['pos'])
            exit_pos = tuple(world_state['objects']['exit_pos'])
            
            # 2. TARGET REACHABILITY - Use BFS to find solution
            solution_exists, min_steps = self._find_solution_bfs(
                H, W, layout, agent_start, crates, storage_tiles, exit_pos
            )
            
            if not solution_exists:
                issues.append("No valid solution exists - level is unsolvable")
                return issues
            
            # 3. STEP COUNTING - Check if solution fits within step limit
            if min_steps > self.max_steps:
                issues.append(f"Minimum solution requires {min_steps} steps, exceeds limit of {self.max_steps}")
            
            # 4. COMMON IMPOSSIBLE PATTERNS CHECK
            
            # Check if any crate is permanently blocked
            for crate_pos in crates:
                if self._is_crate_permanently_blocked(crate_pos, crates, layout, H, W):
                    issues.append(f"Crate at {crate_pos} is permanently blocked and cannot be moved")
            
            # Check if agent can reach necessary positions
            reachable_positions = self._get_reachable_positions(agent_start, crates, layout, H, W)
            
            # Agent must be able to reach positions to push crates effectively
            crate_push_positions = set()
            for crate_pos in crates:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    push_from = (crate_pos[0] - dr, crate_pos[1] - dc)
                    if (0 <= push_from[0] < H and 0 <= push_from[1] < W and 
                        push_from not in crates and layout[push_from[0]][push_from[1]] != 'E'):
                        crate_push_positions.add(push_from)
            
            unreachable_pushes = crate_push_positions - reachable_positions
            if unreachable_pushes and min_steps == float('inf'):
                issues.append("Agent cannot reach necessary positions to push crates")
            
            # Check if exit is reachable after all storage tiles are covered
            if exit_pos not in reachable_positions:
                # This is a simplification - in reality need to check reachability after crates are moved
                issues.append("Exit position may not be reachable")
        
        except Exception as e:
            issues.append(f"Error during solvability analysis: {str(e)}")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate reward structure prevents exploitation"""
        issues = []
        
        # 1. GOAL-ORIENTED REWARDS CHECK
        # The environment uses binary reward (0 or 1), which is good
        # Highest reward only comes from complete objective achievement
        
        # 2. AVOID INCENTIVE MISALIGNMENT
        # Check that the reward structure doesn't allow "action grinding"
        # Binary reward prevents this as no intermediate rewards exist
        
        # 3. VALIDATE REWARD DISTRIBUTION
        # Ensure success condition is clearly defined and achievable
        storage_count = len(world_state['objects']['storage_tiles'])
        crate_count = len(world_state['objects']['crates'])
        
        if storage_count == 0:
            issues.append("No storage tiles to cover - trivial success condition")
        
        if crate_count == 0:
            issues.append("No crates available - impossible to cover storage tiles")
        
        # 4. EFFICIENCY INCENTIVE CHECK
        # Binary reward naturally encourages efficiency (solve or fail)
        # Step limit provides time pressure
        max_steps = world_state.get('globals', {}).get('max_steps', self.max_steps)
        if max_steps > 100:
            issues.append(f"Step limit {max_steps} too high, may encourage inefficient exploration")
        elif max_steps < 10:
            issues.append(f"Step limit {max_steps} too low, may be impossible to solve")
        
        return issues
    
    def _find_solution_bfs(self, H: int, W: int, layout: List[List[str]], 
                          agent_start: Tuple[int, int], crates: List[Tuple[int, int]], 
                          storage_tiles: List[Tuple[int, int]], exit_pos: Tuple[int, int]) -> Tuple[bool, int]:
        """Use BFS to find if a solution exists and minimum steps required"""
        
        # State: (agent_pos, tuple_of_crate_positions, covered_storage_mask)
        initial_crates = tuple(sorted(crates))
        initial_covered = tuple([False] * len(storage_tiles))
        initial_state = (agent_start, initial_crates, initial_covered)
        
        queue = deque([(initial_state, 0)])  # (state, steps)
        visited = {initial_state}
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # North, South, West, East
        
        while queue:
            (agent_pos, crate_positions, covered_mask), steps = queue.popleft()
            
            # Check if solved
            if all(covered_mask) and agent_pos == exit_pos:
                return True, steps
            
            # Prevent excessive search
            if steps >= self.max_steps or len(visited) > 10000:
                continue
            
            # Try each move
            for dr, dc in moves:
                new_agent_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
                
                # Boundary check
                if not (0 <= new_agent_pos[0] < H and 0 <= new_agent_pos[1] < W):
                    continue
                
                # Wall check
                if layout[new_agent_pos[0]][new_agent_pos[1]] == 'E':
                    continue
                
                new_crate_positions = list(crate_positions)
                new_covered = list(covered_mask)
                
                # Check if moving into crate (push attempt)
                crate_index = -1
                for i, crate_pos in enumerate(crate_positions):
                    if crate_pos == new_agent_pos:
                        crate_index = i
                        break
                
                if crate_index >= 0:
                    # Attempt to push crate
                    crate_new_pos = (new_agent_pos[0] + dr, new_agent_pos[1] + dc)
                    
                    # Check if crate push is valid
                    if not (0 <= crate_new_pos[0] < H and 0 <= crate_new_pos[1] < W):
                        continue
                    if layout[crate_new_pos[0]][crate_new_pos[1]] == 'E':
                        continue
                    if crate_new_pos in crate_positions:
                        continue
                    
                    # Valid push
                    new_crate_positions[crate_index] = crate_new_pos
                    
                    # Check if crate lands on storage
                    for j, storage_pos in enumerate(storage_tiles):
                        if crate_new_pos == storage_pos:
                            new_covered[j] = True
                            break
                
                # Create new state
                new_state = (new_agent_pos, tuple(sorted(new_crate_positions)), tuple(new_covered))
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
        
        return False, float('inf')
    
    def _is_crate_permanently_blocked(self, crate_pos: Tuple[int, int], 
                                     all_crates: List[Tuple[int, int]], 
                                     layout: List[List[str]], H: int, W: int) -> bool:
        """Check if a crate is permanently blocked and cannot be moved in any direction"""
        blocked_directions = 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (crate_pos[0] + dr, crate_pos[1] + dc)
            
            # Check if this direction is blocked
            if (not (0 <= new_pos[0] < H and 0 <= new_pos[1] < W) or
                layout[new_pos[0]][new_pos[1]] == 'E' or
                new_pos in all_crates):
                blocked_directions += 1
        
        return blocked_directions == 4
    
    def _get_reachable_positions(self, start_pos: Tuple[int, int], 
                                obstacles: List[Tuple[int, int]], 
                                layout: List[List[str]], H: int, W: int) -> set:
        """Get all positions reachable by the agent using BFS"""
        visited = {start_pos}
        queue = deque([start_pos])
        obstacles_set = set(obstacles)
        
        while queue:
            pos = queue.popleft()
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dr, pos[1] + dc)
                
                if (0 <= new_pos[0] < H and 0 <= new_pos[1] < W and
                    new_pos not in visited and
                    layout[new_pos[0]][new_pos[1]] != 'E' and
                    new_pos not in obstacles_set):
                    
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return visited

# Usage function for integration
def validate_generated_level(world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Main validation function to be called by the environment"""
    validator = InvertedBoxEscapeValidator()
    return validator.validate_level(world_state)