import yaml
from collections import deque
from typing import Dict, Any, List, Tuple, Optional, Set
import os

class UndergroundRuinValidator:
    def __init__(self):
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
    def validate_level(self, world_id: str) -> Tuple[bool, List[str]]:
        """
        Main validation function that checks if a generated level is valid and solvable.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Load the world state
            world_state = self._load_world(world_id)
            
            # 1. Basic structure validation
            structure_issues = self._validate_structure(world_state)
            issues.extend(structure_issues)
            
            # 2. Solvability analysis - most critical check
            solvability_issues = self._validate_solvability(world_state)
            issues.extend(solvability_issues)
            
            # 3. Reward structure validation
            reward_issues = self._validate_reward_structure(world_state)
            issues.extend(reward_issues)
            
            # 4. Action constraint analysis
            action_issues = self._validate_action_constraints(world_state)
            issues.extend(action_issues)
            
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Failed to load or parse world {world_id}: {str(e)}"]
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """Load world state from file"""
        file_path = f"./levels/{world_id}.yaml"
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate basic world structure and required components"""
        issues = []
        
        try:
            # Check required keys exist
            if "tiles" not in world_state:
                issues.append("Missing 'tiles' in world state")
                return issues
                
            if "agent" not in world_state:
                issues.append("Missing 'agent' in world state")
                return issues
            
            tiles = world_state["tiles"]
            agent = world_state["agent"]
            
            # Check grid dimensions
            if len(tiles) != 11:
                issues.append(f"Grid height should be 11, got {len(tiles)}")
            
            if len(tiles) > 0 and len(tiles[0]) != 11:
                issues.append(f"Grid width should be 11, got {len(tiles[0])}")
            
            # Check for required elements
            treasure_count = 0
            fire_count = 0
            has_outer_walls = True
            
            for y, row in enumerate(tiles):
                for x, tile in enumerate(row):
                    if tile == "Treasure":
                        treasure_count += 1
                    elif tile == "Fire":
                        fire_count += 1
                    
                    # Check outer wall
                    if (y == 0 or y == len(tiles)-1 or x == 0 or x == len(row)-1):
                        if tile != "Wall":
                            has_outer_walls = False
            
            if treasure_count == 0:
                issues.append("No treasure found in level")
            elif treasure_count > 1:
                issues.append(f"Multiple treasures found ({treasure_count}), should have exactly 1")
            
            if fire_count < 5 or fire_count > 8:
                issues.append(f"Fire pit count ({fire_count}) outside expected range [5-8]")
            
            if not has_outer_walls:
                issues.append("Incomplete outer wall boundary")
            
            # Check agent position validity
            agent_pos = agent.get("pos", [0, 0])
            if len(agent_pos) != 2:
                issues.append("Invalid agent position format")
            else:
                x, y = agent_pos
                if not (0 <= x < len(tiles[0]) and 0 <= y < len(tiles)):
                    issues.append(f"Agent position {agent_pos} out of bounds")
                elif tiles[y][x] not in ["Empty", "Treasure"]:  # Agent can spawn on treasure
                    issues.append(f"Agent spawned on invalid tile type: {tiles[y][x]}")
            
        except Exception as e:
            issues.append(f"Structure validation error: {str(e)}")
        
        return issues
    
    def _validate_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """
        CRITICAL: Check if the treasure is reachable from agent spawn within step limit.
        This is the most important validation to prevent impossible levels.
        """
        issues = []
        
        try:
            tiles = world_state["tiles"]
            agent_pos = tuple(world_state["agent"]["pos"])
            max_steps = world_state.get("globals", {}).get("max_steps", 40)
            
            # Find treasure position
            treasure_pos = None
            for y, row in enumerate(tiles):
                for x, tile in enumerate(row):
                    if tile == "Treasure":
                        treasure_pos = (x, y)
                        break
                if treasure_pos:
                    break
            
            if not treasure_pos:
                issues.append("CRITICAL: No treasure found for solvability check")
                return issues
            
            # BFS to check reachability within step limit
            is_reachable, min_steps = self._bfs_pathfind(tiles, agent_pos, treasure_pos)
            
            if not is_reachable:
                issues.append("CRITICAL: Treasure is completely unreachable from agent spawn")
            elif min_steps > max_steps:
                issues.append(f"CRITICAL: Treasure requires {min_steps} steps but only {max_steps} available")
            
            # Additional checks for common impossible patterns
            blocking_issues = self._check_blocking_patterns(tiles, agent_pos, treasure_pos)
            issues.extend(blocking_issues)
            
        except Exception as e:
            issues.append(f"CRITICAL: Solvability check failed: {str(e)}")
        
        return issues
    
    def _bfs_pathfind(self, tiles: List[List[str]], start: Tuple[int, int], target: Tuple[int, int]) -> Tuple[bool, int]:
        """
        BFS pathfinding to check if target is reachable and minimum steps required.
        Returns (is_reachable, min_steps)
        """
        if start == target:
            return True, 0
        
        queue = deque([(start, 0)])
        visited = {start}
        height, width = len(tiles), len(tiles[0])
        
        while queue:
            (x, y), steps = queue.popleft()
            
            # Explore all 4 directions
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) in visited:
                    continue
                
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                
                tile_type = tiles[ny][nx]
                
                # Can't move through walls or water
                if tile_type in ["Wall", "Water"]:
                    continue
                
                # Don't pathfind through fire (it's lethal)
                if tile_type == "Fire":
                    continue
                
                if (nx, ny) == target:
                    return True, steps + 1
                
                visited.add((nx, ny))
                queue.append(((nx, ny), steps + 1))
        
        return False, float('inf')
    
    def _check_blocking_patterns(self, tiles: List[List[str]], agent_pos: Tuple[int, int], treasure_pos: Tuple[int, int]) -> List[str]:
        """Check for common patterns that make levels impossible"""
        issues = []
        
        # Check if treasure is surrounded by impassable tiles
        tx, ty = treasure_pos
        accessible_neighbors = 0
        
        for dx, dy in self.directions:
            nx, ny = tx + dx, ty + dy
            if (0 <= nx < len(tiles[0]) and 0 <= ny < len(tiles) and 
                tiles[ny][nx] not in ["Wall", "Water"]):
                accessible_neighbors += 1
        
        if accessible_neighbors == 0:
            issues.append("CRITICAL: Treasure is completely surrounded by walls/water")
        
        # Check if agent spawn area is isolated
        reachable_area = self._get_reachable_area(tiles, agent_pos)
        if treasure_pos not in reachable_area:
            issues.append("CRITICAL: Agent spawn area is isolated from treasure")
        
        return issues
    
    def _get_reachable_area(self, tiles: List[List[str]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all positions reachable from start position"""
        visited = set()
        queue = deque([start])
        height, width = len(tiles), len(tiles[0])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) in visited:
                    continue
                
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                
                tile_type = tiles[ny][nx]
                
                # Can move through Empty, Fire (dangerous but passable), and Treasure
                if tile_type in ["Empty", "Fire", "Treasure"]:
                    queue.append((nx, ny))
        
        return visited
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that reward structure incentivizes goal achievement"""
        issues = []
        
        # The current reward structure is binary: +1 for treasure, 0 otherwise
        # This is actually well-designed for this environment
        
        # Check that there's exactly one high-value goal (treasure)
        tiles = world_state["tiles"]
        treasure_count = sum(row.count("Treasure") for row in tiles)
        
        if treasure_count != 1:
            issues.append(f"Reward structure issue: Should have exactly 1 treasure, found {treasure_count}")
        
        # The sparse reward design is appropriate - no action grinding possible
        # since only treasure gives reward and it immediately terminates the episode
        
        return issues
    
    def _validate_action_constraints(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that actions have sufficient power to solve the level"""
        issues = []
        
        try:
            tiles = world_state["tiles"]
            agent_pos = tuple(world_state["agent"]["pos"])
            
            # Check that movement actions can actually navigate the maze
            # (This is implicitly checked in solvability, but we can add specific checks)
            
            # Verify that the agent has at least some movement options from spawn
            moveable_directions = 0
            x, y = agent_pos
            
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < len(tiles[0]) and 0 <= ny < len(tiles) and 
                    tiles[ny][nx] not in ["Wall", "Water"]):
                    moveable_directions += 1
            
            if moveable_directions == 0:
                issues.append("CRITICAL: Agent has no valid moves from spawn position")
            
            # Check that there's enough open space for navigation
            total_cells = len(tiles) * len(tiles[0])
            passable_cells = 0
            
            for row in tiles:
                for tile in row:
                    if tile not in ["Wall", "Water"]:
                        passable_cells += 1
            
            passable_ratio = passable_cells / total_cells
            if passable_ratio < 0.1:  # Less than 10% passable
                issues.append(f"Warning: Very limited passable area ({passable_ratio:.1%})")
            
        except Exception as e:
            issues.append(f"Action constraint validation error: {str(e)}")
        
        return issues
    
    def validate_batch(self, world_ids: List[str]) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate multiple levels and return results"""
        results = {}
        
        for world_id in world_ids:
            results[world_id] = self.validate_level(world_id)
        
        return results
    
    def generate_validation_report(self, world_ids: List[str]) -> str:
        """Generate a human-readable validation report"""
        results = self.validate_batch(world_ids)
        
        report = "=== UNDERGROUND RUIN LEVEL VALIDATION REPORT ===\n\n"
        
        valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
        total_count = len(results)
        
        report += f"Summary: {valid_count}/{total_count} levels passed validation\n\n"
        
        for world_id, (is_valid, issues) in results.items():
            status = "✓ VALID" if is_valid else "✗ INVALID"
            report += f"{world_id}: {status}\n"
            
            if issues:
                for issue in issues:
                    if "CRITICAL" in issue:
                        report += f"  🚨 {issue}\n"
                    elif "Warning" in issue:
                        report += f"  ⚠️  {issue}\n"
                    else:
                        report += f"  • {issue}\n"
            
            report += "\n"
        
        return report

def validate_generated_level(world_id: str) -> Tuple[bool, List[str]]:
    """Convenience function for validating a single level"""
    validator = UndergroundRuinValidator()
    return validator.validate_level(world_id)

def validate_all_levels_in_directory(levels_dir: str = "./levels") -> Dict[str, Tuple[bool, List[str]]]:
    """Validate all levels in the levels directory"""
    validator = UndergroundRuinValidator()
    world_ids = []
    
    if os.path.exists(levels_dir):
        for filename in os.listdir(levels_dir):
            if filename.endswith('.yaml'):
                world_ids.append(filename[:-5])  # Remove .yaml extension
    
    return validator.validate_batch(world_ids)
