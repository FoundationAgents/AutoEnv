import yaml
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from copy import deepcopy

class BackwardsValleyFarmValidator:
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define action capabilities and constraints
        self.actions = ["MoveN", "MoveS", "MoveE", "MoveW", "Wait", 
                      "UseWateringCan", "SpreadFertilizer", "Feed", 
                      "CleanPen", "Compliment", "Insult"]
        
        # Define transition rules for inverse causality
        self.crop_stages = ["Seed", "Sprout", "Young", "HarvestReady"]
        self.animal_states = ["Weak", "Okay", "Thriving"] 
        self.villager_moods = ["Hostile", "Neutral", "Friendly"]
        
        # Reward values from config
        self.crop_values = self.config["reward"]["crop_value"]
        self.animal_values = self.config["reward"]["animal_value"]
        self.social_values = self.config["reward"]["social_value"]
    
    def validate_level(self, world_id: str) -> Tuple[bool, List[str]]:
        """Main validation function that checks both solvability and reward structure"""
        try:
            world_state = self._load_world(world_id)
            issues = []
            
            # 1. LEVEL SOLVABILITY ANALYSIS
            solvability_issues = self._check_level_solvability(world_state)
            issues.extend(solvability_issues)
            
            # 2. REWARD STRUCTURE VALIDATION
            reward_issues = self._validate_reward_structure(world_state)
            issues.extend(reward_issues)
            
            # 3. BASIC STRUCTURAL CHECKS
            structural_issues = self._check_basic_structure(world_state)
            issues.extend(structural_issues)
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        """Load world state from file"""
        file_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"World file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if the level is actually solvable within constraints"""
        issues = []
        
        # 1. ACTION CONSTRAINT ANALYSIS
        max_steps = world_state["globals"]["max_steps"]
        map_size = world_state["tiles"]["size"]
        
        # Check if agent can physically reach all entities
        agent_pos = tuple(world_state["agent"]["pos"])
        reachability_issues = self._check_reachability(world_state, agent_pos)
        issues.extend(reachability_issues)
        
        # 2. TARGET REACHABILITY - Calculate maximum possible farm value
        max_possible_value = self._calculate_max_possible_farm_value(world_state)
        
        # 3. STEP BUDGET ANALYSIS - Check if goals are achievable within step limit
        min_steps_needed = self._estimate_minimum_steps_needed(world_state)
        
        if min_steps_needed > max_steps:
            issues.append(f"Impossible: Need at least {min_steps_needed} steps but only {max_steps} available")
        
        # 4. RESOURCE AVAILABILITY - Check if entities exist to generate meaningful rewards
        if max_possible_value == 0:
            issues.append("Impossible: No entities present that can generate farm value")
        
        # 5. INVERSE CAUSALITY VALIDATION - Ensure the reversed mechanics work properly
        causality_issues = self._validate_inverse_causality_setup(world_state)
        issues.extend(causality_issues)
        
        return issues
    
    def _check_reachability(self, world_state: Dict[str, Any], start_pos: Tuple[int, int]) -> List[str]:
        """Check if agent can reach all important positions"""
        issues = []
        map_size = world_state["tiles"]["size"]
        
        # Build fence map
        fence_positions = set()
        for fence in world_state["objects"]["fences"]:
            fence_positions.add(tuple(fence["pos"]))
        
        # Use BFS to find all reachable positions
        reachable = self._bfs_reachable_positions(start_pos, map_size, fence_positions)
        
        # Check if all entities are reachable
        unreachable_entities = []
        
        for field in world_state["objects"]["fields"]:
            if tuple(field["pos"]) not in reachable:
                unreachable_entities.append(f"Crop at {field['pos']}")
        
        for pen in world_state["objects"]["pens"]:
            if tuple(pen["pos"]) not in reachable:
                unreachable_entities.append(f"Pen at {pen['pos']}")
        
        for villager in world_state["objects"]["villagers"]:
            if tuple(villager["pos"]) not in reachable:
                unreachable_entities.append(f"Villager at {villager['pos']}")
        
        if unreachable_entities:
            issues.append(f"Unreachable entities: {', '.join(unreachable_entities)}")
        
        return issues
    
    def _bfs_reachable_positions(self, start: Tuple[int, int], map_size: List[int], 
                                fences: set) -> set:
        """BFS to find all positions reachable from start"""
        reachable = set()
        queue = [start]
        reachable.add(start)
        
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # N, S, E, W
        
        while queue:
            x, y = queue.pop(0)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < map_size[0] and 0 <= ny < map_size[1] and 
                    (nx, ny) not in fences and (nx, ny) not in reachable):
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
        
        return reachable
    
    def _calculate_max_possible_farm_value(self, world_state: Dict[str, Any]) -> int:
        """Calculate maximum possible farm value from all entities"""
        max_value = 0
        
        # Crops can be harvested when they reach HarvestReady
        for field in world_state["objects"]["fields"]:
            max_value += field["base_value"]
        
        # Animals give value when first reaching Thriving
        for pen in world_state["objects"]["pens"]:
            max_value += pen["base_value"]
        
        # Villagers give value when first reaching Friendly
        for villager in world_state["objects"]["villagers"]:
            max_value += villager["base_value"]
        
        return max_value
    
    def _estimate_minimum_steps_needed(self, world_state: Dict[str, Any]) -> int:
        """Estimate minimum steps needed to achieve maximum farm value"""
        min_steps = 0
        
        # For crops: need to let them grow naturally (neglect is beneficial)
        # Find crops that need the most growth steps
        max_crop_steps = 0
        for field in world_state["objects"]["fields"]:
            current_stage_idx = self.crop_stages.index(field["stage"])
            steps_to_harvest = len(self.crop_stages) - 1 - current_stage_idx
            max_crop_steps = max(max_crop_steps, steps_to_harvest)
        
        # For animals: need to avoid interacting (neglect is beneficial)
        max_animal_steps = 0
        for pen in world_state["objects"]["pens"]:
            current_state_idx = self.animal_states.index(pen["health_state"])
            steps_to_thriving = len(self.animal_states) - 1 - current_state_idx
            max_animal_steps = max(max_animal_steps, steps_to_thriving)
        
        # For villagers: need to insult them to make them friendly
        villager_interactions = 0
        for villager in world_state["objects"]["villagers"]:
            current_mood_idx = self.villager_moods.index(villager["mood"])
            # To go from Hostile->Neutral->Friendly via insults
            if villager["mood"] == "Hostile":
                villager_interactions += 2  # Two insults needed
            elif villager["mood"] == "Neutral": 
                villager_interactions += 1  # One insult needed
            # Already Friendly needs 0 insults
        
        # Add movement overhead (rough estimate)
        entity_count = (len(world_state["objects"]["fields"]) + 
                       len(world_state["objects"]["pens"]) + 
                       len(world_state["objects"]["villagers"]))
        movement_overhead = entity_count // 2  # Conservative estimate
        
        # The minimum is dominated by the longest natural progression
        min_steps = max(max_crop_steps, max_animal_steps) + villager_interactions + movement_overhead
        
        return min_steps
    
    def _validate_inverse_causality_setup(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that inverse causality mechanics can work properly"""
        issues = []
        
        # Check that entities are in states where inverse causality can take effect
        crops_can_progress = False
        for field in world_state["objects"]["fields"]:
            if field["stage"] != "HarvestReady":
                crops_can_progress = True
                break
        
        animals_can_progress = False
        for pen in world_state["objects"]["pens"]:
            if pen["health_state"] != "Thriving":
                animals_can_progress = True
                break
        
        villagers_can_progress = False
        for villager in world_state["objects"]["villagers"]:
            if villager["mood"] != "Friendly":
                villagers_can_progress = True
                break
        
        if not crops_can_progress and not animals_can_progress and not villagers_can_progress:
            issues.append("No entities can progress further - all already at maximum states")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate reward structure follows good design principles"""
        issues = []
        
        # 1. GOAL-ORIENTED REWARDS CHECK
        max_possible_reward = self._calculate_max_possible_farm_value(world_state)
        
        if max_possible_reward < 10:
            issues.append(f"Low reward potential: Maximum possible farm value is only {max_possible_reward}")
        
        # 2. AVOID INCENTIVE MISALIGNMENT
        # Check that meaningful progress gives substantial rewards
        min_individual_reward = min(
            min(self.crop_values.values()) if self.crop_values else 0,
            min(self.animal_values.values()) if self.animal_values else 0,
            min(self.social_values.values()) if self.social_values else 0
        )
        
        if min_individual_reward <= 0:
            issues.append("Some entity types provide no reward value")
        
        # 3. EFFICIENCY INCENTIVE CHECK
        max_steps = world_state["globals"]["max_steps"]
        min_steps_needed = self._estimate_minimum_steps_needed(world_state)
        
        efficiency_ratio = max_possible_reward / max_steps if max_steps > 0 else 0
        if efficiency_ratio < 0.1:
            issues.append(f"Poor efficiency incentive: reward per step ratio is {efficiency_ratio:.3f}")
        
        # 4. REWARD BALANCE CHECK
        crop_total = sum(field["base_value"] for field in world_state["objects"]["fields"])
        animal_total = sum(pen["base_value"] for pen in world_state["objects"]["pens"])
        social_total = sum(villager["base_value"] for villager in world_state["objects"]["villagers"])
        
        total_rewards = crop_total + animal_total + social_total
        if total_rewards == 0:
            issues.append("No reward sources available")
        else:
            # Check if any single category dominates too much (>80%)
            if crop_total > 0.8 * total_rewards:
                issues.append("Reward structure too crop-heavy - lacks diversity")
            elif animal_total > 0.8 * total_rewards:
                issues.append("Reward structure too animal-heavy - lacks diversity")
            elif social_total > 0.8 * total_rewards:
                issues.append("Reward structure too social-heavy - lacks diversity")
        
        return issues
    
    def _check_basic_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Check basic structural requirements"""
        issues = []
        
        # Check required keys exist
        required_keys = ["globals", "agent", "tiles", "objects"]
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required key: {key}")
        
        # Check map size is valid
        if "tiles" in world_state and "size" in world_state["tiles"]:
            size = world_state["tiles"]["size"]
            if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                issues.append(f"Invalid map size: {size}")
        
        # Check agent position is valid
        if "agent" in world_state and "pos" in world_state["agent"]:
            pos = world_state["agent"]["pos"]
            size = world_state["tiles"]["size"]
            if (len(pos) != 2 or pos[0] < 0 or pos[1] < 0 or 
                pos[0] >= size[0] or pos[1] >= size[1]):
                issues.append(f"Invalid agent position: {pos}")
        
        # Check entities have valid positions
        if "objects" in world_state:
            size = world_state["tiles"]["size"]
            for obj_type, obj_list in world_state["objects"].items():
                if obj_type == "fences":
                    continue
                for i, obj in enumerate(obj_list):
                    if "pos" not in obj:
                        issues.append(f"Missing position in {obj_type}[{i}]")
                    else:
                        pos = obj["pos"]
                        if (len(pos) != 2 or pos[0] < 0 or pos[1] < 0 or 
                            pos[0] >= size[0] or pos[1] >= size[1]):
                            issues.append(f"Invalid position in {obj_type}[{i}]: {pos}")
        
        # Check no entities occupy same position
        occupied_positions = {}
        if "agent" in world_state:
            agent_pos = tuple(world_state["agent"]["pos"])
            occupied_positions[agent_pos] = "agent"
        
        if "objects" in world_state:
            for obj_type, obj_list in world_state["objects"].items():
                for i, obj in enumerate(obj_list):
                    if "pos" in obj:
                        pos = tuple(obj["pos"])
                        if pos in occupied_positions:
                            issues.append(f"Position conflict at {pos}: {occupied_positions[pos]} and {obj_type}[{i}]")
                        else:
                            occupied_positions[pos] = f"{obj_type}[{i}]"
        
        return issues

def validate_generated_levels(levels_dir: str = "./levels") -> Dict[str, Any]:
    """Validate all generated levels in the levels directory"""
    validator = BackwardsValleyFarmValidator()
    results = {
        "valid_levels": [],
        "invalid_levels": [],
        "total_levels": 0,
        "validation_summary": {}
    }
    
    if not os.path.exists(levels_dir):
        return results
    
    level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
    results["total_levels"] = len(level_files)
    
    for level_file in level_files:
        world_id = level_file[:-5]  # Remove .yaml extension
        
        try:
            is_valid, issues = validator.validate_level(world_id)
            
            if is_valid:
                results["valid_levels"].append(world_id)
            else:
                results["invalid_levels"].append({
                    "world_id": world_id,
                    "issues": issues
                })
        
        except Exception as e:
            results["invalid_levels"].append({
                "world_id": world_id,
                "issues": [f"Validation failed: {str(e)}"]
            })
    
    # Generate summary
    results["validation_summary"] = {
        "valid_count": len(results["valid_levels"]),
        "invalid_count": len(results["invalid_levels"]),
        "success_rate": len(results["valid_levels"]) / max(1, results["total_levels"])
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Validate a specific level
    validator = BackwardsValleyFarmValidator()
    is_valid, issues = validator.validate_level("world_12345_67890")
    
    if is_valid:
        print("Level is valid!")
    else:
        print("Level validation failed:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Validate all levels
    results = validate_generated_levels()
    print(f"\nValidation Summary:")
    print(f"Total levels: {results['total_levels']}")
    print(f"Valid levels: {results['validation_summary']['valid_count']}")
    print(f"Invalid levels: {results['validation_summary']['invalid_count']}")
    print(f"Success rate: {results['validation_summary']['success_rate']:.2%}")