import yaml
import os
from typing import Dict, Any, List, Tuple, Set
from copy import deepcopy

class ValleyFarmValidator:
    def __init__(self):
        self.issues = []
        
    def validate_level(self, world_id: str) -> Dict[str, Any]:
        """Main validation entry point"""
        self.issues = []
        
        # Load level file
        try:
            filepath = f"./levels/{world_id}.yaml"
            with open(filepath, "r") as f:
                level_data = yaml.safe_load(f)
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Failed to load level file: {str(e)}"],
                "solvability": "unknown",
                "reward_structure": "unknown"
            }
        
        # Validate solvability
        solvability_result = self._validate_solvability(level_data)
        
        # Validate reward structure
        reward_result = self._validate_reward_structure(level_data)
        
        # Validate basic structure
        structure_result = self._validate_structure(level_data)
        
        overall_valid = (solvability_result["valid"] and 
                        reward_result["valid"] and 
                        structure_result["valid"])
        
        return {
            "valid": overall_valid,
            "issues": self.issues,
            "solvability": solvability_result,
            "reward_structure": reward_result,
            "structure": structure_result
        }
    
    def _validate_solvability(self, level_data: Dict[str, Any]) -> Dict[str, Any]:
        """Critical check for level solvability"""
        solvability_issues = []
        
        # Extract key game elements
        agent_inventory = level_data["agent"]["inventory"]
        crop_fields = level_data["objects"]["crop_fields"]
        barns = level_data["objects"]["barns"]
        villagers = level_data["objects"]["villagers"]
        market_pos = level_data["objects"]["market"]
        
        # 1. RESOURCE AVAILABILITY CHECK
        seeds = agent_inventory["seeds"]
        water = agent_inventory["water"]
        feed = agent_inventory["animal_feed"]
        gifts = agent_inventory["gifts"]
        
        if seeds <= 0:
            solvability_issues.append("No seeds available - cannot plant crops")
        if water <= 0:
            solvability_issues.append("No water available - cannot grow crops")
        if feed <= 0:
            solvability_issues.append("No animal feed available - cannot feed animals")
        if gifts <= 0:
            solvability_issues.append("No gifts available - cannot improve relationships")
        
        # 2. CROP PRODUCTION VIABILITY
        available_fields = len(crop_fields)
        if available_fields == 0:
            solvability_issues.append("No crop fields available")
        
        # Each crop needs 1 seed + 2 water (seedling->growing->mature)
        max_crops_by_seeds = seeds
        max_crops_by_water = water // 2
        max_producible_crops = min(max_crops_by_seeds, max_crops_by_water, available_fields)
        
        if max_producible_crops == 0:
            solvability_issues.append("Cannot produce any crops due to resource constraints")
        
        # 3. ANIMAL PRODUCTION VIABILITY
        if len(barns) == 0:
            solvability_issues.append("No barns available for animal production")
        
        # Each animal needs feed to produce products (every 5 steps while sated for 10 steps)
        max_feeding_cycles = feed
        if max_feeding_cycles == 0:
            solvability_issues.append("Cannot feed any animals")
        
        # 4. SOCIAL INTERACTION VIABILITY
        if len(villagers) == 0:
            solvability_issues.append("No villagers available for social interaction")
        
        max_relationship_gains = gifts * 5  # Each gift gives +5 relationship
        if max_relationship_gains == 0:
            solvability_issues.append("Cannot improve any relationships")
        
        # 5. MARKET ACCESS CHECK
        if market_pos is None:
            solvability_issues.append("No market available - cannot sell goods")
        
        # 6. STEP BUDGET ANALYSIS
        max_steps = level_data.get("termination", {}).get("max_steps", 50)
        
        # Minimum steps needed for basic production cycle:
        # - Move to field (up to 28 steps worst case on 15x15 grid)
        # - Plant + water twice + harvest = 3 steps per crop
        # - Move to barn + feed animal = 2 steps per animal
        # - Wait for animal production (5 steps) + collect = 6 steps per animal
        # - Move to villager + give gift = 2 steps per gift
        # - Move to market + sell = 2 steps
        
        estimated_min_steps = max_producible_crops * 3 + len(barns) * 8 + len(villagers) * 2 + 2
        if estimated_min_steps > max_steps * 0.8:  # Allow some buffer for movement
            solvability_issues.append(f"Estimated minimum steps ({estimated_min_steps}) may exceed step budget ({max_steps})")
        
        # 7. POSITION ACCESSIBILITY CHECK
        all_positions = set()
        agent_pos = tuple(level_data["agent"]["pos"])
        all_positions.add(agent_pos)
        
        for field in crop_fields:
            all_positions.add(tuple(field["pos"]))
        for barn in barns:
            all_positions.add(tuple(barn["pos"]))
        for villager in villagers:
            all_positions.add(tuple(villager["pos"]))
        all_positions.add(tuple(market_pos))
        
        # Check if all positions are within bounds
        for pos in all_positions:
            x, y = pos
            if not (0 <= x < 15 and 0 <= y < 15):
                solvability_issues.append(f"Position {pos} is out of bounds")
        
        # 8. POSITION UNIQUENESS CHECK
        position_list = list(all_positions)
        if len(position_list) != len(set(position_list)):
            solvability_issues.append("Some entities share the same position")
        
        # 9. MINIMUM VIABLE REWARD CHECK
        # Agent should be able to earn at least some reward
        potential_crop_reward = max_producible_crops * 0.1
        potential_animal_reward = len(barns) * max_feeding_cycles * 0.2  # Assuming 1 product per feeding
        potential_relationship_reward = max_relationship_gains * 0.5
        potential_market_reward = (max_producible_crops + len(barns)) * 0.1  # Rough estimate
        
        total_potential_reward = (potential_crop_reward + potential_animal_reward + 
                                potential_relationship_reward + potential_market_reward)
        
        if total_potential_reward <= 0:
            solvability_issues.append("Level has zero reward potential")
        
        self.issues.extend(solvability_issues)
        
        return {
            "valid": len(solvability_issues) == 0,
            "issues": solvability_issues,
            "max_producible_crops": max_producible_crops,
            "max_feeding_cycles": max_feeding_cycles,
            "total_potential_reward": total_potential_reward
        }
    
    def _validate_reward_structure(self, level_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reward structure design"""
        reward_issues = []
        
        # Load reward configuration
        with open("./config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        reward_config = config.get("reward", {})
        
        # 1. GOAL-ORIENTED REWARDS CHECK
        harvest_reward = reward_config.get("harvest_reward", 0)
        product_reward = reward_config.get("product_reward", 0)
        relationship_multiplier = reward_config.get("relationship_reward_multiplier", 0)
        market_multiplier = reward_config.get("market_reward_multiplier", 0)
        
        if harvest_reward <= 0:
            reward_issues.append("Harvest reward is zero or negative")
        if product_reward <= 0:
            reward_issues.append("Product reward is zero or negative")
        if relationship_multiplier <= 0:
            reward_issues.append("Relationship reward multiplier is zero or negative")
        if market_multiplier <= 0:
            reward_issues.append("Market reward multiplier is zero or negative")
        
        # 2. REWARD PROPORTIONALITY CHECK
        # Market sales should provide significant reward (goal achievement)
        # Direct production should provide medium rewards (progress)
        # Individual actions should provide small rewards (process)
        
        if market_multiplier < harvest_reward:
            reward_issues.append("Market sales should be more rewarding than individual harvests")
        
        if product_reward < harvest_reward:
            reward_issues.append("Animal products should be more valuable than crops (higher effort)")
        
        # 3. AVOID ACTION GRINDING
        # Calculate potential reward from different strategies
        agent_inventory = level_data["agent"]["inventory"]
        
        max_harvest_reward = agent_inventory["seeds"] * harvest_reward
        max_gift_reward = agent_inventory["gifts"] * 5 * relationship_multiplier  # 5 points per gift
        
        # Market reward should dominate individual action rewards
        estimated_market_reward = (agent_inventory["seeds"] + len(level_data["objects"]["barns"])) * 2 * market_multiplier
        
        if max_harvest_reward > estimated_market_reward:
            reward_issues.append("Individual harvesting more rewarding than market strategy - encourages action grinding")
        
        # 4. EFFICIENCY INCENTIVE CHECK
        # The reward structure should incentivize completing objectives efficiently
        # rather than taking as many actions as possible
        
        total_possible_actions = 50  # Max steps
        min_reward_per_action = 0.01  # Minimum meaningful reward
        
        if harvest_reward * agent_inventory["seeds"] > min_reward_per_action * total_possible_actions:
            # This is actually good - meaningful rewards for productive actions
            pass
        else:
            reward_issues.append("Individual productive actions provide insufficient reward incentive")
        
        # 5. SPARSE REWARD VALIDATION
        # Rewards should come from meaningful achievements, not every action
        reward_events = len(reward_config.get("events", []))
        if reward_events == 0:
            reward_issues.append("No reward events defined")
        
        # 6. RELATIONSHIP REWARD BALANCE
        max_relationship_reward = sum(v.get("relationship", 0) for v in level_data["objects"]["villagers"])
        max_relationship_reward += agent_inventory["gifts"] * 5  # Additional from gifts
        max_relationship_reward *= relationship_multiplier
        
        # Relationship rewards shouldn't completely dominate other strategies
        if max_relationship_reward > estimated_market_reward * 2:
            reward_issues.append("Relationship rewards may overshadow economic activities")
        
        self.issues.extend(reward_issues)
        
        return {
            "valid": len(reward_issues) == 0,
            "issues": reward_issues,
            "reward_balance": {
                "harvest_reward": harvest_reward,
                "product_reward": product_reward,
                "max_relationship_reward": max_relationship_reward,
                "estimated_market_reward": estimated_market_reward
            }
        }
    
    def _validate_structure(self, level_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic level structure"""
        structure_issues = []
        
        # 1. Required sections
        required_sections = ["agent", "objects", "termination"]
        for section in required_sections:
            if section not in level_data:
                structure_issues.append(f"Missing required section: {section}")
        
        # 2. Agent structure
        if "agent" in level_data:
            agent = level_data["agent"]
            if "pos" not in agent or len(agent["pos"]) != 2:
                structure_issues.append("Agent position invalid")
            if "inventory" not in agent:
                structure_issues.append("Agent inventory missing")
        
        # 3. Objects structure
        if "objects" in level_data:
            objects = level_data["objects"]
            required_objects = ["crop_fields", "barns", "cottages", "villagers", "market"]
            for obj_type in required_objects:
                if obj_type not in objects:
                    structure_issues.append(f"Missing object type: {obj_type}")
        
        # 4. Termination structure
        if "termination" in level_data:
            termination = level_data["termination"]
            if "max_steps" not in termination:
                structure_issues.append("Missing max_steps in termination")
            elif termination["max_steps"] <= 0:
                structure_issues.append("max_steps must be positive")
        
        self.issues.extend(structure_issues)
        
        return {
            "valid": len(structure_issues) == 0,
            "issues": structure_issues
        }

# Convenience function for external use
def validate_valley_farm_level(world_id: str) -> Dict[str, Any]:
    """Validate a Valley Farm level and return detailed results"""
    validator = ValleyFarmValidator()
    return validator.validate_level(world_id)

# Example usage and testing
if __name__ == "__main__":
    # This would be called after generating a level
    # result = validate_valley_farm_level("valley_farm_20240115_143022_seed42")
    # print("Validation Result:", result)
    pass