#!/usr/bin/env python3

import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any, List
from copy import deepcopy

def load_level(filename: str) -> Dict[str, Any]:
    """Load a level file and return its state."""
    with open(f"./levels/{filename}", 'r') as f:
        return yaml.safe_load(f)

def calculate_max_reward_for_level(level_state: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Calculate the theoretical maximum reward for a given level."""
    
    max_steps = level_state["globals"]["max_steps"]
    
    # Calculate maximum possible rewards from each source
    crop_rewards = 0
    animal_rewards = 0
    villager_rewards = 0
    
    # CROP REWARDS
    # Crops that are already HarvestReady can be harvested immediately
    # Crops in earlier stages need time to grow, but we have backwards logic
    for field in level_state["objects"]["fields"]:
        base_value = field["base_value"]
        stage = field["stage"]
        
        if stage == "HarvestReady":
            # Can harvest immediately
            crop_rewards += base_value
        else:
            # Calculate steps needed to reach HarvestReady
            stage_order = ["Seed", "Sprout", "Young", "HarvestReady"]
            current_idx = stage_order.index(stage)
            steps_to_harvest = len(stage_order) - 1 - current_idx
            
            # If we have enough steps, we can get this reward
            if steps_to_harvest <= max_steps:
                crop_rewards += base_value
    
    # ANIMAL REWARDS  
    # Animals that are already Thriving give reward immediately
    # Others need time to reach Thriving (unless we interfere)
    for pen in level_state["objects"]["pens"]:
        base_value = pen["base_value"]
        health_state = pen["health_state"]
        
        if health_state == "Thriving":
            # Already thriving, get reward on first step
            animal_rewards += base_value
        else:
            # Calculate steps needed to reach Thriving
            health_order = ["Weak", "Okay", "Thriving"]
            current_idx = health_order.index(health_state)
            steps_to_thriving = len(health_order) - 1 - current_idx
            
            # If we have enough steps, we can get this reward
            if steps_to_thriving <= max_steps:
                animal_rewards += base_value
    
    # VILLAGER REWARDS
    # Villagers that are already Friendly give reward immediately  
    # Others need to be made Friendly via Insult action (backwards logic)
    for villager in level_state["objects"]["villagers"]:
        base_value = villager["base_value"]
        mood = villager["mood"]
        
        if mood == "Friendly":
            # Already friendly, get reward on first step
            villager_rewards += base_value
        else:
            # Need to insult them to make them friendly
            # Hostile -> Neutral -> Friendly (2 insults)
            # Neutral -> Friendly (1 insult)
            if mood == "Hostile":
                steps_needed = 2  # Two insult actions needed
            else:  # Neutral
                steps_needed = 1  # One insult action needed
            
            # If we have enough steps, we can get this reward
            if steps_needed <= max_steps:
                villager_rewards += base_value
    
    total_max_reward = crop_rewards + animal_rewards + villager_rewards
    
    # Check if total exceeds termination condition (farm_value >= 300)
    # The game ends when farm_value reaches 300, so max reward is capped
    actual_max_reward = min(total_max_reward, 300)
    
    return {
        "max_reward": actual_max_reward,
        "theoretical_max": total_max_reward,
        "breakdown": {
            "crops": crop_rewards,
            "animals": animal_rewards, 
            "villagers": villager_rewards
        },
        "calculation_method": "optimal_path_analysis",
        "max_steps": max_steps,
        "notes": "Assumes optimal play with perfect timing and positioning. Game ends at 300 farm_value."
    }

def main():
    """Calculate maximum rewards for all levels."""
    
    # Get all level files
    level_files = [f for f in os.listdir("./levels/") if f.endswith(".yaml")]
    level_files.sort()
    
    results = {
        "environment_id": "20250918_145530_env_90_agricultural_life_simulation", 
        "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "levels": {},
        "summary": {
            "total_levels": len(level_files),
            "average_max_reward": 0.0,
            "min_max_reward": float('inf'),
            "max_max_reward": 0.0
        }
    }
    
    total_rewards = 0
    
    print(f"Analyzing {len(level_files)} levels...")
    
    for filename in level_files:
        print(f"Processing {filename}...")
        
        try:
            level_state = load_level(filename)
            reward_info = calculate_max_reward_for_level(level_state, filename)
            
            results["levels"][filename] = reward_info
            
            max_reward = reward_info["max_reward"]
            total_rewards += max_reward
            
            # Update summary statistics
            if max_reward < results["summary"]["min_max_reward"]:
                results["summary"]["min_max_reward"] = max_reward
            if max_reward > results["summary"]["max_max_reward"]:
                results["summary"]["max_max_reward"] = max_reward
                
            print(f"  Max reward: {max_reward}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results["levels"][filename] = {
                "max_reward": 0,
                "error": str(e),
                "calculation_method": "failed",
                "notes": "Failed to calculate due to error"
            }
    
    # Calculate average
    if len(level_files) > 0:
        results["summary"]["average_max_reward"] = total_rewards / len(level_files)
    
    # Handle edge case where no levels were processed successfully
    if results["summary"]["min_max_reward"] == float('inf'):
        results["summary"]["min_max_reward"] = 0.0
    
    # Save results to JSON file
    with open("level_max_rewards.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to level_max_rewards.json")
    print(f"Summary:")
    print(f"  Total levels: {results['summary']['total_levels']}")
    print(f"  Average max reward: {results['summary']['average_max_reward']:.1f}")
    print(f"  Min max reward: {results['summary']['min_max_reward']:.1f}")
    print(f"  Max max reward: {results['summary']['max_max_reward']:.1f}")

if __name__ == "__main__":
    main()