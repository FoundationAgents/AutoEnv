import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
from collections import deque

class RewardCalculator:
    def __init__(self):
        self.config_path = "./config.yaml"
        self.levels_dir = "./levels"
        self.load_config()
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_level_files(self) -> List[str]:
        """Get all level files from the levels directory."""
        level_files = []
        for filename in os.listdir(self.levels_dir):
            if filename.endswith('.yaml'):
                level_files.append(filename)
        return sorted(level_files)
    
    def load_level(self, level_filename: str) -> Dict[str, Any]:
        """Load a level from YAML file."""
        level_path = os.path.join(self.levels_dir, level_filename)
        with open(level_path, 'r') as f:
            return yaml.safe_load(f)
    
    def is_position_safe(self, pos: List[int], ice_tiles: List[Dict]) -> bool:
        """Check if a position is safe (not on ice)."""
        for ice_tile in ice_tiles:
            if ice_tile['pos'] == pos:
                return False
        return True
    
    def is_position_valid(self, pos: List[int], grid_size: List[int]) -> bool:
        """Check if position is within grid boundaries."""
        return (0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1])
    
    def get_neighbors(self, pos: List[int]) -> List[List[int]]:
        """Get all neighboring positions (4-directional movement)."""
        x, y = pos
        return [
            [x, y-1],  # North
            [x, y+1],  # South
            [x+1, y],  # East
            [x-1, y]   # West
        ]
    
    def find_shortest_path(self, start: List[int], goal: List[int], 
                          ice_tiles: List[Dict], grid_size: List[int]) -> int:
        """Find shortest safe path from start to goal using BFS."""
        if start == goal:
            return 0
        
        queue = deque([(start, 0)])
        visited = set()
        visited.add(tuple(start))
        
        while queue:
            current_pos, steps = queue.popleft()
            
            for next_pos in self.get_neighbors(current_pos):
                if (self.is_position_valid(next_pos, grid_size) and 
                    self.is_position_safe(next_pos, ice_tiles) and 
                    tuple(next_pos) not in visited):
                    
                    if next_pos == goal:
                        return steps + 1
                    
                    visited.add(tuple(next_pos))
                    queue.append((next_pos, steps + 1))
        
        return -1  # No path found
    
    def calculate_max_reward(self, level_filename: str) -> Dict[str, Any]:
        """Calculate maximum possible reward for a level."""
        level_data = self.load_level(level_filename)
        
        # Extract key information
        start_pos = level_data['agent']['start_pos']
        goal_pos = level_data['objects']['goal_flag']['pos']
        ice_tiles = level_data['objects']['ice_tiles']
        grid_size = level_data['globals']['grid_size']
        max_steps = level_data['globals']['max_steps']
        
        # Get reward values from config
        success_reward = self.config['reward']['goal_values']['success']
        failure_reward = self.config['reward']['goal_values']['failure']
        timeout_reward = self.config['reward']['goal_values']['timeout']
        
        # Check if goal is reachable
        shortest_path_length = self.find_shortest_path(start_pos, goal_pos, ice_tiles, grid_size)
        
        result = {
            "max_reward": 0.0,
            "calculation_method": "path_analysis",
            "notes": "",
            "is_solvable": False,
            "shortest_path_length": shortest_path_length,
            "steps_remaining": 0
        }
        
        if shortest_path_length == -1:
            # No path exists - maximum reward is 0 (failure or timeout)
            result["max_reward"] = max(failure_reward, timeout_reward)
            result["notes"] = "No safe path exists from start to goal"
        elif shortest_path_length <= max_steps:
            # Path exists within step limit - maximum reward is success reward
            result["max_reward"] = success_reward
            result["is_solvable"] = True
            result["steps_remaining"] = max_steps - shortest_path_length
            result["notes"] = f"Goal reachable in {shortest_path_length} steps with {result['steps_remaining']} steps to spare"
        else:
            # Path exists but too long - maximum reward is timeout reward
            result["max_reward"] = timeout_reward
            result["notes"] = f"Path exists ({shortest_path_length} steps) but exceeds max_steps ({max_steps})"
        
        return result
    
    def calculate_all_levels(self) -> Dict[str, Any]:
        """Calculate maximum rewards for all levels."""
        level_files = self.get_level_files()
        results = {
            "environment_id": "20250918_220928_env_84_icy_terrain_navigation",
            "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "levels": {},
            "summary": {
                "total_levels": len(level_files),
                "solvable_levels": 0,
                "average_max_reward": 0.0,
                "min_max_reward": float('inf'),
                "max_max_reward": float('-inf')
            }
        }
        
        total_reward = 0.0
        solvable_count = 0
        
        for level_file in level_files:
            print(f"Analyzing {level_file}...")
            level_result = self.calculate_max_reward(level_file)
            results["levels"][level_file] = level_result
            
            # Update summary statistics
            reward = level_result["max_reward"]
            total_reward += reward
            
            if level_result["is_solvable"]:
                solvable_count += 1
            
            if reward < results["summary"]["min_max_reward"]:
                results["summary"]["min_max_reward"] = reward
            if reward > results["summary"]["max_max_reward"]:
                results["summary"]["max_max_reward"] = reward
        
        results["summary"]["solvable_levels"] = solvable_count
        results["summary"]["average_max_reward"] = total_reward / len(level_files) if level_files else 0.0
        
        return results

def main():
    calculator = RewardCalculator()
    results = calculator.calculate_all_levels()
    
    # Save results to JSON file
    output_file = "level_max_rewards.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total levels analyzed: {results['summary']['total_levels']}")
    print(f"Solvable levels: {results['summary']['solvable_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']:.2f}")
    print(f"Min max reward: {results['summary']['min_max_reward']:.2f}")
    print(f"Max max reward: {results['summary']['max_max_reward']:.2f}")

if __name__ == "__main__":
    main()