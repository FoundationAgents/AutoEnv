import yaml
import json
from datetime import datetime
from typing import Dict, Any, List
import os

class MaxRewardCalculator:
    def __init__(self):
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Reward parameters from config
        self.success_bonus = self.config['reward']['success_bonus']  # 1.0
        self.time_efficiency_factor = self.config['reward']['time_efficiency_factor']  # 0.5
        self.max_steps = self.config['termination']['max_steps']  # 40
        
        # Reaction table from env_main.py
        self.inverted_reaction_table = {
            ("Acid", "Base"): {"product": "Xylene", "pH_change": 2.0, "temp_change": -10},
            ("Solvent", "Acid"): {"product": "Bizarrolene", "pH_change": -1.5, "temp_change": 8},
            ("Acid", "Solvent"): {"product": "Bizarrolene", "pH_change": -1.5, "temp_change": 8},
            ("Base", "Solvent"): {"product": "InvertedAcetate", "pH_change": 1.0, "temp_change": -5}
        }
        
        # Target compounds and their synthesis reactions
        self.synthesis_recipes = {
            "Xylene": ("Acid", "Base"),
            "Bizarrolene": ("Solvent", "Acid"),  # or ("Acid", "Solvent")
            "InvertedAcetate": ("Base", "Solvent")
        }
    
    def calculate_max_reward_for_level(self, level_file: str) -> Dict[str, Any]:
        """Calculate maximum possible reward for a given level."""
        
        # Load level
        with open(f'levels/{level_file}', 'r') as f:
            level_data = yaml.safe_load(f)
        
        target_compound = level_data['globals']['target_compound']
        target_purity = level_data['globals']['target_purity']  # 0.95
        
        # Calculate theoretical maximum reward components:
        
        # 1. Dense reward component: Maximum purity improvement
        # From 0% to target_purity (95%) = 0.95 maximum dense reward
        max_dense_reward = target_purity
        
        # 2. Success bonus component
        # Best case: complete task in minimum steps with maximum efficiency
        # Minimum steps needed for synthesis:
        # - Add reagent 1 (1 step)
        # - Add reagent 2 (1 step) 
        # - Set stir speed to enable reaction (1 step)
        # - Wait for reaction to complete (several steps)
        # - Submit for analysis (1 step)
        # Optimistically: ~8-10 steps minimum
        min_steps_needed = 10
        
        # Time efficiency calculation
        unused_steps = self.max_steps - min_steps_needed
        unused_fraction = unused_steps / self.max_steps
        
        max_success_bonus = self.success_bonus + (self.time_efficiency_factor * unused_fraction)
        
        # Total maximum reward
        max_total_reward = max_dense_reward + max_success_bonus
        
        return {
            "max_reward": round(max_total_reward, 4),
            "calculation_method": "optimal_synthesis_analysis",
            "components": {
                "max_dense_reward": round(max_dense_reward, 4),
                "max_success_bonus": round(max_success_bonus, 4)
            },
            "assumptions": [
                f"Perfect synthesis of {target_compound} to {target_purity*100}% purity",
                f"Optimal completion in {min_steps_needed} steps",
                "No action failures or inefficiencies"
            ],
            "target_compound": target_compound,
            "target_purity": target_purity
        }
    
    def calculate_all_levels(self) -> Dict[str, Any]:
        """Calculate maximum rewards for all levels."""
        
        levels_dir = 'levels'
        level_files = [f for f in os.listdir(levels_dir) if f.endswith('.yaml')]
        level_files.sort()
        
        results = {
            "environment_id": "20250918_150246_env_88_laboratory_experimentation",
            "calculation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "levels": {},
            "summary": {}
        }
        
        max_rewards = []
        
        for level_file in level_files:
            try:
                level_result = self.calculate_max_reward_for_level(level_file)
                results["levels"][level_file] = level_result
                max_rewards.append(level_result["max_reward"])
                print(f"✓ Calculated max reward for {level_file}: {level_result['max_reward']}")
            except Exception as e:
                print(f"✗ Error calculating {level_file}: {e}")
                results["levels"][level_file] = {
                    "error": str(e),
                    "max_reward": 0.0
                }
                max_rewards.append(0.0)
        
        # Calculate summary statistics
        if max_rewards:
            results["summary"] = {
                "total_levels": len(max_rewards),
                "average_max_reward": round(sum(max_rewards) / len(max_rewards), 4),
                "min_max_reward": round(min(max_rewards), 4),
                "max_max_reward": round(max(max_rewards), 4)
            }
        
        return results

if __name__ == "__main__":
    calculator = MaxRewardCalculator()
    results = calculator.calculate_all_levels()
    
    # Save results
    with open('level_max_rewards.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 RESULTS SUMMARY:")
    print(f"Total levels analyzed: {results['summary']['total_levels']}")
    print(f"Average max reward: {results['summary']['average_max_reward']}")
    print(f"Min max reward: {results['summary']['min_max_reward']}")
    print(f"Max max reward: {results['summary']['max_max_reward']}")
    print(f"\n📄 Results saved to level_max_rewards.json")