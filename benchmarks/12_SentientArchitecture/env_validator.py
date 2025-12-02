from typing import Dict, Any, List, Tuple, Optional
import yaml
import random
from copy import deepcopy

class SentientArchitectureValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config['termination']['max_steps']
        self.target_synergy = config['state_template']['globals']['target_synergy']
        
    def validate_level(self, world_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a generated level for solvability and proper reward structure.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # 1. LEVEL SOLVABILITY ANALYSIS
        solvability_issues = self._check_level_solvability(world_state)
        issues.extend(solvability_issues)
        
        # 2. REWARD STRUCTURE VALIDATION
        reward_issues = self._validate_reward_structure(world_state)
        issues.extend(reward_issues)
        
        # 3. BASIC CONSISTENCY CHECKS
        consistency_issues = self._check_basic_consistency(world_state)
        issues.extend(consistency_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _check_level_solvability(self, world_state: Dict[str, Any]) -> List[str]:
        """Check if the level is actually solvable within constraints."""
        issues = []
        
        # ACTION CONSTRAINT ANALYSIS
        buildings = world_state['buildings']
        city = world_state['city']
        
        # Check if any building is in immediate danger
        for building in buildings:
            if building['integrity'] <= 10:
                issues.append(f"Building {building['building_id']} starts with critically low integrity ({building['integrity']})")
            if building['trust'] <= 10:
                issues.append(f"Building {building['building_id']} starts with critically low trust ({building['trust']})")
        
        # TARGET REACHABILITY - Can we reach 100 synergy?
        # Synergy requires 3+ buildings with 80+ trust simultaneously
        min_buildings_needed = 3
        if len(buildings) < min_buildings_needed:
            issues.append(f"Only {len(buildings)} buildings present, need at least {min_buildings_needed} for synergy cascades")
        
        # Resource availability check
        total_bio_materials = city['bio_material_stock']
        total_energy = city['energy_grid_capacity']
        
        # Estimate resource needs for achieving high trust (80+) on 3+ buildings
        # Worst case: all buildings need negotiation (free) + energy allocation + repairs
        worst_case_energy_need = len(buildings) * 30  # Conservative estimate
        worst_case_bio_need = len(buildings) * 20     # For repairs and growth
        
        # Factor in regeneration over max_steps
        total_energy_available = total_energy + (self.max_steps * 3)  # ~3 energy per step regen
        total_bio_available = total_bio_materials + (self.max_steps * 2)  # ~2 bio per step regen
        
        if total_energy_available < worst_case_energy_need:
            issues.append(f"Insufficient total energy resources: need ~{worst_case_energy_need}, have ~{total_energy_available}")
        
        if total_bio_available < worst_case_bio_need:
            issues.append(f"Insufficient total bio-material resources: need ~{worst_case_bio_need}, have ~{total_bio_available}")
        
        # Check for impossible starting conditions
        # If too many buildings have very low trust, it might be impossible to recover all of them
        very_low_trust_buildings = [b for b in buildings if b['trust'] < 30]
        if len(very_low_trust_buildings) > len(buildings) // 2:
            issues.append(f"Too many buildings ({len(very_low_trust_buildings)}) start with very low trust < 30")
        
        # Check for excessive initial conflicts
        conflicts = world_state.get('conflicts', [])
        if len(conflicts) > len(buildings):
            issues.append(f"Too many initial conflicts ({len(conflicts)}) relative to buildings ({len(buildings)})")
        
        # Check if conflicts have manageable intensity
        high_intensity_conflicts = [c for c in conflicts if c['intensity'] > 50]
        if len(high_intensity_conflicts) > 2:
            issues.append(f"Too many high-intensity conflicts ({len(high_intensity_conflicts)})")
        
        # STEP BUDGET ANALYSIS
        # Minimum actions needed: negotiate with 3 buildings + handle conflicts + some repairs/energy
        min_actions_needed = min_buildings_needed + len(conflicts) + len([b for b in buildings if b['integrity'] < 50])
        if min_actions_needed > self.max_steps:
            issues.append(f"Minimum required actions ({min_actions_needed}) exceeds step limit ({self.max_steps})")
        
        # CIRCULAR DEPENDENCY CHECK
        # Check if buildings requiring growth have sufficient bio-materials pathway
        seedling_count = len([b for b in buildings if b['growth_stage'] == 'Seedling'])
        if seedling_count > 0:
            # Each seedling needs 10 bio-materials to grow to Mature, then 20 more to Monumental
            max_growth_cost = seedling_count * 30  # Worst case all go to Monumental
            if total_bio_available < max_growth_cost:
                issues.append(f"Insufficient bio-materials for potential growth: need up to {max_growth_cost} for all growth stages")
        
        return issues
    
    def _validate_reward_structure(self, world_state: Dict[str, Any]) -> List[str]:
        """Validate that reward structure promotes goal achievement over action farming."""
        issues = []
        
        # GOAL-ORIENTED REWARDS CHECK
        config_rewards = self.config.get('reward', {})
        
        # Check that objective completion reward is highest
        completion_reward = config_rewards.get('completion_rewards', {}).get('synergy_target_reached', 0)
        if completion_reward < 30:
            issues.append(f"Objective completion reward too low ({completion_reward}), should be 30+ for goal focus")
        
        # Check trust rewards don't allow easy farming
        trust_per_point = config_rewards.get('trust_rewards', {}).get('per_point', 0)
        max_trust_per_action = 15 * trust_per_point  # Max trust gain per negotiate action
        if max_trust_per_action > 5:
            issues.append(f"Trust rewards too high ({max_trust_per_action} max per action), could enable trust farming")
        
        # INCENTIVE MISALIGNMENT CHECKS
        # Growth rewards should be significant but not higher than completion
        growth_reward = config_rewards.get('growth_rewards', {}).get('stage_completion', 0)
        if growth_reward > completion_reward / 10:
            issues.append(f"Growth rewards ({growth_reward}) too high relative to completion reward")
        
        # Synergy cascade rewards should be meaningful
        cascade_reward = config_rewards.get('cascade_rewards', {}).get('three_plus_high_trust', 0)
        if cascade_reward < 3:
            issues.append("Synergy cascade rewards too low to incentivize cooperation")
        
        # FARMING PREVENTION CHECK
        buildings = world_state['buildings']
        max_possible_trust_farming = len(buildings) * self.max_steps * max_trust_per_action
        if max_possible_trust_farming > completion_reward:
            issues.append(f"Trust farming could yield more ({max_possible_trust_farming}) than completing objective ({completion_reward})")
        
        return issues
    
    def _check_basic_consistency(self, world_state: Dict[str, Any]) -> List[str]:
        """Check basic state consistency and required fields."""
        issues = []
        
        # Required structure checks
        required_keys = ['city', 'buildings', 'conflicts', 'globals']
        for key in required_keys:
            if key not in world_state:
                issues.append(f"Missing required key: {key}")
        
        if 'buildings' in world_state:
            buildings = world_state['buildings']
            
            # Check building count
            if len(buildings) < 4 or len(buildings) > 8:
                issues.append(f"Invalid building count: {len(buildings)}, should be 4-8")
            
            # Check building attributes
            for i, building in enumerate(buildings):
                required_building_keys = ['building_id', 'integrity', 'energy_reserves', 'trust', 'growth_stage', 'mood']
                for key in required_building_keys:
                    if key not in building:
                        issues.append(f"Building {i} missing required key: {key}")
                
                # Value range checks
                if 'integrity' in building and (building['integrity'] < 0 or building['integrity'] > 100):
                    issues.append(f"Building {building.get('building_id', i)} has invalid integrity: {building['integrity']}")
                
                if 'trust' in building and (building['trust'] < 0 or building['trust'] > 100):
                    issues.append(f"Building {building.get('building_id', i)} has invalid trust: {building['trust']}")
                
                if 'energy_reserves' in building and (building['energy_reserves'] < 0 or building['energy_reserves'] > 50):
                    issues.append(f"Building {building.get('building_id', i)} has invalid energy_reserves: {building['energy_reserves']}")
                
                if 'growth_stage' in building and building['growth_stage'] not in ['Seedling', 'Mature', 'Monumental']:
                    issues.append(f"Building {building.get('building_id', i)} has invalid growth_stage: {building['growth_stage']}")
                
                valid_moods = ["Calm", "Restless", "Ambitious", "Contemplative", "Energetic"]
                if 'mood' in building and building['mood'] not in valid_moods:
                    issues.append(f"Building {building.get('building_id', i)} has invalid mood: {building['mood']}")
        
        if 'city' in world_state:
            city = world_state['city']
            required_city_keys = ['bio_material_stock', 'energy_grid_capacity', 'harmony_index', 'synergy_score']
            for key in required_city_keys:
                if key not in city:
                    issues.append(f"City missing required key: {key}")
            
            # Check city resource ranges
            if 'bio_material_stock' in city and city['bio_material_stock'] < 0:
                issues.append(f"Invalid bio_material_stock: {city['bio_material_stock']}")
            
            if 'energy_grid_capacity' in city and city['energy_grid_capacity'] < 0:
                issues.append(f"Invalid energy_grid_capacity: {city['energy_grid_capacity']}")
            
            if 'synergy_score' in city and city['synergy_score'] != 0:
                issues.append(f"Synergy score should start at 0, got: {city['synergy_score']}")
        
        # Check conflicts structure
        if 'conflicts' in world_state:
            for i, conflict in enumerate(world_state['conflicts']):
                required_conflict_keys = ['building_id_1', 'building_id_2', 'intensity']
                for key in required_conflict_keys:
                    if key not in conflict:
                        issues.append(f"Conflict {i} missing required key: {key}")
                
                if 'intensity' in conflict and (conflict['intensity'] <= 0 or conflict['intensity'] > 100):
                    issues.append(f"Conflict {i} has invalid intensity: {conflict['intensity']}")
        
        return issues
    
    def simulate_basic_solvability(self, world_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Run a basic simulation to check if the level is solvable with a simple strategy.
        Returns (is_solvable, reason)
        """
        # Create a copy to avoid modifying original
        sim_state = deepcopy(world_state)
        
        steps_used = 0
        buildings = sim_state['buildings']
        
        # Simple strategy: negotiate with all buildings, then allocate energy/repair as needed
        try:
            # Phase 1: Negotiate with all buildings to boost trust
            for building in buildings:
                if steps_used >= self.max_steps:
                    return False, f"Ran out of steps during negotiation phase (step {steps_used})"
                
                # Simulate negotiation (average positive outcome)
                building['trust'] = min(100, building['trust'] + 8)
                steps_used += 1
            
            # Phase 2: Handle critical repairs
            for building in buildings:
                if building['integrity'] < 50:
                    if steps_used >= self.max_steps:
                        return False, f"Ran out of steps during repair phase (step {steps_used})"
                    
                    if sim_state['city']['bio_material_stock'] >= 10:
                        building['integrity'] = min(100, building['integrity'] + 20)
                        sim_state['city']['bio_material_stock'] -= 10
                        steps_used += 1
            
            # Phase 3: Boost trust high enough for synergy cascade
            high_trust_buildings = [b for b in buildings if b['trust'] >= 80]
            buildings_needing_boost = buildings[:3]  # Focus on first 3
            
            for building in buildings_needing_boost:
                while building['trust'] < 80 and steps_used < self.max_steps:
                    # Simulate additional negotiations
                    building['trust'] = min(100, building['trust'] + 5)
                    steps_used += 1
                    
                    if building['trust'] >= 80:
                        break
            
            # Check if we achieved the goal
            final_high_trust = [b for b in buildings if b['trust'] >= 80]
            if len(final_high_trust) >= 3:
                return True, f"Successfully achieved 3+ high trust buildings in {steps_used} steps"
            else:
                return False, f"Could only achieve {len(final_high_trust)} high-trust buildings, need 3+"
                
        except Exception as e:
            return False, f"Simulation error: {str(e)}"

def validate_generated_world(world_path: str, config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a generated world file.
    Returns (is_valid, list_of_issues)
    """
    try:
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validator = SentientArchitectureValidator(config)
        is_valid, issues = validator.validate_level(world_state)
        
        # Run basic solvability simulation
        if is_valid:
            is_solvable, reason = validator.simulate_basic_solvability(world_state)
            if not is_solvable:
                issues.append(f"Level appears unsolvable: {reason}")
                is_valid = False
        
        return is_valid, issues
        
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]

# Usage example:
if __name__ == "__main__":
    # Example usage
    world_path = "./levels/example_world.yaml"
    config_path = "./config.yaml"
    
    is_valid, issues = validate_generated_world(world_path, config_path)
    
    if is_valid:
        print("✅ Level is valid and appears solvable!")
    else:
        print("❌ Level validation failed:")
        for issue in issues:
            print(f"  - {issue}")