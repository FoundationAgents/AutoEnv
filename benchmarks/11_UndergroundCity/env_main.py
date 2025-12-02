from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import UndergroundObservation
from env_generate import UndergroundWorldGenerator
import yaml
import os
import random
import math
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class SubterraneanMegacityEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = UndergroundObservation()
        super().__init__(env_id, obs_policy)
        self.generator = None
    
    def _dsl_config(self):
        config_path = f"./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = UndergroundWorldGenerator(str(self.env_id), self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        self._t = 0
        self._history = []
        
        if mode == "generate":
            world_id = self._generate_world(seed)
            self._state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            self._state = self._load_world(world_id)
        
        return self.observe_semantic()
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World file not found: {world_path}")
        
        with open(world_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        self._history.append(deepcopy(self._state))
        action_name = action['action']
        params = action.get('params', {})
        
        # Update airflow phase based on cycle
        airflow_cycle = self._state['globals']['airflow_cycle_length']
        if self._t % airflow_cycle == 0 and self._t > 0:
            current_phase = self._state['physics_state']['current_airflow_phase']
            self._state['physics_state']['current_airflow_phase'] = 'reversed' if current_phase == 'normal' else 'normal'
            self._reverse_airflow()
        
        if action_name == 'excavate_cell':
            self._excavate_cell(params.get('x'), params.get('y'))
        elif action_name == 'place_support_column':
            self._place_support_column(params.get('x'), params.get('y'), params.get('material_type'))
        elif action_name == 'dig_ventilation_shaft':
            self._dig_ventilation_shaft(params.get('x'), params.get('y'), params.get('direction'))
        elif action_name == 'install_power_conduit':
            self._install_power_conduit(params.get('path_cells'))
        elif action_name == 'build_district_core':
            self._build_district_core(params.get('x'), params.get('y'))
        elif action_name == 'research_anomaly':
            self._research_anomaly(params.get('research_type'))
        elif action_name == 'diagnostic_scan':
            self._diagnostic_scan(params.get('x'), params.get('y'))
        
        self._update_physics()
        return deepcopy(self._state)
    
    def _excavate_cell(self, x: int, y: int):
        if not self._is_valid_coord(x, y):
            return
        
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        cells = self._state['grid']['cells']
        
        if not cells['excavated'][idx]:
            cells['excavated'][idx] = True
            cells['structure_type'][idx] = 'chamber'
            
            # Counterintuitive stress redistribution
            self._redistribute_stress_excavation(x, y)
            
            # Update power usage
            self._state['metrics']['total_power_usage'] += 2
    
    def _place_support_column(self, x: int, y: int, material_type: str):
        if not self._is_valid_coord(x, y) or material_type not in self._state['agent']['available_materials']:
            return
        
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        cells = self._state['grid']['cells']
        
        if cells['excavated'][idx] and not cells['has_support'][idx]:
            cells['has_support'][idx] = True
            cells['structure_type'][idx] = f'supported_{material_type}'
            
            # Support strength depends on material and gravity gradients
            support_strength = self._calculate_support_strength(x, y, material_type)
            self._apply_support_effects(x, y, support_strength)
            
            self._state['metrics']['total_power_usage'] += 1
    
    def _dig_ventilation_shaft(self, x: int, y: int, direction: str):
        if not self._is_valid_coord(x, y):
            return
        
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        cells = self._state['grid']['cells']
        
        if cells['excavated'][idx]:
            cells['ventilation_shaft'][idx] = True
            
            # Modify airflow vectors
            direction_vectors = {
                'north': (0, -1), 'south': (0, 1),
                'east': (1, 0), 'west': (-1, 0)
            }
            
            if direction in direction_vectors:
                dir_vec = direction_vectors[direction]
                cells['airflow_vector'][idx][0] += dir_vec[0] * 0.5
                cells['airflow_vector'][idx][1] += dir_vec[1] * 0.5
            
            self._state['metrics']['total_power_usage'] += 3
    
    def _install_power_conduit(self, path_cells: List[List[int]]):
        if not path_cells or len(path_cells) > 3:
            return
        
        grid_size = self._state['grid']['size']
        cells = self._state['grid']['cells']
        
        for coord in path_cells:
            if len(coord) >= 2:
                x, y = coord[0], coord[1]
                if self._is_valid_coord(x, y):
                    idx = y * grid_size[0] + x
                    if cells['excavated'][idx]:
                        cells['power_conduit'][idx] = True
                        
                        # Power conduits affect gravity fields
                        self._apply_power_gravity_feedback(x, y)
        
        self._state['agent']['power_storage'] += len(path_cells) * 5
        self._state['metrics']['total_power_usage'] += len(path_cells)
    
    def _build_district_core(self, x: int, y: int):
        if not self._is_valid_coord(x, y):
            return
        
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        cells = self._state['grid']['cells']
        
        # Check prerequisites: excavated, has support, has power
        if (cells['excavated'][idx] and 
            cells['has_support'][idx] and 
            cells['power_conduit'][idx] and
            not cells['district_core'][idx]):
            
            cells['district_core'][idx] = True
            cells['structure_type'][idx] = 'district'
            self._state['agent']['districts_built'] += 1
            
            # Districts consume power and air
            self._state['metrics']['total_power_usage'] += 5
    
    def _research_anomaly(self, research_type: str):
        research = self._state['research']
        
        if research_type == 'gravity_mechanics' and not research['gravity_mechanics_unlocked']:
            research['gravity_mechanics_unlocked'] = True
            self._state['agent']['available_materials'].append('gravity_anchor')
            
        elif research_type == 'airflow_patterns' and not research['airflow_patterns_unlocked']:
            research['airflow_patterns_unlocked'] = True
            self._state['agent']['available_materials'].append('air_compressor')
            
        elif research_type == 'advanced_materials' and not research['advanced_materials_unlocked']:
            research['advanced_materials_unlocked'] = True
            self._state['agent']['available_materials'].extend(['nano_support', 'energy_crystal'])
        
        self._state['metrics']['total_power_usage'] += 4
    
    def _diagnostic_scan(self, x: int, y: int):
        if not self._is_valid_coord(x, y):
            return
        
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        cells = self._state['grid']['cells']
        
        # Reveal more detailed information about the cell
        scan_info = {
            'stress': cells['rock_stress'][idx],
            'airflow': cells['airflow_vector'][idx],
            'hidden_stress_sources': self._get_hidden_stress_sources(x, y)
        }
        
        self._last_action_result = scan_info
        self._state['metrics']['total_power_usage'] += 1
    
    def _redistribute_stress_excavation(self, x: int, y: int):
        """Counterintuitive stress redistribution - excavation can help distant areas"""
        grid_size = self._state['grid']['size']
        cells = self._state['grid']['cells']
        
        # Find mineral vein connections
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if self._is_valid_coord(nx, ny) and (dx != 0 or dy != 0):
                    nidx = ny * grid_size[0] + nx
                    
                    # Counterintuitive: reduce stress in connected areas
                    if abs(dx) + abs(dy) <= 2:  # Within mineral vein range
                        stress_reduction = random.randint(5, 15)
                        cells['rock_stress'][nidx] = max(0, cells['rock_stress'][nidx] - stress_reduction)
    
    def _calculate_support_strength(self, x: int, y: int, material_type: str) -> float:
        """Calculate support effectiveness based on material and anomalous physics"""
        base_strengths = {
            'basic_support': 1.0,
            'gravity_anchor': 2.0,
            'nano_support': 3.0
        }
        
        base_strength = base_strengths.get(material_type, 1.0)
        
        # Strength varies with local stress and gravity gradients
        grid_size = self._state['grid']['size']
        idx = y * grid_size[0] + x
        local_stress = self._state['grid']['cells']['rock_stress'][idx]
        
        # Counterintuitive: higher stress areas give more support effectiveness
        stress_bonus = local_stress / 100.0
        
        return base_strength * (1.0 + stress_bonus)
    
    def _apply_support_effects(self, x: int, y: int, strength: float):
        """Apply support column effects to surrounding area"""
        grid_size = self._state['grid']['size']
        cells = self._state['grid']['cells']
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if self._is_valid_coord(nx, ny):
                    nidx = ny * grid_size[0] + nx
                    stress_reduction = int(strength * 10)
                    cells['rock_stress'][nidx] = max(0, cells['rock_stress'][nidx] - stress_reduction)
    
    def _apply_power_gravity_feedback(self, x: int, y: int):
        """Power conduits affect local gravity fields"""
        grid_size = self._state['grid']['size']
        cells = self._state['grid']['cells']
        idx = y * grid_size[0] + x
        
        # Power conduits stabilize local stress
        stress_stabilization = random.randint(3, 8)
        cells['rock_stress'][idx] = max(0, cells['rock_stress'][idx] - stress_stabilization)
    
    def _reverse_airflow(self):
        """Reverse airflow directions during cyclical changes"""
        cells = self._state['grid']['cells']
        
        for i in range(len(cells['airflow_vector'])):
            cells['airflow_vector'][i][0] *= -1
            cells['airflow_vector'][i][1] *= -1
    
    def _update_physics(self):
        """Update structural integrity and air quality based on current state"""
        cells = self._state['grid']['cells']
        grid_size = self._state['grid']['size']
        
        # Calculate structural integrity
        total_stress = sum(cells['rock_stress'])
        avg_stress = total_stress / len(cells['rock_stress'])
        excavated_count = sum(cells['excavated'])
        support_count = sum(cells['has_support'])
        
        # Base integrity decreases with stress and excavations, increases with supports
        base_integrity = 100 - (avg_stress * 0.4) - (excavated_count * 3) + (support_count * 8)
        self._state['metrics']['structural_integrity'] = max(0, min(100, base_integrity))
        
        # Calculate breathable air
        ventilation_count = sum(cells['ventilation_shaft'])
        district_count = self._state['agent']['districts_built']
        
        total_airflow = sum(abs(v[0]) + abs(v[1]) for v in cells['airflow_vector'])
        avg_airflow = total_airflow / len(cells['airflow_vector'])
        
        base_air = 85 + (ventilation_count * 12) + (avg_airflow * 15) - (district_count * 8) - (excavated_count * 2)
        self._state['metrics']['breathable_air_index'] = max(0, min(100, base_air))
    
    def _get_hidden_stress_sources(self, x: int, y: int) -> List[str]:
        """Get information about hidden stress sources for diagnostic scan"""
        sources = []
        grid_size = self._state['grid']['size']
        
        # Check for nearby mineral veins
        high_stress_neighbors = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if self._is_valid_coord(nx, ny) and (dx != 0 or dy != 0):
                    nidx = ny * grid_size[0] + nx
                    if self._state['grid']['cells']['rock_stress'][nidx] > 70:
                        high_stress_neighbors += 1
        
        if high_stress_neighbors >= 2:
            sources.append("mineral_vein_intersection")
        if x == 1 or x == 3:  # Near fault line locations
            sources.append("geological_fault")
        
        return sources
    
    def _is_valid_coord(self, x: int, y: int) -> bool:
        grid_size = self._state['grid']['size']
        return 0 <= x < grid_size[0] and 0 <= y < grid_size[1]
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        if len(self._history) == 0:
            return 0.0, [], {}
        
        prev_state = self._history[-1]
        curr_state = self._state
        
        total_reward = 0.0
        events = []
        reward_info = {}
        
        # Base survival bonus - small positive base reward
        base_survival_bonus = 0.2  # Reduced from 0.5
        total_reward += base_survival_bonus
        events.append('base_survival')
        reward_info['base_survival'] = base_survival_bonus
        
        # Structural improvement bonus - exponential scaling
        prev_integrity = prev_state['metrics']['structural_integrity']
        curr_integrity = curr_state['metrics']['structural_integrity']
        
        if curr_integrity > prev_integrity:
            improvement = curr_integrity - prev_integrity
            # Exponential scaling rewards larger improvements more
            integrity_bonus = improvement * 2.0 + (improvement ** 1.5) * 0.1
            total_reward += integrity_bonus
            events.append('structural_improvement')
            reward_info['integrity_bonus'] = integrity_bonus
        
        # Air improvement bonus - with diminishing returns at high levels
        prev_air = prev_state['metrics']['breathable_air_index']
        curr_air = curr_state['metrics']['breathable_air_index']
        
        if curr_air > prev_air:
            improvement = curr_air - prev_air
            # Diminishing returns - harder to improve already good air
            if curr_air > 90:
                air_bonus = improvement * 2.0  # Higher reward for excellence
            else:
                air_bonus = improvement * 1.0
            total_reward += air_bonus
            events.append('air_improvement')
            reward_info['air_bonus'] = air_bonus
        
        # District completion bonus - scaled by complexity
        prev_districts = prev_state['agent']['districts_built']
        curr_districts = curr_state['agent']['districts_built']
        
        if curr_districts > prev_districts:
            districts_added = curr_districts - prev_districts
            # Each district becomes more valuable as system becomes more complex
            district_bonus = districts_added * (5.0 + curr_districts * 2.0)
            total_reward += district_bonus
            events.append('district_completed')
            reward_info['district_bonus'] = district_bonus
        
        # Research breakthrough bonus - with synergy bonus
        prev_research = sum(prev_state['research'].values())
        curr_research = sum(curr_state['research'].values())
        
        if curr_research > prev_research:
            research_bonus = 8.0  # Base research bonus
            # Synergy bonus if multiple research completed
            if curr_research >= 2:
                research_bonus += 5.0  # Synergy bonus
            if curr_research >= 3:
                research_bonus += 10.0  # Full research mastery
            total_reward += research_bonus
            events.append('research_breakthrough')
            reward_info['research_bonus'] = research_bonus
        
        # Power milestone bonus - one-time achievement
        curr_power = curr_state['agent']['power_storage']
        if curr_power > 80 and not hasattr(self, '_power_milestone_achieved'):
            power_bonus = 15.0  # Increased significantly
            total_reward += power_bonus
            events.append('power_milestone')
            reward_info['power_bonus'] = power_bonus
            self._power_milestone_achieved = True
        
        # Strategic Excellence Bonus - requires high performance in multiple areas
        excellence_conditions = 0
        if curr_integrity > 85:
            excellence_conditions += 1
        if curr_air > 80:
            excellence_conditions += 1  
        if curr_districts >= 3:
            excellence_conditions += 1
        if curr_research >= 2:
            excellence_conditions += 1
        if curr_power > 75:
            excellence_conditions += 1
        
        if excellence_conditions >= 4:  # Meeting 4+ conditions shows strategic mastery
            excellence_bonus = excellence_conditions * 3.0
            total_reward += excellence_bonus
            events.append('strategic_excellence')
            reward_info['excellence_bonus'] = excellence_bonus
        elif excellence_conditions >= 2:  # Basic competence
            competence_bonus = excellence_conditions * 1.0
            total_reward += competence_bonus
            events.append('strategic_competence')
            reward_info['competence_bonus'] = competence_bonus
        
        # Mission progress bonus - scaled by current performance
        mission_progress = min(1.0, curr_districts / 6.0)
        if mission_progress > 0.5:  # Only reward significant progress
            progress_bonus = (mission_progress - 0.5) * 20.0
            total_reward += progress_bonus
            events.append('mission_progress')
            reward_info['progress_bonus'] = progress_bonus
        
        # Mission complete bonus - ultimate achievement
        if curr_districts >= 6 and curr_integrity > 75 and curr_air > 70 and self._t <= 40:
            completion_bonus = 100.0  # Increased significantly
            total_reward += completion_bonus
            events.append('mission_complete')
            reward_info['completion_bonus'] = completion_bonus
        
        # Smart penalties - only for truly poor performance
        if curr_integrity < 50:  # Only severe structural problems
            integrity_penalty = (50 - curr_integrity) * -0.3
            total_reward += integrity_penalty
            events.append('integrity_penalty')
            reward_info['integrity_malus'] = integrity_penalty
        
        if curr_air < 30:  # Only critical air problems
            air_penalty = (30 - curr_air) * -0.2
            total_reward += air_penalty
            events.append('air_penalty')
            reward_info['air_malus'] = air_penalty
        
        # Failure cascade penalty - for systems failing together
        failing_systems = 0
        if curr_integrity < 40:
            failing_systems += 1
        if curr_air < 40:
            failing_systems += 1
        if curr_power < 20:
            failing_systems += 1
        
        if failing_systems >= 2:  # Multiple system failure
            cascade_penalty = failing_systems * -10.0
            total_reward += cascade_penalty
            events.append('system_cascade')
            reward_info['cascade_penalty'] = cascade_penalty
        
        # Catastrophic failure - complete system collapse
        if curr_integrity <= 0 or curr_air <= 0:
            failure_penalty = -100.0  # Increased penalty
            total_reward += failure_penalty
            events.append('catastrophic_failure')
            reward_info['failure_penalty'] = failure_penalty
        
        return total_reward, events, reward_info
    
    def done(self, state=None) -> bool:
        max_steps = self._state['globals']['max_steps'] if self._state else self.configs["termination"]["max_steps"]
        if self._t >= max_steps:
            return True
        
        if self._state:
            if (self._state['metrics']['structural_integrity'] <= 0 or 
                self._state['metrics']['breathable_air_index'] <= 0):
                return True
        
        return False
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        if not omega:
            return "Environment not initialized"
        
        # Extract data
        # Display step starts from 1 (visual only)
        t = omega.get('t', 0)
        t_display = int(t) + 1
        max_steps = omega.get('max_steps', 40)
        districts = omega.get('districts_built', 0)
        power = omega.get('power_storage', 0)
        integrity = omega.get('structural_integrity', 0)
        air_index = omega.get('breathable_air_index', 0)
        airflow_phase = omega.get('current_airflow_phase', 'normal')
        power_usage = omega.get('total_power_usage', 0)
        
        # Research status
        research = omega.get('research', {})
        gravity_research = "✓" if research.get('gravity_mechanics_unlocked', False) else "✗"
        airflow_research = "✓" if research.get('airflow_patterns_unlocked', False) else "✗"
        materials_research = "✓" if research.get('advanced_materials_unlocked', False) else "✗"
        
        # Grid display
        grid_display = self._render_grid(omega.get('grid_cells', []))
        
        skin_output = f"""=== SUBTERRANEAN MEGACITY CONTROL CENTER ===
Step {t_display}/{max_steps} | Districts: {districts}/6 | Power: {power}

VITAL SYSTEMS:
Structural Integrity: {integrity:.1f}% | Air Quality: {air_index:.1f}%
Airflow Phase: {airflow_phase} | Power Usage: {power_usage}

UNDERGROUND GRID (5x5):
{grid_display}

RESEARCH STATUS:
Gravity Mechanics: {gravity_research} | Airflow Patterns: {airflow_research}
Advanced Materials: {materials_research}

AVAILABLE ACTIONS:
- excavate_cell(x,y): Remove rock and trigger stress redistribution
- place_support_column(x,y,material): Install structural support
- dig_ventilation_shaft(x,y,direction): Create air circulation
- install_power_conduit(path): Establish energy distribution 
- build_district_core(x,y): Convert space to habitable district
- research_anomaly(type): Unlock physics understanding
- diagnostic_scan(x,y): Analyze local conditions"""
        
        return skin_output
    
    def _render_grid(self, grid_cells: List[List[Dict]]) -> str:
        if not grid_cells:
            return "Grid not available"
        
        lines = []
        
        # Header
        lines.append("     0   1   2   3   4")
        lines.append("   ┌───┬───┬───┬───┬───┐")
        
        for y, row in enumerate(grid_cells):
            line = f" {y} │"
            for cell in row:
                symbol = self._get_cell_symbol(cell)
                line += f" {symbol} │"
            lines.append(line)
            
            if y < len(grid_cells) - 1:
                lines.append("   ├───┼───┼───┼───┼───┤")
        
        lines.append("   └───┴───┴───┴───┴───┘")
        
        # Legend
        lines.append("")
        lines.append("Legend: █=Rock, ·=Excavated, ║=Support, ∩=Ventilation")
        lines.append("        ═=Power, ✦=District, ?=Multiple")
        
        return "\n".join(lines)
    
    def _get_cell_symbol(self, cell: Dict) -> str:
        if cell.get('district_core', False):
            return '✦'
        elif cell.get('has_support', False):
            return '║'
        elif cell.get('ventilation_shaft', False):
            return '∩'
        elif cell.get('power_conduit', False):
            return '═'
        elif cell.get('excavated', False):
            return '·'
        else:
            # Show stress level for rock
            stress = cell.get('rock_stress', 0)
            if stress > 80:
                return '█'
            elif stress > 50:
                return '▓'
            elif stress > 20:
                return '▒'
            else:
                return '░'
