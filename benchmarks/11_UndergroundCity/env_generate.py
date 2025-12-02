from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import time

class UndergroundWorldGenerator(WorldGenerator):
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        world_id = self._generate_world_id(seed)
        
        # Get base state from config template
        base_state = self.config['state_template']
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        world_state['world_id'] = world_id
        
        # Save to file
        if save_path is None:
            os.makedirs('./levels/', exist_ok=True)
            save_path = f'./levels/{world_id}.yaml'
        
        with open(save_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        state = yaml.safe_load(yaml.dump(base_state))
        grid_size = state['grid']['size']
        total_cells = grid_size[0] * grid_size[1]
        
        # Initialize grid cell arrays
        state['grid']['cells']['rock_stress'] = [0] * total_cells
        state['grid']['cells']['airflow_vector'] = [[0.0, 0.0]] * total_cells
        state['grid']['cells']['structure_type'] = ['rock'] * total_cells
        state['grid']['cells']['has_support'] = [False] * total_cells
        state['grid']['cells']['excavated'] = [False] * total_cells
        state['grid']['cells']['district_core'] = [False] * total_cells
        state['grid']['cells']['power_conduit'] = [False] * total_cells
        state['grid']['cells']['ventilation_shaft'] = [False] * total_cells
        
        # Step 1: Generate rock stress map
        self._generate_rock_stress_map(state, grid_size)
        
        # Step 2: Generate airflow field
        self._generate_airflow_field(state, grid_size)
        
        # Step 3: Place starting infrastructure
        self._place_starting_infrastructure(state, grid_size)
        
        # Step 4: Calculate initial physics
        self._calculate_initial_physics(state)
        
        return state
    
    def _generate_rock_stress_map(self, state: Dict[str, Any], grid_size: list):
        """Create realistic geological stress patterns"""
        cells = state['grid']['cells']
        
        # Base stress distribution
        for i in range(len(cells['rock_stress'])):
            cells['rock_stress'][i] = random.randint(10, 90)
        
        # Add mineral veins (linear high-stress patterns)
        num_veins = random.randint(1, 3)
        for _ in range(num_veins):
            if random.random() < 0.3:  # mineral_vein_probability
                # Create a diagonal vein
                start_x = random.randint(0, grid_size[0]-1)
                start_y = 0
                dx = random.choice([-1, 0, 1])
                
                x, y = start_x, start_y
                while 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                    idx = y * grid_size[0] + x
                    cells['rock_stress'][idx] = min(95, cells['rock_stress'][idx] + random.randint(20, 40))
                    x += dx
                    y += 1
        
        # Add fault lines (create stress concentrations)
        for _ in range(2):  # fault_line_count
            fault_x = random.randint(1, grid_size[0]-2)
            for y in range(grid_size[1]):
                idx = y * grid_size[0] + fault_x
                cells['rock_stress'][idx] = min(100, cells['rock_stress'][idx] + random.randint(15, 30))
    
    def _generate_airflow_field(self, state: Dict[str, Any], grid_size: list):
        """Establish initial air current directions and magnitudes"""
        cells = state['grid']['cells']
        
        # Generate predominant flow direction
        base_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        predominant_dir = random.choice(base_directions)
        
        for i in range(len(cells['airflow_vector'])):
            base_strength = random.uniform(0.2, 0.8)
            
            # Apply directional bias
            if random.random() < 0.6:  # predominant_direction_bias
                direction = predominant_dir
            else:
                direction = random.choice(base_directions)
            
            cells['airflow_vector'][i] = [
                direction[0] * base_strength,
                direction[1] * base_strength
            ]
    
    def _place_starting_infrastructure(self, state: Dict[str, Any], grid_size: list):
        """Add minimal initial excavation and basic power systems"""
        cells = state['grid']['cells']
        
        # Place 2 starting excavated cells
        excavated_count = 0
        attempts = 0
        while excavated_count < 2 and attempts < 20:
            x = random.randint(0, grid_size[0]-1)
            y = random.randint(0, grid_size[1]-1)
            idx = y * grid_size[0] + x
            
            if not cells['excavated'][idx]:
                cells['excavated'][idx] = True
                cells['structure_type'][idx] = 'chamber'
                # Reduce stress when excavated
                cells['rock_stress'][idx] = max(0, cells['rock_stress'][idx] - 30)
                excavated_count += 1
            attempts += 1
        
        # Place 1 initial power conduit
        conduit_placed = False
        attempts = 0
        while not conduit_placed and attempts < 10:
            x = random.randint(0, grid_size[0]-1)
            y = random.randint(0, grid_size[1]-1)
            idx = y * grid_size[0] + x
            
            if cells['excavated'][idx]:  # Place on excavated cell
                cells['power_conduit'][idx] = True
                conduit_placed = True
            attempts += 1
    
    def _calculate_initial_physics(self, state: Dict[str, Any]):
        """Compute starting structural and airflow metrics"""
        cells = state['grid']['cells']
        
        # Calculate structural integrity based on stress distribution and excavations
        total_stress = sum(cells['rock_stress'])
        excavated_cells = sum(cells['excavated'])
        support_cells = sum(cells['has_support'])
        
        # Higher stress and more excavations reduce integrity
        # Supports increase integrity
        base_integrity = 100 - (total_stress / len(cells['rock_stress'])) * 0.3
        excavation_penalty = excavated_cells * 5
        support_bonus = support_cells * 10
        
        state['metrics']['structural_integrity'] = max(0, min(100, 
            base_integrity - excavation_penalty + support_bonus))
        
        # Calculate breathable air based on airflow and ventilation
        ventilation_shafts = sum(cells['ventilation_shaft'])
        airflow_strength = sum(
            abs(vector[0]) + abs(vector[1]) 
            for vector in cells['airflow_vector']
        ) / len(cells['airflow_vector'])
        
        base_air = 85 - excavated_cells * 2  # More space needs more air
        ventilation_bonus = ventilation_shafts * 15
        airflow_bonus = airflow_strength * 10
        
        state['metrics']['breathable_air_index'] = max(0, min(100,
            base_air + ventilation_bonus + airflow_bonus))
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = int(time.time())
        if seed is not None:
            return f"underground_{seed}_{timestamp}"
        return f"underground_{timestamp}_{random.randint(1000, 9999)}"