from base.env.base_generator import WorldGenerator
import random
import yaml
import os
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

class CrimeWorldGenerator(WorldGenerator):
    def __init__(self, env_id: str, config: Dict[str, Any]):
        super().__init__(env_id, config)
        
    def generate(self, seed: Optional[int] = None, save_path: Optional[str] = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        # Initialize from state template
        base_state = self.config.get('state_template', {})
        
        # Execute generation pipeline
        world_state = self._execute_pipeline(base_state, seed)
        
        # Generate world ID and save
        world_id = self._generate_world_id(seed)
        self._save_world(world_state, world_id)
        
        return world_id
    
    def _execute_pipeline(self, base_state: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        world_state = self._deep_copy_state(base_state)
        
        # Step 1: Initialize from template (already done)
        
        # Step 2: Generate crime scenario
        perpetrators = [f"person_{i}" for i in range(1, random.randint(4, 6) + 1)]
        root_time = random.randint(5, 15)
        root_perpetrator = random.choice(perpetrators)
        root_actions = ["theft", "sabotage", "fraud", "assault", "vandalism"]
        root_action = random.choice(root_actions)
        
        world_state['ground_truth']['root_cause_time'] = root_time
        world_state['ground_truth']['root_cause_perpetrator'] = root_perpetrator
        world_state['ground_truth']['root_cause_action'] = root_action
        
        # Step 3: Build causal chain
        chain_length = random.randint(6, 8)
        causal_events = {}
        
        # Create root event (use string keys for consistency)
        causal_events[str(root_time)] = {
            'event': f"{root_perpetrator} performs {root_action}",
            'perpetrator': root_perpetrator,
            'action': root_action,
            'type': 'root_cause'
        }
        
        # Create chain of consequences
        current_time = root_time
        for i in range(chain_length - 1):
            current_time += random.randint(2, 5)
            if current_time >= 40:
                current_time = 39
            
            consequence_types = ["discovery", "escalation", "cover_attempt", "evidence_creation"]
            event_type = random.choice(consequence_types)
            involved_person = random.choice(perpetrators)
            
            causal_events[str(current_time)] = {
                'event': f"{event_type} by {involved_person} at time {current_time}",
                'perpetrator': involved_person,
                'type': event_type,
                'caused_by': current_time - random.randint(2, 5)
            }
        
        world_state['timeline']['events'] = causal_events
        
        # Step 4: Distribute clues
        essential_clues = []
        locations = ["office", "warehouse", "parking_lot", "reception", "storage_room"]
        objects = ["document", "key", "tool", "computer", "phone"]
        
        for i in range(random.randint(6, 8)):
            clue = {
                'id': f"clue_{i+1}",
                'type': random.choice(["physical", "digital", "witness", "forensic"]),
                'location': random.choice(locations),
                'object': random.choice(objects),
                'time_available': random.randint(root_time + 1, 35),
                'relevance': "essential",
                'description': f"Evidence piece {i+1} found at {random.choice(locations)}"
            }
            essential_clues.append(clue)
        
        # Add decoy clues
        for i in range(random.randint(2, 3)):
            decoy_clue = {
                'id': f"decoy_{i+1}",
                'type': random.choice(["physical", "digital", "witness"]),
                'location': random.choice(locations),
                'object': random.choice(objects),
                'time_available': random.randint(20, 35),
                'relevance': "decoy",
                'description': f"Misleading evidence {i+1}"
            }
            essential_clues.append(decoy_clue)
        
        world_state['ground_truth']['essential_clues'] = essential_clues
        
        # Step 5: Setup crime scene - ENSURE ROOT PERPETRATOR IS IN SUSPECT LIST
        # First add the root perpetrator, then add others
        initial_suspects = [root_perpetrator]  # Start with root perpetrator
        other_suspects = [p for p in perpetrators if p != root_perpetrator]
        
        # Add 3-5 more suspects (including some from perpetrators list)
        needed_suspects = random.randint(3, 5)
        if len(other_suspects) >= needed_suspects:
            initial_suspects.extend(random.sample(other_suspects, needed_suspects))
        else:
            initial_suspects.extend(other_suspects)
            # Add additional random suspects if needed
            for i in range(needed_suspects - len(other_suspects)):
                initial_suspects.append(f"person_{len(perpetrators) + i + 1}")
        
        world_state['crime_scene']['suspect_list'] = [
            {
                'name': suspect,
                'involvement_level': 'unknown',
                'evidence_against': [],
                'alibi_status': 'unverified'
            } for suspect in initial_suspects
        ]
        
        # Create unresolved effects
        unresolved_effects = [
            "Unexplained access to secured area",
            "Missing security footage from critical time period", 
            "Contradictory witness statements about suspect locations"
        ]
        world_state['investigation']['unresolved_effects'] = random.sample(unresolved_effects, random.randint(2, 3))
        
        # Initial evidence at crime scene
        world_state['crime_scene']['initial_evidence'] = [
            "Crime scene photographs",
            "Initial police report",
            "Witness contact information"
        ]
        
        # Step 6: Validate solvability
        world_state['ground_truth']['decoy_suspects'] = [s for s in initial_suspects if s != root_perpetrator]
        
        return world_state
    
    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(state)
    
    def _save_world(self, world_state: Dict[str, Any], world_id: str) -> str:
        levels_dir = "./levels/"
        os.makedirs(levels_dir, exist_ok=True)
        
        file_path = os.path.join(levels_dir, f"{world_id}.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(world_state, f, default_flow_style=False)
        
        return world_id
    
    def _generate_world_id(self, seed: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if seed is not None:
            return f"crime_world_seed_{seed}_{timestamp}"
        else:
            return f"crime_world_{uuid.uuid4().hex[:8]}_{timestamp}"
