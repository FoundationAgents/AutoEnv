from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import FullTimelineAccessPolicy
from env_generate import CrimeWorldGenerator
import yaml
import os
from typing import Dict, Any, Optional, Tuple, List

class BackwardsInvestigationEnv(SkinEnv):
    def __init__(self, env_id: int):
        obs_policy = FullTimelineAccessPolicy(show_future_events=False, show_unvalidated_connections=True)
        super().__init__(env_id, obs_policy)
        
    def _dsl_config(self):
        config_path = f"./config.yaml"
        try:
            with open(config_path, 'r') as f:
                self.configs = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
            world_state = self._load_world(world_id)
        elif mode == "load":
            if world_id is None:
                raise ValueError("world_id must be provided when mode='load'")
            world_state = self._load_world(world_id)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'load' or 'generate'")
        
        self._state = world_state
        self._t = 0
        self._history = []
        self._last_action_result = None
        
        # Initialize agent at time 40
        if 'agent' not in self._state:
            self._state['agent'] = {}
        self._state['agent']['current_time_index'] = 40
        self._state['agent']['investigation_status'] = 'active'
        
        # Initialize investigation tracking
        if 'investigation' not in self._state:
            self._state['investigation'] = {}
        self._state['investigation']['timeline_ledger'] = {}
        self._state['investigation']['proposed_connections'] = []
        
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        levels_dir = "./levels/"
        file_path = os.path.join(levels_dir, f"{world_id}.yaml")
        
        try:
            with open(file_path, 'r') as f:
                world_state = yaml.safe_load(f)
            return world_state
        except FileNotFoundError:
            raise FileNotFoundError(f"World file not found: {file_path}")
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        generator = CrimeWorldGenerator(self.env_id, self.configs)
        return generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(self._state.copy())
        
        action_name = action.get('action')
        params = action.get('params', {})
        
        current_time = self._state['agent']['current_time_index']
        
        if action_name == "ExamineScene":
            location = params.get('location')
            result = self._examine_scene(location, current_time - 1)
            self._state['agent']['current_time_index'] -= 1
            self._last_action_result = f"Examined {location}: {result}"
            
        elif action_name == "InterrogatePerson":
            person = params.get('person')
            result = self._interrogate_person(person, current_time - 1)
            self._state['agent']['current_time_index'] -= 1
            self._last_action_result = f"Interrogated {person}: {result}"
            
        elif action_name == "TraceObject":
            obj = params.get('object')
            result = self._trace_object(obj, current_time - 1)
            self._state['agent']['current_time_index'] -= 1
            self._last_action_result = f"Traced {obj}: {result}"
            
        elif action_name == "ConnectClues":
            clue_a = params.get('clueA')
            clue_b = params.get('clueB')
            result = self._connect_clues(clue_a, clue_b)
            self._last_action_result = f"Connection attempt: {result}"
            
        elif action_name == "JumpEarlier":
            steps = params.get('steps', 1)
            self._state['agent']['current_time_index'] -= min(int(steps), int(current_time))
            self._last_action_result = f"Jumped {steps} time steps earlier"
            
        elif action_name == "IdentifyRootCause":
            time_index = params.get('time_index')
            perpetrator = params.get('perpetrator')
            action_performed = params.get('action')
            self._state['agent']['investigation_status'] = 'completed'
            self._state['agent']['final_identification'] = {
                'time_index': time_index,
                'perpetrator': perpetrator,
                'action': action_performed
            }
            self._last_action_result = f"Identified root cause: {perpetrator} performed {action_performed} at time {time_index}"
        
        return self._state
    
    def _examine_scene(self, location: str, time_index: int) -> str:
        # Find clues available at this location and time
        essential_clues = self._state.get('ground_truth', {}).get('essential_clues', [])
        timeline_events = self._state.get('timeline', {}).get('events', {})
        
        found_info = []
        
        # Check for clues at this location and time
        for clue in essential_clues:
            if (clue.get('location') == location and 
                clue.get('time_available', 0) <= time_index and
                clue['id'] not in [c['id'] for c in self._state['investigation'].get('collected_clues', [])]):
                
                self._state['investigation'].setdefault('collected_clues', []).append(clue)
                found_info.append(f"Found {clue['description']}")
        
        # Check for events at this time
        if str(time_index) in timeline_events:
            event = timeline_events[str(time_index)]
            self._state['investigation']['timeline_ledger'][str(time_index)] = event
            found_info.append(f"Discovered event: {event['event']}")
        
        return "; ".join(found_info) if found_info else "No new information found"
    
    def _interrogate_person(self, person: str, time_index: int) -> str:
        timeline_events = self._state.get('timeline', {}).get('events', {})
        found_info = []
        
        # Look for events involving this person around this time
        for time_str, event in timeline_events.items():
            event_time = int(time_str)
            if (abs(event_time - time_index) <= 2 and 
                event.get('perpetrator') == person and
                time_str not in self._state['investigation']['timeline_ledger']):
                
                self._state['investigation']['timeline_ledger'][time_str] = event
                found_info.append(f"{person} was involved in: {event['event']}")
        
        return "; ".join(found_info) if found_info else f"{person} provided no useful information for this time period"
    
    def _trace_object(self, obj: str, time_index: int) -> str:
        essential_clues = self._state.get('ground_truth', {}).get('essential_clues', [])
        found_info = []
        
        for clue in essential_clues:
            if (clue.get('object') == obj and 
                clue.get('time_available', 0) <= time_index and
                clue['id'] not in [c['id'] for c in self._state['investigation'].get('collected_clues', [])]):
                
                self._state['investigation'].setdefault('collected_clues', []).append(clue)
                found_info.append(f"Traced {obj}: {clue['description']}")
        
        return "; ".join(found_info) if found_info else f"No trace information found for {obj}"
    
    def _connect_clues(self, clue_a: str, clue_b: str) -> str:
        collected_clues = self._state['investigation'].get('collected_clues', [])
        
        # Find the clues
        clue_a_obj = next((c for c in collected_clues if c['id'] == clue_a), None)
        clue_b_obj = next((c for c in collected_clues if c['id'] == clue_b), None)
        
        if not clue_a_obj or not clue_b_obj:
            return "One or both clues not found in collected evidence"
        
        # Simple validation logic - in a real implementation this would be more sophisticated
        if (clue_a_obj.get('type') == clue_b_obj.get('type') or 
            clue_a_obj.get('location') == clue_b_obj.get('location')):
            
            connection = f"Validated connection between {clue_a} and {clue_b}"
            self._state['timeline'].setdefault('validated_connections', {})[f"{clue_a}-{clue_b}"] = True
            
            # Remove from unresolved effects if relevant
            unresolved = self._state['investigation'].get('unresolved_effects', [])
            if unresolved:
                self._state['investigation']['unresolved_effects'] = unresolved[1:]
            
            return connection
        else:
            return f"No valid connection found between {clue_a} and {clue_b}"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        
        if action.get('action') == 'IdentifyRootCause':
            ground_truth = self._state.get('ground_truth', {})
            identification = self._state.get('agent', {}).get('final_identification', {})
            
            correct_time = identification.get('time_index') == ground_truth.get('root_cause_time')
            correct_perpetrator = identification.get('perpetrator') == ground_truth.get('root_cause_perpetrator')
            correct_action = identification.get('action') == ground_truth.get('root_cause_action')
            
            if correct_time and correct_perpetrator and correct_action:
                events.append("correct_identification")
                reward_info['identification_result'] = 'correct_full'
                return 1.0, events, reward_info
            else:
                events.append("incorrect_identification")
                reward_info['identification_result'] = 'incorrect_full'
                return 0.0, events, reward_info
        
        # Check for timeout
        if self._state['agent']['current_time_index'] <= 0:
            events.append("timeout_failure")
            reward_info['identification_result'] = 'timeout'
            return 0.0, events, reward_info
        
        return 0.0, events, reward_info
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        current_time = omega.get('current_time_index', 40)
        max_steps = omega.get('max_steps', 40)
        t = omega.get('t', 0)
        
        # Format timeline ledger
        timeline_ledger = omega.get('timeline_ledger', {})
        timeline_display = []
        for time_idx in sorted(timeline_ledger.keys(), key=int, reverse=True):
            event = timeline_ledger[time_idx]
            timeline_display.append(f"  Time {time_idx}: {event.get('event', 'Unknown event')}")
        timeline_ledger_display = "\n".join(timeline_display) if timeline_display else "  No events discovered yet"
        
        # Format unresolved effects
        unresolved_effects = omega.get('unresolved_effects', [])
        unresolved_effects_list = "\n".join([f"  - {effect}" for effect in unresolved_effects]) if unresolved_effects else "  None"
        
        # Format clues inventory
        collected_clues = omega.get('collected_clues', [])
        clues_inventory = "\n".join([f"  {clue['id']}: {clue['description']}" for clue in collected_clues]) if collected_clues else "  No clues collected"
        
        # Format suspect status
        suspect_list = omega.get('suspect_list', [])
        suspect_status = "\n".join([f"  {suspect['name']}: {suspect['involvement_level']}" for suspect in suspect_list]) if suspect_list else "  No suspects identified"
        
        # Last action result
        last_action_result = self._last_action_result or "No actions taken yet"
        
        template = f"""=== BACKWARDS INVESTIGATION - TIME INDEX {current_time} ===
Step {t}/{max_steps}

TIMELINE LEDGER:
{timeline_ledger_display}

UNRESOLVED EFFECTS:
{unresolved_effects_list}

COLLECTED CLUES:
{clues_inventory}

CURRENT SUSPECTS:
{suspect_status}

RECENT ACTION RESULT:
{last_action_result}

Available Actions: ExamineScene(location), InterrogatePerson(person), 
TraceObject(object), ConnectClues(clueA,clueB), JumpEarlier(steps),
IdentifyRootCause(time_index,perpetrator,action)"""
        
        return template
    
    def done(self, state=None) -> bool:
        investigation_status = self._state.get('agent', {}).get('investigation_status', 'active')
        current_time = self._state.get('agent', {}).get('current_time_index', 40)
        
        return (investigation_status == 'completed' or 
                current_time <= 0 or 
                self._t >= self.configs["termination"]["max_steps"])