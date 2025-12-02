from base.env.base_env import BaseEnv, ObsEnv, SkinEnv
from env_obs import FullHiveObservation
from env_generate import HiveMindGenerator
import yaml
import os
import random
import statistics
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

class HiveMindEnv(SkinEnv):
    def __init__(self, env_id: str = "hive_mind_consensus"):
        obs_policy = FullHiveObservation()
        super().__init__(env_id, obs_policy)
        self.generator = None
    
    def _dsl_config(self):
        config_path = "./config.yaml"
        with open(config_path, 'r') as f:
            self.configs = yaml.safe_load(f)
        self.generator = HiveMindGenerator(self.env_id, self.configs)
    
    def reset(self, mode: str = "load", world_id: Optional[str] = None, seed: Optional[int] = None):
        if mode == "generate":
            world_id = self._generate_world(seed)
        elif mode == "load" and world_id is None:
            raise ValueError("world_id must be provided when mode is 'load'")
        
        self._state = self._load_world(world_id)
        self._t = 0
        self._history = []
        self._last_action_result = None
        return self._state
    
    def _load_world(self, world_id: str) -> Dict[str, Any]:
        world_path = f"./levels/{world_id}.yaml"
        with open(world_path, 'r') as f:
            world_state = yaml.safe_load(f)
        return world_state
    
    def _generate_world(self, seed: Optional[int] = None) -> str:
        return self.generator.generate(seed)
    
    def transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append(deepcopy(self._state))
        
        action_name = action.get('action')
        params = action.get('params', {})
        
        if action_name == 'MERGE':
            self._merge_streams(params.get('stream_a_id'), params.get('stream_b_id'))
        elif action_name == 'SPLIT':
            self._split_stream(params.get('stream_id'), params.get('split_ratio'))
        elif action_name == 'STIMULATE':
            self._stimulate_stream(params.get('stream_id'), params.get('energy_amount'))
        elif action_name == 'MEDITATE':
            self._meditate_stream(params.get('stream_id'), params.get('energy_amount'))
        elif action_name == 'ARCHIVE':
            self._archive_stream(params.get('stream_id'))
        elif action_name == 'REDISTRIBUTE_CE':
            self._redistribute_energy(params.get('energy_allocations'))
        
        self._update_knowledge_production()
        self._update_global_metrics()
        self._regenerate_energy()
        self._check_fragmentation()
        self._state['globals']['step_count'] += 1
        
        return self._state
    
    def _merge_streams(self, stream_a_id: int, stream_b_id: int):
        streams = self._state['sub_streams']
        stream_a = next((s for s in streams if s['id'] == stream_a_id), None)
        stream_b = next((s for s in streams if s['id'] == stream_b_id), None)
        
        if stream_a is None or stream_b is None or stream_a_id == stream_b_id:
            self._last_action_result = "Invalid stream IDs for merge"
            return
        
        total_size = stream_a['size'] + stream_b['size']
        merged_coherence = (stream_a['coherence'] * stream_a['size'] + stream_b['coherence'] * stream_b['size']) / total_size
        merged_novelty = (stream_a['novelty'] * stream_a['size'] + stream_b['novelty'] * stream_b['size']) / total_size
        
        stream_a['size'] = total_size
        stream_a['coherence'] = merged_coherence
        stream_a['novelty'] = merged_novelty
        
        streams.remove(stream_b)
        self._last_action_result = f"Merged streams {stream_a_id} and {stream_b_id}"
    
    def _split_stream(self, stream_id: int, split_ratio: float):
        if not (0.2 <= split_ratio <= 0.8):
            self._last_action_result = "Split ratio must be between 0.2 and 0.8"
            return
        
        streams = self._state['sub_streams']
        stream = next((s for s in streams if s['id'] == stream_id), None)
        
        if stream is None:
            self._last_action_result = "Stream not found"
            return
        
        new_id = max(s['id'] for s in streams) + 1
        
        stream_a_size = stream['size'] * split_ratio
        stream_b_size = stream['size'] * (1 - split_ratio)
        
        coherence_var = random.uniform(-10, 10)
        novelty_var = random.uniform(-10, 10)
        
        new_stream = {
            'id': new_id,
            'size': stream_b_size,
            'coherence': max(0, min(100, stream['coherence'] + coherence_var)),
            'novelty': max(0, min(100, stream['novelty'] + novelty_var)),
            'knowledge_per_turn': 0.0
        }
        
        stream['size'] = stream_a_size
        stream['coherence'] = max(0, min(100, stream['coherence'] - coherence_var))
        stream['novelty'] = max(0, min(100, stream['novelty'] - novelty_var))
        
        streams.append(new_stream)
        self._last_action_result = f"Split stream {stream_id} into {stream_id} and {new_id}"
    
    def _stimulate_stream(self, stream_id: int, energy_amount: int):
        if energy_amount > self._state['globals']['cognitive_energy']:
            self._last_action_result = "Insufficient cognitive energy"
            return
        
        stream = next((s for s in self._state['sub_streams'] if s['id'] == stream_id), None)
        if stream is None:
            self._last_action_result = "Stream not found"
            return
        
        novelty_boost = energy_amount * 0.5
        stream['novelty'] = min(100, stream['novelty'] + novelty_boost)
        self._state['globals']['cognitive_energy'] -= energy_amount
        self._last_action_result = f"Stimulated stream {stream_id} with {energy_amount} energy"
    
    def _meditate_stream(self, stream_id: int, energy_amount: int):
        if energy_amount > self._state['globals']['cognitive_energy']:
            self._last_action_result = "Insufficient cognitive energy"
            return
        
        stream = next((s for s in self._state['sub_streams'] if s['id'] == stream_id), None)
        if stream is None:
            self._last_action_result = "Stream not found"
            return
        
        coherence_boost = energy_amount * 0.5
        stream['coherence'] = min(100, stream['coherence'] + coherence_boost)
        self._state['globals']['cognitive_energy'] -= energy_amount
        self._last_action_result = f"Meditated stream {stream_id} with {energy_amount} energy"
    
    def _archive_stream(self, stream_id: int):
        stream = next((s for s in self._state['sub_streams'] if s['id'] == stream_id), None)
        if stream is None:
            self._last_action_result = "Stream not found"
            return
        
        if stream['novelty'] <= 0:
            self._last_action_result = "Stream has no novelty to archive"
            return
        
        archived_knowledge = stream['novelty'] * stream['size'] / 10
        self._state['globals']['knowledge_score'] += int(archived_knowledge)
        stream['novelty'] = max(0, stream['novelty'] - 20)
        self._last_action_result = f"Archived {int(archived_knowledge)} knowledge from stream {stream_id}"
    
    def _redistribute_energy(self, energy_allocations: Dict[str, int]):
        total_allocation = sum(energy_allocations.values())
        available_energy = self._state['globals']['cognitive_energy']
        
        if total_allocation != available_energy:
            self._last_action_result = f"Total allocation ({total_allocation}) must equal available energy ({available_energy})"
            return
        
        self._last_action_result = f"Redistributed {total_allocation} cognitive energy"
    
    def _update_knowledge_production(self):
        for stream in self._state['sub_streams']:
            knowledge_production = stream['size'] * (stream['coherence'] + stream['novelty']) / 200
            stream['knowledge_per_turn'] = knowledge_production
            self._state['globals']['knowledge_score'] += knowledge_production
    
    def _update_global_metrics(self):
        streams = self._state['sub_streams']
        if not streams:
            self._state['globals']['unity'] = 0
            self._state['globals']['diversity'] = 0
            return
        
        total_size = sum(s['size'] for s in streams)
        unity = sum(s['coherence'] * s['size'] for s in streams) / total_size if total_size > 0 else 0
        
        novelties = [s['novelty'] for s in streams]
        if len(novelties) > 1:
            novelty_variance = statistics.variance(novelties)
            diversity = max(0, 100 - novelty_variance)
        else:
            diversity = 0
        
        self._state['globals']['unity'] = unity
        self._state['globals']['diversity'] = diversity
    
    def _regenerate_energy(self):
        self._state['globals']['cognitive_energy'] = 100
    
    def _check_fragmentation(self):
        if self._state['globals']['unity'] < 40 and len(self._state['sub_streams']) > 1:
            if random.random() < 0.3:
                weakest_stream = min(self._state['sub_streams'], key=lambda s: s['coherence'])
                self._state['sub_streams'].remove(weakest_stream)
                knowledge_loss = weakest_stream['size'] * 2
                self._state['globals']['knowledge_score'] = max(0, self._state['globals']['knowledge_score'] - knowledge_loss)
                self._last_action_result = f"Fragmentation: Lost stream {weakest_stream['id']} and {knowledge_loss} knowledge"
    
    def reward(self, action: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, Any]]:
        events = []
        reward_info = {}
        total_reward = 0.0
        
        unity = self._state['globals']['unity']
        diversity = self._state['globals']['diversity']
        knowledge_score = self._state['globals']['knowledge_score']
        
        continuous_reward = 0.05 * unity + 0.05 * diversity
        if len(self._history) > 0:
            prev_knowledge = self._history[-1]['globals']['knowledge_score']
            knowledge_gain = knowledge_score - prev_knowledge
            continuous_reward += 0.2 * knowledge_gain
        
        total_reward += continuous_reward
        reward_info['continuous'] = continuous_reward
        
        milestones = [50, 100, 150, 200]
        for milestone in milestones:
            milestone_key = f'knowledge_{milestone}'
            if knowledge_score >= milestone and not self._state['milestones'][milestone_key]:
                total_reward += 10.0
                events.append(f'Knowledge milestone: {milestone}')
                reward_info['milestone'] = 10.0
                self._state['milestones'][milestone_key] = True
        
        if unity > 80 and diversity > 60:
            total_reward += 20.0
            events.append('Synergy bonus achieved')
            reward_info['synergy'] = 20.0
        
        if knowledge_score >= 200:
            total_reward += 50.0
            events.append('Target knowledge reached')
            reward_info['completion'] = 50.0
        
        return total_reward, events, reward_info
    
    def done(self, state=None) -> bool:
        unity = self._state['globals']['unity']
        diversity = self._state['globals']['diversity']
        max_steps = self.configs['termination']['max_steps']
        
        return (self._state['globals']['step_count'] >= max_steps or 
                unity < 20.0 or 
                diversity < 20.0)
    
    def observe_semantic(self) -> Dict[str, Any]:
        return self.obs_policy(self._state, self._t)
    
    def render_skin(self, omega: Dict[str, Any]) -> str:
        # Extract data from semantic observation
        streams = omega.get('sub_streams', [])
        globals_data = omega.get('globals', {})
        # Display step starts from 1 (visual only)
        step_display = int(globals_data.get('step_count', 0)) + 1
        
        # Build streams display
        streams_display = ""
        if streams:
            streams_display = "\n".join([
                f"  Stream {s['id']}: Size={s['size']:.1f} | Coherence={s['coherence']:.1f} | Novelty={s['novelty']:.1f} | Knowledge/turn={s['knowledge_per_turn']:.2f}"
                for s in sorted(streams, key=lambda x: x['id'])
            ])
        else:
            streams_display = "  No active streams"
        
        # Build milestones display
        milestones = omega.get('milestones', {})
        milestones_status = []
        for milestone in ['knowledge_50', 'knowledge_100', 'knowledge_150', 'knowledge_200']:
            achieved = "✓" if milestones.get(milestone, False) else "○"
            target = milestone.split('_')[1]
            milestones_status.append(f"{achieved} {target}")
        
        template = """=== COLLECTIVE CONSCIOUSNESS NEXUS - STEP {step_display}/{max_steps} ===

CONSCIOUSNESS STREAMS:
{streams_display}

GLOBAL METRICS:
Unity Index: {unity:.1f}% | Diversity Index: {diversity:.1f}%
Knowledge Score: {knowledge_score:.1f} | Cognitive Energy: {cognitive_energy}

MILESTONES: {milestones_status}

LAST ACTION: {last_action_result}

AVAILABLE ACTIONS:
- MERGE(stream_a_id, stream_b_id) - Combine two consciousness streams
- SPLIT(stream_id, split_ratio) - Divide stream (ratio: 0.2-0.8)
- STIMULATE(stream_id, energy_amount) - Boost novelty with cognitive energy
- MEDITATE(stream_id, energy_amount) - Enhance coherence with cognitive energy  
- ARCHIVE(stream_id) - Convert novelty to permanent knowledge
- REDISTRIBUTE_CE(energy_allocations) - Reallocate cognitive energy

WARNING: Unity < 40% or Diversity < 20% may cause termination!"""
        
        return template.format(
            step_display=step_display,
            max_steps=globals_data.get('max_steps', 50),
            streams_display=streams_display,
            unity=globals_data.get('unity', 0),
            diversity=globals_data.get('diversity', 0),
            knowledge_score=globals_data.get('knowledge_score', 0),
            cognitive_energy=globals_data.get('cognitive_energy', 100),
            milestones_status=" | ".join(milestones_status),
            last_action_result=omega.get('last_action_result', 'None')
        )
