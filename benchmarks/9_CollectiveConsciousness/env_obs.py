from base.env.base_observation import ObservationPolicy
from typing import Dict, Any
import statistics

class FullHiveObservation(ObservationPolicy):
    def __call__(self, env_state: Dict[str, Any], t: int) -> Dict[str, Any]:
        obs = {
            'unity': round(env_state['globals']['unity'], 1),
            'diversity': round(env_state['globals']['diversity'], 1),
            'knowledge_score': env_state['globals']['knowledge_score'],
            'cognitive_energy': env_state['globals']['cognitive_energy'],
            'step_count': env_state['globals']['step_count'],
            't': t,
            'sub_streams': [],
            'milestones': env_state['milestones'].copy(),
            'warnings': {
                'unity_critical': env_state['globals']['unity'] < 50.0,
                'diversity_critical': env_state['globals']['diversity'] < 35.0
            }
        }
        
        for stream in env_state['sub_streams']:
            stream_obs = {
                'id': stream['id'],
                'size': round(stream['size'], 1),
                'coherence': round(stream['coherence'], 1),
                'novelty': round(stream['novelty'], 1),
                'knowledge_per_turn': round(stream['knowledge_per_turn'], 2)
            }
            obs['sub_streams'].append(stream_obs)
        
        obs['knowledge_to_milestone'] = 0
        milestones = [50, 100, 150, 200]
        for milestone in milestones:
            if obs['knowledge_score'] < milestone:
                obs['knowledge_to_milestone'] = milestone - obs['knowledge_score']
                break
        
        return obs