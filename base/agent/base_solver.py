# this is euqal to solver.py
import json
import time
from pydantic import Field
from typing import Dict, Optional, List, Any, Tuple

from base.agent.base_agent import BaseAgent
from base.engine.utils import parse_llm_action_response, parse_xml_content, collect_trajectory
from base.engine.logs import logger as _logger
from base.env.base_env import SkinEnv
from base.engine.logs import logger

# Maximum number of past actions to retain; if None, retain all past actions
MAX_PAST_ACTIONS = None

AGENT_ACT_PROMPT = """
==== Environment Instruction ====
{env_instruction}

==== Action Space ====
{action_space}

==== Output Format ====

==== thinking output format ====
Before outputting an action, you should think step by step, and write any necessary reasoning (such as environment rules or information relevant for future actions) inside the <thinking_memory></thinking_memory> tag.

==== Action Output Format ====
When you output the action, 
you should output the action name and parameters in the json format, and only one action.
Such as, 
```json
{{
    "action": "",
    "params": {{
        "<param_name>": "<param_value>"
    }}
}}
```

The thinking and action should be outputted separately:
- First, write your reasoning inside <thinking_memory></thinking_memory> tag
- Then, output the action directly in ```json``` code block (no additional tags needed)

==== Past Actions ====
Your recent actions are:
{recent_actions}

==== Now, your observation is:====
{obs}
"""

class SolverAgent(BaseAgent):
    """
    Solver Agent is used to recive different environment desc and action space desc.
    And then solve the environment.
    """
    name: str = Field(default="sovler")
    description: str = Field(default="A solver agent for solving the environment.")
    current_environment_total_reward: int = Field(default=0)
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    past_actions: List[Dict[str, Any]] = Field(default_factory=list)
    trajectory_folder_path: str = Field(default="")
    
    def parse_action(self, resp: str):
        """Parse LLM response to extract action data."""
        return parse_llm_action_response(resp)

    def _get_recent_actions(self):
        lines = []

        actions_to_show = self.past_actions if MAX_PAST_ACTIONS is None else self.past_actions[-MAX_PAST_ACTIONS:]
        for i, entry in enumerate(actions_to_show):
            if isinstance(entry, dict) and "action" in entry:
                thought_part = entry.get("thought")
                action_part = entry.get("action")
                result_part = entry.get("result")
                # Compact result representation
                if isinstance(result_part, (dict, list)):
                    try:
                        result_str = json.dumps(result_part, ensure_ascii=False)
                    except Exception:
                        result_str = str(result_part)
                else:
                    result_str = str(result_part)
                lines.append(f"{i+1}. {thought_part} \n {action_part} -> {result_str}")
            else:
                lines.append(f"{i+1}. {entry}")
        return "\n".join(lines)

    async def step(self, agent_obs:Dict) -> Tuple[Dict, str]:
        act_prompt = AGENT_ACT_PROMPT.format(
            env_instruction = self.current_env_instruction,
            action_space = self.current_action_space,
            obs = agent_obs,
            recent_actions = self._get_recent_actions()
        )
        
        try:
            resp = await self.llm(act_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = None
            
        action = self.parse_action(resp)
        thought = None

        # First try to extract thinking_memory tag content
        if resp:
            thinking_memory_content = parse_xml_content(resp, "thinking_memory")
            if thinking_memory_content.get("thinking_memory"):
                thought = thinking_memory_content["thinking_memory"]
            # Fallback to original extraction logic if no thinking_memory tag found
            elif '```json' in resp:
                thought = resp.split('```json')[0].strip()
            elif '```' in resp:
                thought = resp.split('```')[0].strip()
            else:
                thought = resp.strip()
        else:
            thought = "No response from LLM"
        
        if thought:
            logger.agent_thinking(f"Agent Thought: {thought}")

        logger.agent_action(f"Agent Action: {action}")

        return action, thought
    
    def _apply_prompt(self, env_instruction: str, action_space: str):
        self.current_env_instruction = env_instruction
        self.current_action_space = action_space

    def _resolve_max_steps(self, env: SkinEnv, env_info: Dict) -> int:
        """
        Resolve step limit with precedence:
        1) If env_info provides max_step (not None), use it.
        2) Else, fallback to env.configs termination.max_steps.
        3) Else, default to 20.
        """
        # 1) Explicit override from caller
        if "max_step" in env_info and env_info.get("max_step") is not None:
            try:
                return int(env_info.get("max_step"))
            except (TypeError, ValueError):
                pass

        # 2) Config fallbacks
        configs = getattr(env, "configs", {}) if getattr(env, "configs", None) else {}
        term_steps = configs.get("termination", {}).get("max_steps")
        if isinstance(term_steps, int):
            return term_steps
        try:
            if term_steps is not None:
                return int(term_steps)
        except (TypeError, ValueError):
            pass

        # 3) Final default
        return 20

    @collect_trajectory(save_dir=None)
    async def run(self, env:SkinEnv, env_info:Dict):
        """
        env info:
        {
            "world_id": str,
            "agent_instruction": str,
            "action_space": str,
            "max_step": int,
        }
        """
        world_id = env_info["world_id"]
        self.past_actions = []
        world_id = env_info["world_id"]
        self._apply_prompt(env_info["agent_instruction"], env_info["action_space"])
        env.reset(mode="load", world_id=world_id)
        # Resolve step limit with override-first precedence
        max_step = self._resolve_max_steps(env, env_info)
        cur_steps = 0
        cur_reward = 0
        events_count = {}
        
        raw_obs = env.observe_semantic()
        agent_obs = env.render_skin(raw_obs)
        initial_observation = agent_obs

        while cur_steps < max_step and not env.done():
            logger.info(f"Environment Observation: \n{agent_obs}")
            # Retry up to 3 times for invalid actions without consuming a step
            retries = 0
            action = None
            thought = None
            while True:
                action, thought = await self.step(agent_obs)
                act_name = None
                try:
                    act_name = (action or {}).get("action")
                except Exception:
                    act_name = None
                is_invalid = act_name in {None, "", "Invalid", "no_action"}
                if not is_invalid:
                    break
                retries += 1
                if retries >= 3:
                    _logger.warning("Invalid action after 3 retries; proceeding with invalid action.")
                    break
                else:
                    _logger.warning(f"Invalid action, retrying ({retries}/3) without consuming step...")
                    continue
            _, reward, done, info = env.step(action)
            # Record action along with last action result for better context
            self.past_actions.append({
                "action": action,
                "thought": thought,
                "observation": agent_obs,
                "result": info.get("last_action_result"),
                "events": info.get("events", []),
                "reward": reward,
                "parse_error": (action or {}).get("_parse_error"),
            })
            cur_reward += reward
            agent_obs = info["skinned"]
            for e in info.get("events", []):
                events_count[e] = events_count.get(e, 0) + 1
            cur_steps += 1
            if done:
                break

        return {
            "total_reward": cur_reward,
            "events_count": events_count,
            "step": cur_steps,
            "initial_observation": initial_observation,
        }
        
