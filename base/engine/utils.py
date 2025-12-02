
"""
@Time    : 2025-06-06
@Author  : didi & Zhaoyang
"""
import os
import re
import json
import types
import inspect

from typing import Any, Awaitable, Callable, Dict, Optional, List

from base.engine.logs import logger
from base.engine.trajectory import TrajectoryCollector, Trajectory

def parse_xml_content(content: str, tag: str) -> dict:
    """
    Parse the given content string and extract all occurrences of the specified XML tag.

    Args:
        content (str): The string containing XML-like data.
        tag (str): The tag name to search for.

    Returns:
        dict: A dictionary with the tag as key and a list of extracted values as value.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    # If only one match, return as string, else as list
    if not matches:
        return {tag: None}
    elif len(matches) == 1:
        return {tag: matches[0].strip()}
    else:
        return {tag: [m.strip() for m in matches]}

def read_file_content(file_path):
    """
    Read the entire content of a Python or YAML file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    

def write_file_content(file_path, content):
    """
    Write the given content to a file, overwriting if it exists.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.

    Returns:
        None
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def get_env_paths(base_path: str) -> List[str]:
    env_paths = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.startswith("env_") and os.path.isdir(os.path.join(base_path, item)):
                env_paths.append(os.path.join(base_path, item))
    return env_paths


def archive_files(env_folder_path: str, env_id: str = None) -> bool:
    """
    Clean up environment directory by archiving auxiliary files.
    Keeps only core environment files in the root directory.
    
    Args:
        env_folder_path (str): Path to the environment folder
        env_id (str, optional): Environment ID for logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not env_folder_path:
        raise ValueError("env_folder_path cannot be empty")
    
    import subprocess
    import sys
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get the path to the archive script
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    archive_script = os.path.join(project_root, "scripts", "run_archive_files.py")
    
    if env_id:
        logger.info(f"Archiving auxiliary files for environment: {env_id}")
    logger.info(f"Environment folder: {env_folder_path}")
    
    try:
        # Run the archive script
        result = subprocess.run(
            [sys.executable, archive_script, env_folder_path],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            logger.info("Directory cleanup completed successfully")
            logger.info(f"Archive output: {result.stdout}")
            
            # Create done.txt file to mark completion
            done_file_path = os.path.join(env_folder_path, "done.txt")
            write_file_content(done_file_path, "")
            logger.info(f"Created done.txt file: {done_file_path}")
            
            return True
        else:
            logger.error(f"Archive script failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running archive script: {e}")
        return False


def parse_llm_action_response(resp: str) -> Dict[str, Any]:
    """Parse LLM response to extract action data.
    
    This function handles various LLM response formats:
    - JSON wrapped in ```json``` blocks
    - JSON wrapped in ``` blocks  
    - Raw JSON strings
    - List responses (takes first action)
    - Malformed responses (returns default action)
    
    Args:
        resp: Raw LLM response string
        
    Returns:
        Dict containing action data with 'action' and 'params' keys
    """
    try:
        # Check if response is None or empty
        if not resp:
            logger.warning("Received None or empty response from LLM")
            return {"action": "no_action", "params": {}, "_parse_error": "Empty LLM response"}
        
        # Extract JSON content from response
        start_idx = resp.find('```json')
        if start_idx != -1:
            start_idx += 7  # Skip '```json'
            end_idx = resp.find('```', start_idx)
            if end_idx != -1:
                json_str = resp[start_idx:end_idx].strip()
            else:
                json_str = resp[start_idx:].strip()
        else:
            # Fallback: try to find JSON content within ```
            start_idx = resp.find('```')
            if start_idx != -1:
                start_idx += 3  # Skip '```'
                end_idx = resp.find('```', start_idx)
                if end_idx != -1:
                    json_str = resp[start_idx:end_idx].strip()
                else:
                    json_str = resp[start_idx:].strip()
            else:
                # Final fallback: try to find JSON-like content
                json_str = resp.strip()
        
        try:
            action_data = json.loads(json_str)
        except Exception as e:
            # JSON parsing failed; include error detail for trajectory consumers
            logger.warning(f"Failed to parse action JSON '{resp}': {e}. Using default action.")
            return {
                "action": "Invalid",
                "params": {},
                "_parse_error": f"{type(e).__name__}: {e}",
            }
        
        # Handle case where LLM returns a list instead of single action
        if isinstance(action_data, list):
            if len(action_data) > 0:
                logger.warning("LLM returned a list of actions; taking the first entry")
                action_data = action_data[0]  # Take the first action
            else:
                logger.warning("LLM returned an empty list, using default action.")
                return {"action": "Invalid", "params": {}, "_parse_error": "Empty list returned by LLM"}
        
        # Ensure action_data has required structure
        if not isinstance(action_data, dict) or "action" not in action_data:
            logger.warning(f"Invalid action format: {action_data}. Using default action.")
            return {"action": "Invalid", "params": {}, "_parse_error": "Missing 'action' key or invalid dict"}
            
        return action_data
    except Exception as e:
        logger.warning(f"Unexpected error while parsing action: {e}. Using default action.")
        return {"action": "Invalid", "params": {}, "_parse_error": f"{type(e).__name__}: {e}"}


def collect_trajectory(
    *,
    save_dir: Optional[str] = None,
    save_jsonl: bool = False,
    on_finish: Optional[Callable[[Trajectory], None]] = None,
):
    """Decorator for an agent's async run(env, env_info, ...) method to collect trajectory.

    This will record pairs of (obs provided to self.step, action returned), and the
    subsequent env.step(action) result (reward, last_action_result, events) without
    changing the original method's logic.
    """

    def decorator(func: Callable[..., Awaitable[Any]]):
        async def wrapper(self, env, env_info: Dict, *args, **kwargs):
            # Resolve save_dir dynamically
            if callable(save_dir):
                resolved_save_dir = save_dir(self)
            elif save_dir is None and hasattr(self, 'trajectory_folder_path'):
                resolved_save_dir = self.trajectory_folder_path
            else:
                resolved_save_dir = save_dir
            collector = TrajectoryCollector(save_dir=resolved_save_dir, save_jsonl=save_jsonl)
            
            metadata = {
                "agent_name": getattr(self, "name", self.__class__.__name__),
                "world_id": env_info.get("world_id"),
                "env_desc_hash": hash(env_info.get("agent_instruction", "")),
                "action_space_hash": hash(env_info.get("action_space", "")),
            }
            try:
                collector.start_run(metadata)
            except Exception:
                pass

            last: Dict[str, Any] = {"step_index": 0}
            # Save original
            original_env_step = env.step

            def wrapped_env_step(env_bound, action: Dict, *a, **k):
                result = original_env_step(action, *a, **k)
                try:
                    _, reward, _, info = result
                except Exception:
                    reward, info = 0, {}
                try:
                    # Capture current agent_obs from caller frame locals
                    frame = inspect.currentframe()
                    caller = frame.f_back if frame else None
                    obs = None
                    if caller is not None:
                        obs = caller.f_locals.get("agent_obs")
                    thinking = caller.f_locals.get("thought") if caller is not None else None
                    parse_err = None
                    try:
                        if isinstance(action, dict):
                            parse_err = action.get("_parse_error")
                    except Exception:
                        parse_err = None
                    collector.record_step(
                        step_index=last["step_index"],
                        obs=obs,
                        action=action,
                        thinking=thinking,
                        result=(info or {}).get("last_action_result"),
                        reward=reward,
                        events=(info or {}).get("events", []),
                        parse_error=parse_err,
                    )
                    last["step_index"] += 1
                except Exception:
                    pass
                return result

            # Bind wrapper as a bound method to ensure 'self' is inserted
            try:
                env.step = types.MethodType(wrapped_env_step, env)
            except Exception:
                pass

            try:
                result = await func(self, env, env_info, *args, **kwargs)
                try:
                    trajectory = collector.end_run(summary=result if isinstance(result, dict) else None)
                    if on_finish is not None:
                        on_finish(trajectory)
                except Exception:
                    pass
                return result
            finally:
                # Restore original
                try:
                    env.step = original_env_step
                except Exception:
                    pass

        return wrapper

    return decorator

def _load_basic_info(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_candidates(workspace_path: str) -> Dict[str, Any]:
    """
    Summarize candidates under <workspace_path>/candidates.

    For each candidate, compute deltas vs parent and a success flag:
      success := (acc_child > acc_parent) or (acc_child == acc_parent and cost_child < cost_parent)

    Returns a dict and also writes:
      - <workspace_path>/summary.json
      - <workspace_path>/candidates/candidate_<n>/optimization_result.json
    """
    cdir = os.path.join(workspace_path, "candidates")
    result: Dict[str, Any] = {
        "workspace_path": workspace_path,
        "candidates": [],
        "edges": [],
        "best": None,
    }

    if not os.path.isdir(cdir):
        with open(os.path.join(workspace_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # Load basics
    basics_by_round: Dict[int, Dict[str, Any]] = {}
    for name in sorted(os.listdir(cdir)):
        if not name.startswith("candidate_"):
            continue
        try:
            r = int(name.split("_")[-1])
        except Exception:
            continue
        info = _load_basic_info(os.path.join(cdir, name, "basic_info.json")) or {}
        info["folder_name"] = name
        basics_by_round[r] = info

    # Build summaries
    def _m(info: Dict[str, Any], key: str) -> Optional[float]:
        try:
            val = (info.get("metrics") or {}).get(key)
            return None if val is None else float(val)
        except Exception:
            return None

    best = {"round": None, "accuracy": -1.0, "cost": None}
    for r in sorted(basics_by_round.keys()):
        info = basics_by_round[r]
        parent = info.get("parent")
        acc = _m(info, "accuracy")
        cost = _m(info, "cost")
        parent_acc = None
        parent_cost = None
        acc_delta = None
        cost_delta = None
        success = None

        if parent is not None and parent in basics_by_round:
            pinfo = basics_by_round[parent]
            parent_acc = _m(pinfo, "accuracy")
            parent_cost = _m(pinfo, "cost")
            if parent_acc is not None and acc is not None:
                acc_delta = acc - parent_acc
            if parent_cost is not None and cost is not None:
                cost_delta = cost - parent_cost
            # Success rule
            if acc is not None and parent_acc is not None:
                if acc > parent_acc:
                    success = True
                elif acc == parent_acc and (cost is not None and parent_cost is not None) and cost < parent_cost:
                    success = True
                else:
                    success = False
            else:
                success = False

            result["edges"].append([parent, r])

        # Update best
        if acc is not None and acc > best["accuracy"]:
            best = {"round": r, "accuracy": acc, "cost": cost}

        item = {
            "round": r,
            "parent": parent,
            "accuracy": acc,
            "cost": cost,
            "parent_accuracy": parent_acc,
            "parent_cost": parent_cost,
            "acc_delta": acc_delta,
            "cost_delta": cost_delta,
            "success": success,
            "trajectory_path": info.get("trajectory_path"),
        }
        result["candidates"].append(item)

        # Write per-candidate summary
        try:
            with open(os.path.join(cdir, info.get("folder_name"), "optimization_result.json"), "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    result["best"] = best if best["round"] is not None else None

    # Write root summary
    try:
        with open(os.path.join(workspace_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return result

