#!/usr/bin/env python3
"""
Scripts for environment generator. It run this for test baisc environments execution.
"""

import argparse
import os
import sys
import yaml
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from base.env.base_env import SkinEnv
from base.agent.base_solver import SolverAgent
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.utils import read_file_content


class SolverRunner:
    """Solver runner that loads environment and level, then runs the solver."""
    
    def __init__(self, workspace_dir: str = "workspace/envs"):
        self.workspace_dir = Path(workspace_dir)
        if not self.workspace_dir.exists():
            raise ValueError(f"Workspace directory does not exist: {self.workspace_dir}")
    
    def list_environments(self) -> List[str]:
        """List all available environments."""
        envs = []
        for env_dir in self.workspace_dir.iterdir():
            if env_dir.is_dir() and (env_dir / "config.yaml").exists():
                envs.append(env_dir.name)
        return sorted(envs)
    
    def list_levels(self, env_name: str) -> List[str]:
        """List all levels for the specified environment."""
        env_dir = self.workspace_dir / env_name
        if not env_dir.exists():
            raise ValueError(f"Environment does not exist: {env_name}")
        
        levels_dir = env_dir / "levels"
        if not levels_dir.exists():
            return []
        
        levels = []
        for level_file in levels_dir.glob("*.yaml"):
            if level_file.name != "intelligent_generation_summary.yaml":
                # Remove .yaml suffix as level id
                level_id = level_file.stem
                levels.append(level_id)
        
        return sorted(levels)
    
    def get_env_info(self, env_name: str) -> Dict[str, str]:
        """Get environment information."""
        env_dir = self.workspace_dir / env_name
        
        info = {}
        
        # Read agent instruction
        agent_instruction_path = env_dir / "agent_instruction.txt"
        if agent_instruction_path.exists():
            info["agent_instruction"] = read_file_content(str(agent_instruction_path))
        
        # Read action space
        action_space_path = env_dir / "action_space.txt"
        if action_space_path.exists():
            info["action_space"] = read_file_content(str(action_space_path))
        
        # Read configuration
        config_path = env_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                info["config"] = config
        
        return info
    
    def load_environment_class(self, env_name: str):
        """Dynamically load the environment class."""
        # Use absolute path to avoid working directory issues
        env_dir = self.workspace_dir.resolve() / env_name
        env_main_path = env_dir / "env_main.py"
        
        if not env_main_path.exists():
            raise ValueError(f"Environment main file does not exist: {env_main_path}")
        
        # Dynamically import module
        spec = importlib.util.spec_from_file_location("env_main", str(env_main_path))
        env_module = importlib.util.module_from_spec(spec)
        
        # Add environment directory to sys.path to support relative imports
        env_dir_str = str(env_dir)
        if env_dir_str not in sys.path:
            sys.path.insert(0, env_dir_str)
        
        try:
            spec.loader.exec_module(env_module)
        except Exception as e:
            raise ImportError(f"Failed to load environment module {env_name}: {e}")
        finally:
            # Clean up sys.path
            if env_dir_str in sys.path:
                sys.path.remove(env_dir_str)
        
        # Find class that inherits from SkinEnv
        
        
        env_class = None
        for attr_name in dir(env_module):
            attr = getattr(env_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, SkinEnv) and 
                attr != SkinEnv):
                env_class = attr
                break
        
        if env_class is None:
            raise ValueError(f"No valid environment class found in {env_name}")
        
        return env_class
    
    def _load_environment_class_from_current_dir(self):
        """Load environment class from current directory (assuming we are in the env dir)."""
        env_main_path = Path("env_main.py")
        
        if not env_main_path.exists():
            raise ValueError(f"Environment main file does not exist: {env_main_path.resolve()}")
        
        # Dynamically import module
        spec = importlib.util.spec_from_file_location("env_main", str(env_main_path))
        env_module = importlib.util.module_from_spec(spec)
        
        # Add current directory to sys.path to support relative imports
        current_dir = str(Path.cwd())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            spec.loader.exec_module(env_module)
        except Exception as e:
            raise ImportError(f"Failed to load environment module: {e}")
        finally:
            # Clean up sys.path
            if current_dir in sys.path:
                sys.path.remove(current_dir)
    
        env_class = None
        for attr_name in dir(env_module):
            attr = getattr(env_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, SkinEnv) and 
                attr != SkinEnv):
                env_class = attr
                break
        
        if env_class is None:
            raise ValueError(f"No valid environment class found in environment module")
        
        return env_class
    
    def validate_level(self, env_name: str, level_id: str) -> bool:
        """Validate that the level file exists and is valid."""
        env_dir = self.workspace_dir / env_name
        level_path = env_dir / "levels" / f"{level_id}.yaml"
        
        if not level_path.exists():
            return False
        
        try:
            with open(level_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError:
            return False
    
    async def run_solver(self, env_name: str, level_id: str, 
                        max_steps: Optional[int] = None,
                        llm_model: str = "deepseek/deepseek-chat-v3.1") -> Dict[str, Any]:
        """Run the solver."""
        
        print(f"🚀 Starting Solver")
        print(f"   Environment: {env_name}")
        print(f"   Level: {level_id}")
        print(f"   LLM: {llm_model}")
        if max_steps:
            print(f"   Max steps: {max_steps}")
        print()
        
        # Validate level
        if not self.validate_level(env_name, level_id):
            raise ValueError(f"Level {level_id} does not exist or is invalid in environment {env_name}")
        
        # Get environment info
        env_info = self.get_env_info(env_name)
        if not env_info.get("agent_instruction") or not env_info.get("action_space"):
            raise ValueError(f"Environment {env_name} is missing required configuration files")
        
        # Get environment directory
        env_dir = self.workspace_dir / env_name
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        try:
            # First, create the LLM instance in the project root
            llm_config = LLMsConfig.default()
            available_models = llm_config.get_all_names()
            if llm_model not in available_models:
                print(f"⚠️  Warning: LLM model {llm_model} is not available, using default model")
                if available_models:
                    llm_model = available_models[0]
                else:
                    raise ValueError("No available LLM model configuration")
            
            llm = create_llm_instance(llm_config.get(llm_model))
            
            # Switch to the environment directory for environment-related operations
            os.chdir(env_dir)
            
            # Load environment class (in the correct directory now)
            env_class = self._load_environment_class_from_current_dir()
            
            # Create environment instance
            env = env_class(env_id=f"{env_name}_solver_test")
            
            # Prepare environment info
            solver_env_info = {
                "agent_instruction": env_info["agent_instruction"],
                "action_space": env_info["action_space"],
                "world_id": level_id,
            }
            
            if max_steps is not None:
                solver_env_info["max_step"] = max_steps
            
            # Create and run solver
            solver = SolverAgent(llm=llm)
            
            print("🤖 Solver is running...")
            print("-" * 50)
            
            result = await solver.run(env, solver_env_info)
            
            print("-" * 50)
            print("✅ Solver finished running!")
            print(f"   Total reward: {result.get('total_reward', 0)}")
            print(f"   Steps taken: {result.get('step', 0)}")
            if result.get('events_count'):
                print("   Event stats:")
                for event, count in result['events_count'].items():
                    print(f"     {event}: {count}")
            
            return result
            
        except Exception as e:
            print(f"❌ Solver error: {e}")
            raise
            
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Solver Agent for a specified environment and level.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--env", help="Environment name")
    parser.add_argument("--level", help="Level ID")
    parser.add_argument("--env-dir", default="workspace/envs", help="Environment workspace directory (default: workspace/envs)")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps")
    parser.add_argument("--llm-model", default="deepseek/deepseek-chat-v3.1", help="LLM model to use (default: deepseek/deepseek-chat-v3.1)")
    parser.add_argument("--list-envs", action="store_true", help="List all available environments")
    parser.add_argument("--list-levels", action="store_true", help="List all levels for the specified environment")
    
    args = parser.parse_args()
    
    try:
        runner = SolverRunner(args.env_dir)
        
        if args.list_envs:
            print("📋 Available environments:")
            envs = runner.list_environments()
            if envs:
                for env in envs:
                    print(f"  - {env}")
            else:
                print("  No environments found")
            return
        
        if args.list_levels:
            if not args.env:
                print("❌ Error: --env must be specified to list levels")
                return
            
            print(f"📋 Levels available for environment {args.env}:")
            levels = runner.list_levels(args.env)
            if levels:
                for level in levels:
                    print(f"  - {level}")
            else:
                print(f"  No levels found in environment {args.env}")
            return
        
        # Run solver
        if not args.env or not args.level:
            print("❌ Error: --env and --level must be specified")
            parser.print_help()
            return
        
        # Run async function
        result = asyncio.run(runner.run_solver(
            args.env, 
            args.level, 
            args.max_steps,
            args.llm_model
        ))
        
        print(f"\n🎉 Finished! Result: {result}")
        
    except KeyboardInterrupt:
        print("\n⚠️  User interrupted operation")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
