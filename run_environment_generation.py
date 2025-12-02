import argparse
import asyncio
import os
from typing import Dict, List, Optional

import yaml

from autoenv.generator import Generator
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger

DEFAULT_CONFIG_PATH = "config/env_gen.yaml"


def _load_config(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _clean_value(value: Optional[str]) -> Optional[str]:
    """Treat placeholders or empty strings as missing."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or (stripped.startswith("<") and stripped.endswith(">")):
            return None
    return value


def _pick(config_val, cli_val, default=None):
    value = _clean_value(config_val)
    if value is not None:
        return value
    value = _clean_value(cli_val)
    if value is not None:
        return value
    return default


def _get_requirements_files(folder_path: str) -> List[str]:
    files = []
    if not folder_path:
        return files
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.endswith(".txt"):
            files.append(fpath)
    return sorted(files)


async def run_single_requirement(requirement, envs_root_path, gen_llm, re_llm):
    label = requirement if isinstance(requirement, str) else "inline"
    try:
        generator = Generator(llm=gen_llm, envs_root_path=envs_root_path, re_llm=re_llm)
        print(f"🚀 Starting generation for: {label}")
        await generator.run(requirement)
        print(f"✅ Finished generation for: {label}")
    except Exception as e:
        print(f"❌ Error for {label}: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Generate environments from themes.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Config YAML path (defaults to config/env_gen.yaml)")
    parser.add_argument("--model", help="LLM model name for generation")
    parser.add_argument("--theme", help="Single theme file path or inline requirement text")
    parser.add_argument("--themes-folder", help="Folder containing multiple theme txt files")
    parser.add_argument("--envs-root", help="Root folder to place generated environments")
    parser.add_argument("--concurrency", type=int, help="Parallelism when using themes folder")
    args = parser.parse_args()

    config = _load_config(args.config)

    # Config has priority; CLI is used only when config is missing/placeholder.
    theme_config = config.get("theme") or config.get("theme_path")
    themes_folder = _pick(config.get("themes_folder"), args.themes_folder)
    theme = _pick(theme_config, args.theme)
    if theme:
        themes_folder = None  # Theme wins over folder when both are set

    model_name = _pick(config.get("model"), args.model)
    if not model_name:
        raise ValueError("Model is required. Set it in the config file or pass --model.")

    concurrency = _pick(config.get("concurrency"), args.concurrency, 1)
    try:
        concurrency = max(1, int(concurrency))
    except Exception:
        concurrency = 1

    envs_root_path = _pick(config.get("envs_root_path"), args.envs_root, "workspace/envs")
    os.makedirs(envs_root_path, exist_ok=True)
    logger.info(f"Using envs_root_path: {envs_root_path}")

    llm_config_mgr = LLMsConfig.default()
    gen_llm = create_llm_instance(llm_config_mgr.get(model_name))
    re_llm = gen_llm

    print(f"🔧 Config file: {args.config}")
    print(f"🚀 Generation model: {model_name}")
    print(f"📁 Output path: {envs_root_path}")

    if theme:
        await run_single_requirement(theme, envs_root_path, gen_llm, re_llm)
        return

    if not themes_folder:
        print("❌ No theme or themes_folder provided. Set it in the config or pass --theme / --themes-folder.")
        return

    requirement_files = _get_requirements_files(themes_folder)
    if not requirement_files:
        print(f"❌ No .txt theme files found in folder: {themes_folder}")
        return

    print(f"📄 Found {len(requirement_files)} theme files in {themes_folder}")
    print(f"🔢 Using concurrency: {concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async def sem_task(req_path):
        async with sem:
            await run_single_requirement(req_path, envs_root_path, gen_llm, re_llm)

    tasks = [sem_task(f) for f in requirement_files]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
