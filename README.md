# AutoEnv: Automating Environment Generation For Language Model Agents

[![Arxiv](https://img.shields.io/badge/2511.19304-arXiv-red)](https://arxiv.org/abs/2511.19304)

> If you encounter any difficulties in usingthe code, please contact us at [jzhang361@connect.hkust-gz.edu.cn](mailto:jzhang361@connect.hkust-gz.edu.cn) or [amagipeng@gmail.com](mailto:amagipeng@gmail.com).

## Overview

AutoEnv is an automated environment infrastructure for language-model agents, designed to scale both across environments and within each environment. Instead of hand-crafting a few fixed tasks, AutoEnv factorizes an environment into reward rules, transition dynamics, and observation “skins,” so that the same core world can be instantiated with different rule distributions and presentations (text-only, tabular, grid-based, etc.).

<figure style="text-align: center;">
  <img src="assets/autoenv-process.png" alt="Process of Automating Environment Generation">
  <figcaption>
    <em>Figure 1. Process of automating environment generation in AutoEnv.</em>
  </figcaption>
</figure>


Our long-term goal is to provide a unified way to automatically expand environments from text themes to richer modalities, including multimodal settings and 3D game worlds, while also scaling data inside each environment via level generators, validators, and large amounts of interaction trajectories.

Built on top of this infrastructure, we run cross-environment learning experiments with agents on the environments constructed by AutoEnv, and the results reveal robustness limitations in current agent learning methods. Beyond the original paper, however, AutoEnv is intended to be a general research platform—for studying environment generation, agent learning, reward design, and scaling laws in interactive worlds.


<figure style="text-align: center;">
  <img src="assets/learning.png" alt="Learning Experiments">
  <figcaption>
    <em>Figure 2. Impact of environment diversity on learning performance.</em>
  </figcaption>
</figure>


## Generated Environments

### AutoEnv-36 Dataset.

Using AutoEnv, we generate 36 environments with fully distinct rule sets, forming the AutoEnv-36 dataset. These environments are represented in text, and each environment contains 10 test levels and 5 validation levels. We provide the source code together with level generation scripts in the `benchmarks` directory.

### Inverse Semantic Control

This example shows two observation “skins” for the same underlying gridworld. On the left, symbols like `#`, `.`, `$`, `^`, and `@` follow one semantic mapping (e.g., walls, free cells, goals, hazards, agent), while on the right we systematically invert this mapping (e.g., swapping walls and free space) without changing the true transition or reward rules. By comparing agent performance across these two views, we can test whether an agent is actually learning the environment dynamics rather than relying on fixed prior assumptions about what each symbol should mean.

```text
######################           ......................
#....##....#....##...#           .####..####.####..###.
#..$.....###.....#...#           .##$#####...#####.###.
#..###....#....###..^#           .##...####.####...##^.
##..#.^....##..#..#..#           ..##.#^####..##.##.##.
#...##..##..^.##..#..#           .###..##..##^#..##.##.
#.#...........@.#..#.#           .#.###########@#.##.#.
#..##..^.#..##....#..#           .##..##^#.##..####.##.
#.....##....#..##....#           .#####..####.##..####.
######################           ......................
```

### MultiModal Environments

We generate multimodal skin for partial environments from **AutoEnv-36** as listed below:

<figure style="text-align: center;">
  <img src="assets/multimodal-diff-skin.png" alt="MultiModal Skin Generated For AutoEnv-36">
  <figcaption>
    <em>Figure 3. MultiModal Skin Generated For AutoEnv-36.</em>
  </figcaption>
</figure>

We also generate multimodal skin based on the rules of the same maze. Generated multimodal environments are listed here: 

<figure style="text-align: center;">
  <img src="assets/multimodal-skin.png" alt="MultiModal Skin Generated based on one rule">
  <figcaption>
    <em>Figure 4. MultiModal Skin Generated based on one Rule.</em>
  </figcaption>
</figure>

 
## RoadMap

- [ ] Add Environments Level Scaling Feature.
- [ ] Add Skin Control pipeline abstraction.
- [ ] Add Multimodal Environment Generation Pipelines.
- [ ] Add Three-stage Verification Pipeline for both text and multimodal environments.
- [ ] Add Learning Experiments Scripts.
- [x] Add Coding Agents Option: Codex, Claude Code SDK.
- [ ] Add 3D Environment Generation Pipelines.

## Repository Layout

```
AutoEnv/
├── autoenv/              # Environment generation logic and pipelines
├── base/                 # Core abstractions (LLM client, pipeline, env)
├── benchmarks/           # AutoEnv-36 benchmark environments
├── config/               # Configuration files
├── scripts/              # Utility scripts
└── workspace/            # Runtime outputs (envs, logs, costs)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

### 2. Configure Model Keys

Fill `config/model_config.yaml` with your model names and endpoints.

### 3. Run

**Environment Generation** (`run_environment_generation.py`): Generates text-based game environments from theme descriptions.

```bash
cp config/env_gen_example.yaml config/env_gen.yaml
# Edit config/env_gen.yaml with your settings
python run_environment_generation.py
```

**Skin Generation** (`run_environment_skin_generation.py`): Generates visual assets for existing environments or from text instructions.

```bash
cp config/env_skin_gen_example.yaml config/env_skin_gen.yaml
# Edit config/env_skin_gen.yaml with your settings
python run_environment_skin_generation.py
```

Cost summaries are automatically saved to `workspace/costs/`.

## Coding Agents

AutoEnv supports multiple coding agent backends for environment code generation and fixing. The code agent is used in the pipeline's CodeFixNode, LevelGenNode, and MaxRewardNode stages.

### Backend Options

| Backend   | Description                                                                  | Best For                            |
| --------- | ---------------------------------------------------------------------------- | ----------------------------------- |
| `miniswe` | Default [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)        | General use, works with any LLM     |
| `codex`   | OpenAI [Codex CLI](https://github.com/openai/codex)                          | OpenAI API users, fast execution    |
| `claude`  | Anthropic [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) | Anthropic API users, custom proxies |

### Configuration

Set the `code_agent_backend` field in `config/env_gen.yaml`:

```yaml
# Code agent backend: "miniswe" (default), "codex", "claude"
code_agent_backend: "codex" # or "claude" or "miniswe"
```

### MiniSWE Agent (Default)

The default agent using mini-swe-agent. Works with any LLM configured in `config/model_config.yaml`.

```yaml
# config/env_gen.yaml
code_agent_backend: "miniswe"
model: "gpt-4o" # or any configured LLM
```

No additional setup required beyond the standard model configuration.

### Codex Agent

Uses OpenAI's official Codex CLI for code generation.

#### Prerequisites

```bash
# Install Codex CLI
npm install -g @openai/codex
# or
brew install --cask codex

# Authenticate (recommended)
codex login

# Verify installation
codex whoami
```

#### Authentication Options

**Option 1: CLI Login (Recommended)**

```bash
codex login  # Opens browser for OAuth
```

**Option 2: Environment Variable**

```bash
export OPENAI_API_KEY=your-api-key
```

**Option 3: Custom Base URL** (for proxies)

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://your-proxy.example.com/v1
```

#### Configuration

```yaml
# config/env_gen.yaml
code_agent_backend: "codex"
```

### Claude Agent

Uses Anthropic's Claude Agent SDK for code generation.

#### Prerequisites

```bash
# Install Claude Agent SDK
pip install claude-agent-sdk
```

The Python SDK authenticates via environment variables (see Authentication Options below).

#### Authentication Options

**Option 1: Environment Variables (Recommended)**

```bash
export ANTHROPIC_API_KEY=your-api-key
```

**Option 2: Custom Base URL** (for proxies)

```bash
export ANTHROPIC_API_KEY=your-api-key
export ANTHROPIC_BASE_URL=https://your-proxy.example.com/api
```

#### Configuration

```yaml
# config/env_gen.yaml
code_agent_backend: "claude"
```

### Generate Environment with Code Agent

#### Quick Start

```bash
# 1. Copy example config
cp config/env_gen_example.yaml config/env_gen.yaml

# 2. Edit config (set code_agent_backend, theme, etc.)
vim config/env_gen.yaml

# 3. Run environment generation
python run_environment_generation.py
```

#### Example Configuration

```yaml
# config/env_gen.yaml
mode: "textual"
model: "gpt-4o"
concurrency: 1
theme: "A strategic puzzle game with resource management"
envs_root_path: "workspace/envs"
code_agent_backend: "codex" # Use Codex for code generation
```

#### Background Execution (Recommended for Long Tasks)

Code agents can take 10-30 minutes for complex environments. Run in background:

```bash
# Run in background with logging
nohup python run_environment_generation.py > /tmp/autoenv_gen.log 2>&1 &

# Monitor progress
tail -f /tmp/autoenv_gen.log

# Check if complete
ls workspace/envs/*/done.txt
```

### Troubleshooting

#### Codex CLI Issues

```bash
# Check if Codex is installed
codex --version

# Re-authenticate
codex logout && codex login

# Check current user
codex whoami
```

#### Claude Agent Issues

```bash
# Check if Python SDK is installed
pip show claude-agent-sdk

# Verify API key
echo $ANTHROPIC_API_KEY

# Test import
python -c "from claude_agent_sdk import query; print('SDK available')"
```

#### Timeout Issues

For complex environments, increase timeout in `autoenv/coder.py`:

```python
# Current default: 900 seconds (15 minutes)
agent = CodexAgent(timeout=1200)  # 20 minutes
```

## Benchmarking AutoEnv-36

Evaluate agents on the 36 benchmark environments (scores for all; cost only for LLM branch). See `benchmarks/README.md` for details.

- Built-in SolverAgent + LLMs (cost tracked):
  ```bash
  python benchmarks/run.py \
    --config config/benchmark/bench_llm_example.yaml \
    --mode test \
    --max-worlds 5
  ```
  `--mode` switches `levels/` vs `val_levels/`; `--max-worlds` limits worlds per env.

- Custom agent (score only): implement `run(env, env_info)`, then
  ```bash
  python benchmarks/run.py \
    --agent your_module:YourAgentAttr \
    --agent-kwargs '{"foo": 1}' \
    --mode val
  ```
  `--agent` accepts `module:Attr` or `/path/to/file.py:Attr`; Attr can be a class, factory, or pre-built instance.

Programmatic APIs are available in `benchmarks/api.py` (`benchmark_llms`, `benchmark_custom_agent`).

## Awesome work powered by AutoEnv

- [Reasoning via Video](https://arxiv.org/abs/2511.15065): The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks


## Acknowledgements

Thanks to
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
[codex](https://github.com/openai/codex),
[claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python),
[rembg](https://github.com/danielgatis/rembg),
for providing basic support for this project!

## Citation

If you find AutoEnv useful, we would appreciate it if you consider citing our work:
```
@article{zhang2025autoenv,
  title={AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning},
  author={Zhang, Jiayi and Peng, Yiran and Kong, Fanqi and Cheng, Yang and Wu, Yifan and Yu, Zhaoyang and Xiang, Jinyu and Ruan, Jianhao and Wang, Jinlin and Song, Maojia and others},
  journal={arXiv preprint arXiv:2511.19304},
  year={2025}
}
```
