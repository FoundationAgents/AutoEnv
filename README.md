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
- [ ] Add Coding Agents Option: Codex, Gemini Cli.
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

## Awesome work powered by AutoEnv

- [Reasoning via Video](https://arxiv.org/abs/2511.15065): The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks


## Acknowledgements

Thanks to 
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), 
[codex](https://github.com/openai/codex), 
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
