# AGENTS.md

## Purpose

AutoEnv should generate and evaluate environments that share one core game logic while supporting multiple presentation modalities:
- Text environments for LLM policy learning.
- 2D visual environments for image-based interaction.
- 3D visual environments for richer embodied interaction.

This document defines the current architecture, explains how 3D is built in PR #8, and sets the abstraction contract to align text/2D/3D under one model.

## Current Code Architecture

Core runtime is split into three layers.

1. Environment semantics (`base/env`)
- `BaseEnv`: true state, transition, reward.
- `ObsEnv`: semantic observation extraction (`observe_semantic`).
- `SkinEnv`: skin rendering over semantic observations.

2. Solver/runtime (`base/agent`, `benchmarks/base`)
- Solver consumes skin output as agent observation.
- Benchmark runner dynamically loads each env and executes solver loop.

3. Generation pipelines (`autoenv/pipeline`)
- `GeneratorPipeline`: environment code/config/levels.
- `VisualPipeline`: visual assets and runnable visual scene.

## How 3D Is Built Today

`VisualPipeline.create_default(dimension="3d")` builds the DAG:

`Analyzer -> Strategist -> AssetGenerator -> BackgroundRemoval -> Image3DConvert -> ThreeJSAssembly`

Node responsibilities:
- `AnalyzerNode`: parse benchmark/instruction and write `analysis.json`.
- `StrategistNode`: produce `strategy.json` with asset plan.
- `AssetGeneratorNode`: generate style-consistent 2D assets.
- `BackgroundRemovalNode`: remove background/crop for cleaner conversion.
- `Image3DConvertNode`: call Meshy image-to-3D, poll progress, download `.glb` to `models_3d/`.
- `ThreeJSAssemblyNode`: build `game/index.html`, copy models to `game/models/`, optionally ask agent to enhance JS logic.

Current 3D output is a scene runtime (`index.html + models/*.glb`) and not yet a canonical environment runtime with the same authority as `transition/reward` in Python env classes.

## Alignment Problem

Without a shared contract, text/2D/3D can diverge:
- Agent input format differs by modality.
- Human display and agent display are not explicitly separated.
- Visual pipeline artifacts are not normalized for upstream orchestrators.
- 3D scene logic can drift away from canonical state transition rules.

## Aligned Abstraction (Adopted)

### 1) Unified skin output contract

`base/env/skin_output.py` introduces `SkinRenderOutput`:
- `agent_view`: what solver receives.
- `human_view`: optional richer UI view.
- `modality`: `text | 2d | 3d | multi`.
- `artifacts`: paths/handles for render assets.
- `metadata`: auxiliary render metadata.

`SkinEnv.render()` now normalizes output to this contract while keeping backward compatibility with existing `render_skin()` implementations.

### 2) Backward-compatible solver behavior

Both solver implementations (`base/agent/base_solver.py`, `benchmarks/base/agent.py`) now:
- Prefer `env.render(...)` when present.
- Fallback to legacy `render_skin(...)` behavior.
- Read observation from `info["agent_obs"]` first, then fallback to `info["skinned"]`.

This keeps all existing benchmark envs runnable while enabling richer modality outputs.

### 3) Unified visual artifact manifest

`AutoEnvContext` now carries `skin_manifest`.
- 2D `AssemblyNode` sets manifest with game entrypoint and assets dir.
- 3D `ThreeJSAssemblyNode` sets manifest with HTML entrypoint and models dir.

This gives upper layers a single way to consume generated skins regardless of modality.

## Canonical Rule For Future Work

- `transition()` and `reward()` remain the only source of environment truth.
- All modalities are projections of semantic state, not separate game authorities.
- Visual code (pygame/three.js) should be treated as view/runtime adapters unless intentionally promoted to core env logic.

## Recommended Next Steps

1. Add a `TextAssemblyNode` that emits the same `skin_manifest` contract for text-only runs.
2. Add consistency tests:
- same action sequence over same seed,
- text/2D/3D should produce identical semantic state traces and reward traces.
3. Move three.js enhancement prompts toward generated view adapters, not independent game rule engines.
4. Define versioned schema for `skin_manifest` to stabilize downstream tooling.

## Practical Usage Pattern

When adding a new modality:

1. Keep semantic env unchanged (`BaseEnv/ObsEnv`).
2. Implement renderer adapter that returns `SkinRenderOutput`.
3. Ensure generation pipeline writes `skin_manifest` with a valid entrypoint.
4. Validate semantic consistency against text baseline.

This keeps AutoEnv extensible without modality-specific logic forks.
