"""
Centralized prompt templates for MiniSWE agents.
"""

from pathlib import Path


def benchmark_analysis_prompt(benchmark_path: Path, output_file: Path) -> str:
    """Prompt for benchmark-based analysis."""
    return f"""You are a game visualization expert. Analyze the benchmark at `{benchmark_path}` and create a comprehensive visualization plan.

**Working Directory:** {Path.cwd()}
**Output File (REQUIRED):** {output_file}

**Task: Generate a detailed JSON analysis file**

**IMPORTANT INSTRUCTIONS:**
- DO NOT create any summary files or README files
- ONLY create the required JSON file: {output_file.name}
- After writing the JSON file, IMMEDIATELY use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

**Steps:**

1. Read and analyze these files from `{benchmark_path}`:
   - env_desc.txt (environment description)
   - action_space.txt (action space definition)
   - config.yaml (environment configuration)
   - env_main_use.py or env_main.py (implementation code)
   - agent_instruction.txt (task instructions)
   - levels/*.yaml (sample level - pick any one)

2. Based on the analysis, create ONLY ONE JSON file at `{output_file}` with this structure. For every entry in `required_assets`, add a `processing_guidance` field that provides specific, actionable instructions for post-processing:

```json
{{
  "visual_theme": "Detailed theme description (e.g., medieval underground dungeon, sci-fi space station)",
  "art_style": "Art style (e.g., pixel art, hand-drawn cartoon, low-poly 3D, minimalist)",
  "color_palette": "Color scheme (e.g., dark atmospheric, vibrant colorful, monochrome)",
  "rendering_type": "Rendering type (grid_2d / abstract_dashboard / symbolic)",

  "environment_analysis": {{
    "is_spatial": true/false,
    "has_tiles": true/false,
    "has_agent": true/false,
    "has_objects": true/false,
    "observation_type": "full/partial/egocentric/noisy"
  }},

  "required_assets": [
    {{
      "name": "asset_name",
      "type": "tile/character/object/ui/overlay",
      "description": "Detailed visual description",
      "purpose": "Purpose in the game",
      "priority": 1-5,
      "is_tileable": true/false,
      "processing_guidance": "🔥 CRITICAL: Specific instructions for raw generation AND post-processing. Follow this template:\\n\\nRAW GENERATION REQUIREMENTS:\\n- Generate on perfectly solid pure white (#FFFFFF) background with NO glow/bloom effects\\n- [If light subject: use pure black (#000000) instead]\\n- Fill [70-85% for sprites / 90-95% for tiles] of 256x256 canvas\\n- [Additional generation requirements specific to this asset]\\n\\nPOST-PROCESSING:\\n- [Specific cropping/scaling/padding instructions]\\n- [Tiling requirements if applicable]\\n- [Transparency requirements]\\n\\nExample for tile: 'CRITICAL: Generate on perfectly solid pure white (#FFFFFF) background with NO external glow/bloom effects. Fill the entire 256x256 canvas edge-to-edge (95%). Ensure all four edges seamlessly connect for tiling. POST-PROCESSING: Verify seamless tiling, resize to target size preserving aspect ratio, no background removal needed (will be overlaid).'\\n\\nExample for sprite: 'CRITICAL: Generate on perfectly solid pure white (#FFFFFF) background with NO glow/bloom effects. Center the subject filling 75% of canvas with even padding. Subject should have crisp edges. POST-PROCESSING: Remove white background for transparency, auto-crop to content bounds with 5px padding, resize to fit target size preserving aspect ratio, center in square canvas with transparent padding.'"
    }}
  ],

  "style_anchor_recommendation": {{
    "asset_name": "Which asset to use as style anchor",
    "reason": "Why this asset"
  }},

  "generation_strategy": {{
    "total_assets": number,
    "generation_order": ["order of generation"],
    "style_keywords": ["style keywords"]
  }}
}}
```

Use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT when done.
"""


def instruction_analysis_prompt(instruction: str, output_file: Path) -> str:
    """Prompt for instruction-based analysis."""
    return f"""You are a game design and visualization expert. Based on the user's game description, create a comprehensive game design and visualization plan.

**User's Game Description:**
{instruction}

**Working Directory:** {Path.cwd()}
**Output File (REQUIRED):** {output_file}

**Task: Generate a detailed JSON analysis file**

**IMPORTANT INSTRUCTIONS:**
- DO NOT create any summary files or README files
- ONLY create the required JSON file: {output_file.name}
- After writing the JSON file, IMMEDIATELY use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

**Your Task:**

1. Understand the Game Concept.
2. Design Game Mechanics.
3. Plan Visualization.
4. Create the JSON file at `{output_file}` following the schema used for benchmark analysis (include required_assets with processing_guidance as above).

Use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT when done.
"""


def strategy_prompt(analysis_file: Path, output_file: Path) -> str:
    """Prompt for strategy generation."""
    return f"""Read `{analysis_file}` and create asset generation strategy in `{output_file}`.

**Input:** {analysis_file}
**Output (REQUIRED):** {output_file}
**Working Directory:** {Path.cwd()}

**Instructions:**
1. Read analysis JSON
2. Identify "style anchor" asset (generated first using text-to-image)
3. Other assets depend on style anchor (generated in parallel using image-to-image)
4. Write JSON to {output_file.name}
5. Use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

**Output JSON Structure:**
```json
{{
  "rendering_approach": {{"type": "tilemap/sprite/dashboard", "rationale": "..."}},
  "style_anchor": "asset_id",
  "assets": [
    {{
      "id": "style_anchor",
      "name": "...",
      "dependencies": [],
      "priority": 100,
      "is_tileable": true/false,
      "processing_guidance": "copy from analysis",
      "prompt_strategy": {{
        "base_prompt": "[Subject], [art style], [view angle], centered filling 70-85%, solid white bg, NO glow/bloom, clean edges"
      }},
      "generation_method": "text-to-image"
    }},
    {{
      "id": "other_asset",
      "dependencies": ["style_anchor"],
      "priority": 10,
      "prompt_strategy": {{"base_prompt": "...match style_anchor..."}},
      "generation_method": "image-to-image",
      "reference_assets": ["style_anchor"]
    }}
  ]
}}
```

**CRITICAL prompt requirements:**
- Subject description + art style + view angle
- "centered filling 70-85% of canvas with even padding"
- High-contrast solid background (white for dark subjects, black for light subjects, opposite color for colorful subjects), no gradients
- Absolutely NO bloom/glow/halo effects
- Clean crisp edges for background removal

Write {output_file.name} then use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
"""


__all__ = [
    "benchmark_analysis_prompt",
    "instruction_analysis_prompt",
    "strategy_prompt",
]
