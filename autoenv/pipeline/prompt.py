"""
Visualization Pipeline Prompts
统一管理所有 Pipeline 节点使用的提示词
使用 .format() 替换占位符
"""

# ============== Analysis Prompts ==============

BENCHMARK_ANALYSIS_PROMPT = """You are a game visualization expert. Analyze the benchmark at `{benchmark_path}` and create a comprehensive visualization plan.

**Working Directory:** {cwd}
**Output File (REQUIRED):** {output_file}

**Task: Generate a detailed JSON analysis file**

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**IMPORTANT INSTRUCTIONS:**
- DO NOT create any summary files or README files
- ONLY create the required JSON file: {output_filename}
- After writing the JSON file, IMMEDIATELY use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

**Steps:**

1. Read and analyze these files from `{benchmark_path}`:
   - env_desc.txt (environment description)
   - action_space.txt (action space definition)
   - config.yaml (environment configuration)
   - env_main_use.py or env_main.py (implementation code)
   - agent_instruction.txt (task instructions)
   - levels/*.yaml (sample level - pick any one)

2. Based on the analysis, create ONLY ONE JSON file at `{output_file}` with this structure:

```json
{{
  "visual_theme": "Detailed theme description",
  "art_style": "Art style (pixel art, hand-drawn, etc.)",
  "color_palette": "Color scheme",
  "rendering_type": "grid_2d / abstract_dashboard / symbolic",

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
      "is_tileable": true/false
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

INSTRUCTION_ANALYSIS_PROMPT = """You are a game design and visualization expert.

**User's Game Description:**
{instruction}

**Working Directory:** {cwd}
**Output File (REQUIRED):** {output_file}

**Task: Generate a detailed JSON analysis file**

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**IMPORTANT INSTRUCTIONS:**
- DO NOT create any summary files or README files
- ONLY create the required JSON file: {output_filename}
- After writing the JSON file, IMMEDIATELY use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

**Your Task:**

1. Understand the Game Concept.
2. Design Game Mechanics.
3. Plan Visualization.
4. Create the JSON file at `{output_file}` with this structure:

```json
{{
  "visual_theme": "Detailed theme description",
  "art_style": "Art style (pixel art, hand-drawn, etc.)",
  "color_palette": "Color scheme",
  "rendering_type": "grid_2d / abstract_dashboard / symbolic",

  "required_assets": [
    {{
      "name": "asset_name",
      "type": "tile/character/object/ui/overlay",
      "description": "Detailed visual description",
      "purpose": "Purpose in the game",
      "priority": 1-5,
      "is_tileable": true/false
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

# ============== Strategy Prompts ==============

STRATEGY_PROMPT = """Read `{analysis_file}` and create asset generation strategy in `{output_file}`.

**Input:** {analysis_file}
**Output (REQUIRED):** {output_file}
**Working Directory:** {cwd}

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**Instructions:**
1. Read analysis JSON
2. Identify "style anchor" asset (generated first using text-to-image)
3. Other assets depend on style anchor (generated in parallel using image-to-image)
4. Write JSON to {output_filename}
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
- High-contrast solid background (white for dark subjects, black for light subjects)
- Absolutely NO bloom/glow/halo effects
- Clean crisp edges for background removal

Write {output_filename} then use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
"""

# ============== Asset Generation Prompts ==============

STYLE_CONSISTENT_PROMPT = """Above is the style reference image. Generate a new asset matching this exact visual style.

{base_prompt}

CRITICAL: Match the art style, color palette, and rendering technique of the reference image.
The new asset MUST look like it comes from the SAME GAME as the reference.
"""

GAME_ASSEMBLY_PROMPT = """You are a game developer. Generate a complete pygame game.

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**Strategy:**
```json
{strategy_json}
```

**Available Assets:** {asset_list}
**Assets Directory:** {game_dir}/assets/
**Output File:** {game_dir}/game.py

**Requirements:**
1. Create a complete, runnable pygame game
2. Load all assets from the assets/ directory
3. Implement basic game mechanics based on the strategy
4. Include proper game loop, event handling, and rendering
5. Make the game playable and visually appealing

Write the game code to {game_dir}/game.py, then use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
"""

DEFAULT_GAME_CODE = '''"""Auto-generated pygame game"""
import pygame
import os

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AutoEnv Game")

# Load assets
assets = {{}}
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
for asset_name in {asset_list}:
    path = os.path.join(assets_dir, f"{{asset_name}}.png")
    if os.path.exists(path):
        assets[asset_name] = pygame.image.load(path).convert_alpha()

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    # Display loaded assets
    x, y = 50, 50
    for name, img in assets.items():
        screen.blit(img, (x, y))
        x += img.get_width() + 20
        if x > SCREEN_WIDTH - 100:
            x = 50
            y += img.get_height() + 20

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
'''

# ============== QA Analysis Prompts ==============

QA_ANALYSIS_PROMPT = """You are a game art quality control expert.

**Target Specifications:**
- Art Style: {art_style}
- Theme: {theme}

Examine both images and identify visual problems:

1. **Style Inconsistency** (CRITICAL) - Do all assets follow the same art style?
2. **Contrast Issues** (CRITICAL) - Can player/characters be distinguished?
3. **Missing/Broken Assets** (CRITICAL) - Any placeholders or glitches?
4. **Quality Issues** (WARNING) - Resolution/detail appropriate?
5. **Composition Issues** (INFO) - Overall balance and hierarchy?

Output JSON array of issues:
```json
[{{"severity": "critical|warning|info", "category": "...", "description": "...", "suggestion": "..."}}]
```

If no issues: `[]`
"""

# ============== Refinement Prompts ==============

REGENERATION_PROMPT = """🔧 **ASSET REGENERATION FOR GAME ENGINE**

**Original Specification:**
{base_prompt}

**Problem Identified:**
{instruction}

**Your Task:**
Create an IMPROVED version that:
1. Keeps the SAME visual content
2. Fixes the technical issues
3. Makes it EASY to process automatically

**CRITICAL Technical Requirements:**
- SOLID white background (#FFFFFF) - NO gradients
- Subject centered, filling 70-75% of canvas
- NO glow/bloom/halo effects
- Clean, sharp edges
- Subject isolated and complete
"""
