"""
Visualization Pipeline Prompts
Unified management of all pipeline node prompts.
Use .format() to replace placeholders.
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
   - action_space.txt (action space definition - CRITICAL for game controls)
   - config.yaml (environment configuration)
   - env_main_use.py or env_main.py (implementation code - CRITICAL for game rules)
   - agent_instruction.txt (task instructions - win/lose conditions)
   - levels/*.yaml (sample level - pick any one to understand level format)

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

  "game_rules": {{
    "actions": [
      {{
        "name": "ActionName",
        "key_binding": "arrow_up / space / etc",
        "description": "What this action does",
        "implementation": "Brief description of transition logic from env_main.py"
      }}
    ],
    "win_condition": "Exact win condition from code",
    "lose_condition": "Exact lose condition from code (if any)",
    "special_mechanics": ["list of special game mechanics like pushing boxes, sliding on ice, etc."]
  }},

  "level_format": {{
    "file_pattern": "levels/level_*.yaml",
    "structure": {{
      "agent": {{"pos": "[row, col]"}},
      "tiles": {{"grid": "2D array of tile types", "size": "[rows, cols]"}},
      "objects": {{"description": "object types and their properties"}},
      "globals": {{"max_steps": "number"}}
    }},
    "tile_types": ["list of all tile type strings used in grid"],
    "object_types": ["list of all object types"]
  }},

  "required_assets": [
    {{
      "name": "asset_name",
      "type": "tile/character/object/ui/overlay",
      "description": "Detailed visual description",
      "purpose": "Purpose in the game",
      "maps_to": "Which tile_type or object_type this asset represents",
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
- High-contrast solid background (white for dark subjects, black for light subjects)

Write {output_filename} then use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
"""

# ============== Asset Generation Prompts ==============

STYLE_CONSISTENT_PROMPT = """Above is the style reference image. Generate a new asset matching this exact visual style.

{base_prompt}

CRITICAL: Match the art style, color palette, and rendering technique of the reference image.
The new asset MUST look like it comes from the SAME GAME as the reference.
"""

# Game assembly prompt for instruction-based generation (simple demo game)
GAME_ASSEMBLY_INSTRUCTION_PROMPT = """You are a game developer. Generate a pygame demo game.

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**Strategy:**
```json
{strategy_json}
```

**Asset Dimensions (actual pixel sizes - MUST scale appropriately):**
{asset_dimensions}

**Assets Directory:** {game_dir}/assets/
**Output File:** {game_dir}/game.py

**Requirements:**
1. Create a complete, runnable pygame game
2. Load ALL assets and scale them appropriately based on dimensions above
3. Implement game mechanics based on the strategy
4. Include proper game loop, event handling, and rendering
5. Make the game playable with keyboard controls

**Scaling Requirements:**
- Assets are high-resolution and MUST be scaled down for game use
- Use pygame.transform.scale() to resize assets to appropriate game sizes
- Maintain aspect ratio when scaling

Write the game code to {game_dir}/game.py, then use COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
"""

# Game assembly prompt for benchmark-based generation (loads real levels)
GAME_ASSEMBLY_BENCHMARK_PROMPT = """You are a game developer. Generate a pygame game that loads and plays benchmark levels.

**CRITICAL - COMMAND EXECUTION RULES:**
- Execute ONE command at a time
- Wait for each command to complete before proceeding
- Never combine multiple commands in one response

**Strategy (contains game rules, level format, and assets):**
```json
{strategy_json}
```

**Asset Dimensions (actual pixel sizes - MUST scale appropriately):**
{asset_dimensions}

**Benchmark Path:** {benchmark_path}
**Assets Directory:** {game_dir}/assets/
**Output File:** {game_dir}/game.py

**CRITICAL REQUIREMENTS - The game MUST:**

1. **Load levels from YAML files:**
   - Load levels from `{benchmark_path}/levels/*.yaml`
   - Parse the exact YAML structure defined in `level_format` from strategy
   - Support switching levels (e.g., N/P keys for next/previous)

2. **Implement exact game rules from strategy:**
   - Map keyboard keys to actions as defined in `game_rules.actions`
   - Implement transition logic exactly as described
   - Check win/lose conditions as specified

3. **Render the game state:**
   - Use assets from {game_dir}/assets/
   - Map tile types and object types to corresponding asset images
   - Scale assets appropriately for the grid

4. **Show game status:**
   - Display current level name
   - Show step count / max steps
   - Display win/lose messages

**Scaling Requirements:**
- Assets are high-resolution and MUST be scaled down for game use
- Use pygame.transform.scale() to resize assets to appropriate game sizes
- Calculate scale ratio: target_size / actual_size
- Maintain aspect ratio when scaling

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