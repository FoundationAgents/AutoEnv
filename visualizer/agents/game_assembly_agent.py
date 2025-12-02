"""
Game Assembly Agent - assemble generated assets into a runnable pygame game.
Takes generated assets + benchmark logic → produces a playable game.
"""

import ast
import base64
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from autoenv.miniswe_agent import MiniSWEAutoEnvAgent
from base.engine.async_llm import AsyncLLM


@dataclass
class GameMetadata:
    """Game metadata."""
    name: str
    description: str
    game_type: str  # 'grid_based', 'card_game', 'platformer', etc.
    grid_size: Optional[tuple] = None
    controls: List[str] = None


class GameCodeGenerator:
    """Generate pygame code with an LLM."""

    def __init__(
        self,
        llm_name: str = "claude-sonnet-4-5",
        use_miniswe: bool = True,
        step_limit: int = 80,
        cost_limit: float = 8.0,
        timeout: int = 120
    ):
        self.llm_name = llm_name
        self.use_miniswe = use_miniswe
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.timeout = timeout
        self.llm: Optional[AsyncLLM] = None if use_miniswe else AsyncLLM(llm_name)

    async def generate_game_code(
        self,
        benchmark_data: Dict[str, Any],
        asset_list: List[str],
        metadata: GameMetadata,
        environment_artifacts: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[Path] = None,
        benchmark_reference_paths: Optional[List[Path]] = None,
        benchmark_root: Optional[Path] = None
    ) -> str:
        """
        Generate complete pygame game code.

        Args:
            benchmark_data: Benchmark YAML data
            asset_list: Available asset files
            metadata: Game metadata

        Returns:
            Full Python/pygame code
        """

        prompt = self._build_code_generation_prompt(
            benchmark_data,
            asset_list,
            metadata,
            environment_artifacts or {},
            benchmark_reference_paths or [],
            benchmark_root
        )

        if self.use_miniswe and workspace_dir is not None:
            try:
                print("\n   🤖 Generating pygame code with MiniSWE agent...")
                return await self._generate_with_miniswe(
                    prompt,
                    workspace_dir,
                    benchmark_root,
                    benchmark_reference_paths or []
                )
            except Exception as exc:
                print(f"   ⚠️  MiniSWE generation failed: {exc}")

        if not self.llm:
            self.llm = AsyncLLM(self.llm_name)

        print("\n   🤖 Generating pygame code with direct LLM...")

        try:
            response = await self.llm(prompt)
            code = self._extract_code(response)
            line_count = len(code.splitlines())
            print(f"   ✅ Generated {line_count} lines of code")
            return code
        except Exception as e:
            print(f"   ❌ Code generation failed: {e}")
            return self._generate_fallback_code(metadata, asset_list)

    async def _generate_with_miniswe(
        self,
        prompt: str,
        workspace_dir: Path,
        benchmark_root: Optional[Path],
        reference_paths: List[Path]
    ) -> str:
        workspace_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = workspace_dir / "assembly_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        game_file = workspace_dir / "game.py"
        if game_file.exists():
            game_file.unlink()

        task = self._build_miniswe_task(workspace_dir, prompt_file, benchmark_root, reference_paths)

        agent = MiniSWEAutoEnvAgent(
            llm_name=self.llm_name,
            mode="yolo",
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
            environment_type="local",
            cwd=str(workspace_dir),
            timeout=self.timeout
        )

        result_str = await agent.run(task=task)

        status = self._parse_agent_status(result_str)
        print(f"   MiniSWE agent status: {status}")

        if not game_file.exists():
            raise RuntimeError("MiniSWE agent did not produce game.py")

        code = game_file.read_text(encoding="utf-8")
        if not code.strip():
            raise RuntimeError("Generated game.py is empty")

        return code

    def _build_miniswe_task(
        self,
        workspace_dir: Path,
        prompt_file: Path,
        benchmark_root: Optional[Path],
        reference_paths: List[Path]
    ) -> str:
        benchmark_section = "(no benchmark source files discovered)"
        if reference_paths:
            benchmark_section = "\n".join(f"  - {path.resolve()}" for path in reference_paths)

        benchmark_hint = "Unknown"
        if benchmark_root is not None:
            benchmark_hint = str(benchmark_root.resolve())

        return (
            "You are the Game Assembly automation agent. Follow the requirements in "
            f"{prompt_file.name} to build a runnable pygame game.\n\n"
            f"Working directory: {workspace_dir}\n"
            f"Instructions file: {prompt_file.name}\n"
            f"Benchmark reference directory: {benchmark_hint}\n"
            "Benchmark files to review before coding:\n"
            f"{benchmark_section}\n\n"
            "Required workflow:\n"
            "1. Inspect the benchmark implementation (transition, reward, observation code and YAML levels) so the pygame runtime matches its mechanics exactly.\n"
            "2. Inspect the instructions file to capture rendering, CLI, and asset requirements.\n"
            "3. Generate game.py in this directory using a single heredoc command (cat <<'EOF' > game.py).\n"
            "4. After writing the file, run python -m compileall game.py to confirm syntax.\n"
            "5. If compilation fails or behaviour is inconsistent with the benchmark logic (e.g., water impassable, fire ends the run, coordinate conventions), fix the file and re-run the check.\n"
            "6. Do not create unrelated files.\n"
            "7. When the build is correct, finish with: echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'.\n"
        )

    def _parse_agent_status(self, result_str: str) -> str:
        try:
            payload = ast.literal_eval(result_str)
            if isinstance(payload, dict):
                return str(payload.get("exit_status", "unknown"))
        except Exception:
            pass
        return "unknown"

    def _build_code_generation_prompt(
        self,
        benchmark_data: Dict[str, Any],
        asset_list: List[str],
        metadata: GameMetadata,
        environment_artifacts: Dict[str, Any],
        benchmark_reference_paths: List[Path],
        benchmark_root: Optional[Path]
    ) -> str:
        """Build the code-generation prompt."""

        # Simplify benchmark data for the prompt
        simplified_benchmark = {
            'meta': benchmark_data.get('meta', {}),
            'state_template': benchmark_data.get('state_template', {}),
            'transition': benchmark_data.get('transition', {}),
            'reward': benchmark_data.get('reward', {})
        }

        benchmark_str = yaml.dump(simplified_benchmark, default_flow_style=False)
        artifacts_str = yaml.dump(environment_artifacts, default_flow_style=False)

        asset_lines = chr(10).join(f"- {asset}" for asset in asset_list) or "- (no rendered assets)"

        benchmark_root_line = str(benchmark_root.resolve()) if benchmark_root else "(unknown)"
        reference_lines = "\n".join(
            f"- {path.resolve()}" for path in benchmark_reference_paths
        ) or "- (no additional benchmark files discovered)"

        return f"""You are the Game Assembly Agent in the AutoEnv visualizer pipeline. Produce a fully runnable pygame implementation that mirrors the benchmark behaviour exactly.

### Benchmark Context
{benchmark_str}

### Benchmark Root
- {benchmark_root_line}

### Benchmark Source Files (inspect these to understand mechanics)
{reference_lines}

### Bundled Environment Artifacts
{artifacts_str}

### Available Art Assets (PNG under assets/)
{asset_lines}

### Absolute Requirements
1. **Study the Benchmark Implementation**
    - Before writing code, read the benchmark Python sources listed above (e.g., `env_main.py`, observation policy, level generator, config, agent instructions).
    - Mirror the transition, reward, termination, and observation semantics exactly. Pay attention to coordinate conventions, impassable tiles, hazards, win conditions, and how `steps_left` is decremented.

2. **Deterministic Level Loading**
    - At runtime, load YAML level files from `levels/` and `val_levels/` (when present).
    - Build a TAB/ESC-friendly menu listing every YAML file; default to the first level when none selected. Respect CLI `--level` hints by name or 1-based index.

3. **Level State Hydration**
    - Hydrate the entire level state from YAML (`agent`, `globals`, `objects`, etc.) whenever a level is loaded or reset. Treat YAML fields as canonical—never replace them with random defaults.
    - When a field is absent, fall back to neutral values (0, empty list/dict) but preserve every known key so behaviour matches the benchmark.

4. **Benchmark-Accurate Mechanics**
    - Implement movement, collision, hazard effects, goal detection, reward accumulation, and termination checks exactly as defined in the benchmark implementation.
    - If the env code blocks certain tiles or ends the run on hazards, enforce the same rules in pygame. Do not introduce additional randomness beyond what the benchmark specifies.

5. **User Experience & Controls**
    - Show controls, current level name, and relevant statistics onscreen (steps, observations, profit, reward totals, etc.).
    - Support ESC to toggle menu/quit, R to restart current level, TAB (or ESC) to open the menu when multiple levels exist.
    - Provide lightweight animations/FX so state changes (e.g., observations, trades, collapses) are readable.

6. **Assets & Rendering**
    - Load sprites from `assets/<asset_id>.png`. If a file is missing, create a pygame Surface fallback with a solid colour.
    - Keep layout responsive to grid size and data-driven panels; do not hard-code pixel positions beyond derived constants.

7. **CLI Hooks**
    - Recognise `--level <name_or_index>` to auto-load a specific level name or 1-based index.
    - Implement `--screenshot <file>` mode: run an automatic short showcase (no interaction) for ~2-3 seconds, save the frame, then exit cleanly.

8. **Code Quality**
    - Provide a single `game.py` with clear classes (e.g., Maze, Agent, Game) and meaningful constants.
    - Avoid placeholder comments or TODOs. Use concise comments only where logic is non-obvious.
    - Ensure the code compiles under Python 3.10+ with pygame installed.

### Deliverable
Produce the complete `game.py` source code ready to run (`python game.py`) or capture (`python game.py --screenshot preview.png`). Do not include explanations—only the executable code.
"""

    def _extract_code(self, response: str) -> str:
        """Extract code from an LLM response."""

        import re

        # Extract fenced Python block
        code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Fallback: pull the whole response, trimming leading prose
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                in_code = True

            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        # Final fallback
        return response

    def _generate_fallback_code(
        self,
        metadata: GameMetadata,
        asset_list: List[str]
    ) -> str:
        """Generate fallback code (basic template)."""

        return f'''"""
{metadata.name}
{metadata.description}

Auto-generated pygame game
"""

import pygame
import sys
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='{metadata.name}')
parser.add_argument('--screenshot', type=str, help='Save screenshot and exit')
args = parser.parse_args()

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
TILE_SIZE = 256

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

class Game:
    """Main game class"""

    def __init__(self, screenshot_mode=False, screenshot_file=None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("{metadata.name}")
        self.clock = pygame.time.Clock()
        self.running = True
        self.assets = {{}}
        self.screenshot_mode = screenshot_mode
        self.screenshot_file = screenshot_file
        self.frame_count = 0

        self.load_assets()

    def load_assets(self):
        """Load game assets"""
        asset_dir = Path(__file__).parent / "assets"

        # Available assets
        asset_files = {asset_list}

        for asset_name in asset_files:
            asset_path = asset_dir / f"{{asset_name}}.png"
            try:
                if asset_path.exists():
                    self.assets[asset_name] = pygame.image.load(str(asset_path))
                    print(f"Loaded: {{asset_name}}")
                else:
                    # Fallback: colored rectangle
                    surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
                    surf.fill(GRAY)
                    self.assets[asset_name] = surf
                    print(f"Missing: {{asset_name}} (using fallback)")
            except Exception as e:
                print(f"Error loading {{asset_name}}: {{e}}")
                surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
                surf.fill((255, 0, 0))
                self.assets[asset_name] = surf

    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()

    def update(self):
        """Update game state"""
        pass

    def draw(self):
        """Render game"""
        self.screen.fill(WHITE)

        # Draw title
        font = pygame.font.Font(None, 48)
        title_text = font.render("{metadata.name}", True, BLACK)
        self.screen.blit(title_text, (20, 20))

        # Draw instructions
        font_small = pygame.font.Font(None, 24)
        instructions = [
            "ESC - Quit",
            "R - Restart",
            "",
            "Game implementation in progress..."
        ]
        y = 100
        for line in instructions:
            text = font_small.render(line, True, BLACK)
            self.screen.blit(text, (20, y))
            y += 30

        pygame.display.flip()

    def reset(self):
        """Reset game state"""
        pass

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

            # Screenshot mode: capture after a few frames
            if self.screenshot_mode:
                self.frame_count += 1
                if self.frame_count >= 60:  # After ~1 second
                    print(f"💾 Saving screenshot to: {{self.screenshot_file}}")
                    pygame.image.save(self.screen, self.screenshot_file)
                    print(f"✅ Screenshot saved successfully!")
                    self.running = False

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    if args.screenshot:
        print(f"📸 Screenshot mode: {{args.screenshot}}")
        game = Game(screenshot_mode=True, screenshot_file=args.screenshot)
    else:
        print("🎮 Interactive mode")
        game = Game()

    game.run()
'''


class GameAssemblyAgent:
    """Assemble assets and logic into a runnable game."""

    def __init__(
        self,
        llm_name: str = "claude-sonnet-4-5",
        use_miniswe: bool = True,
        agent_step_limit: int = 80,
        agent_cost_limit: float = 8.0,
        agent_timeout: int = 120
    ):
        self.code_generator = GameCodeGenerator(
            llm_name=llm_name,
            use_miniswe=use_miniswe,
            step_limit=agent_step_limit,
            cost_limit=agent_cost_limit,
            timeout=agent_timeout
        )

    async def assemble_game(
        self,
        benchmark_path: Optional[Path],
        analysis: Dict[str, Any],
        generated_assets: Dict[str, str],  # {asset_id: image_base64}
        output_dir: Path,
        enable_code_generation: bool = True
    ) -> Dict[str, Any]:
        """
        Assemble a fully runnable game.

        Args:
            benchmark_path: Benchmark path (None in instruction mode)
            analysis: Analysis results (game design info)
            generated_assets: Generated assets
            output_dir: Output directory
            enable_code_generation: Whether to invoke LLM code generation

        Returns:
            {
                'success': bool,
                'game_dir': Path,
                'entry_point': Path,
                'assets_count': int
            }
        """

        print(f"\n🎮 Game Assembly Agent")

        # Determine benchmark vs instruction mode
        is_instruction_mode = (benchmark_path is None)

        if is_instruction_mode:
            # Instruction mode: derive from analysis
            print(f"   Mode: Instruction-based")
            benchmark_display_name = analysis.get('game_name', 'CustomGame')
            benchmark_data = self._convert_analysis_to_benchmark_data(analysis)
            benchmark_root = None
            benchmark_file = None
        else:
            # Benchmark mode: load from files
            print(f"   Mode: Benchmark-based")
            benchmark_display_name = benchmark_path.name
            benchmark_file = benchmark_path

            if benchmark_path.is_dir():
                # Prefer config.yaml, else any .yaml file
                config_yaml = benchmark_path / "config.yaml"
                if config_yaml.exists():
                    benchmark_file = config_yaml
                else:
                    yaml_files = sorted(benchmark_path.glob("*.yaml"))
                    if yaml_files:
                        benchmark_file = yaml_files[0]
                    else:
                        raise FileNotFoundError(
                            f"No YAML benchmark config found in directory: {benchmark_path}"
                        )

                benchmark_display_name = benchmark_path.name
            else:
                benchmark_display_name = benchmark_path.stem

            benchmark_root = benchmark_file.parent

            # Load benchmark data
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                benchmark_data = yaml.safe_load(f)

        print(f"   Game: {benchmark_display_name}")
        print(f"   Assets: {len(generated_assets)}")
        print(f"   Output: {output_dir}\n")

        try:
            # 1. Extract metadata
            print("   📋 Step 1: Extracting metadata...")
            if is_instruction_mode:
                metadata = self._extract_metadata_from_analysis(analysis)
            else:
                metadata = self._extract_metadata(benchmark_data)

            print(f"      Game: {metadata.name}")
            print(f"      Type: {metadata.game_type}")

            # 2. Create game directory structure
            print("\n   📁 Step 2: Creating game directory...")
            game_dir = output_dir / "assembled_game"
            game_dir.mkdir(exist_ok=True)

            assets_dir = game_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            print(f"      Created: {game_dir.relative_to(output_dir)}")

            # 3. Copy assets
            print("\n   🎨 Step 3: Copying assets...")
            asset_count = self._copy_assets(generated_assets, assets_dir)
            print(f"      Copied {asset_count} assets")

            # 4. Copy benchmark levels (benchmark mode only)
            if not is_instruction_mode:
                print("\n   🗺️  Step 4: Copying benchmark levels...")
                level_file_count = self._copy_benchmark_levels(benchmark_root, game_dir)
                print(f"      Copied {level_file_count} level files")
            else:
                print("\n   🗺️  Step 4: Skipping benchmark levels (instruction mode)")
                level_file_count = 0

            # 5. Generate game code
            print("\n   💻 Step 5: Generating game code...")
            environment_artifacts = self._collect_environment_artifacts(game_dir, benchmark_data)
            benchmark_references = self._gather_benchmark_references(benchmark_root) if benchmark_root else []

            if enable_code_generation:
                game_code = await self.code_generator.generate_game_code(
                    benchmark_data,
                    list(generated_assets.keys()),
                    metadata,
                    environment_artifacts,
                    workspace_dir=game_dir,
                    benchmark_reference_paths=benchmark_references,
                    benchmark_root=benchmark_root
                )
            else:
                print("      Using basic template (LLM disabled)")
                game_code = self.code_generator._generate_fallback_code(
                    metadata,
                    list(generated_assets.keys())
                )

            # 6. Save game code
            game_file = game_dir / "game.py"
            with open(game_file, 'w', encoding='utf-8') as f:
                f.write(game_code)

            print(f"      Saved: {game_file.name}")

            # 7. Generate support files
            print("\n   📝 Step 6: Creating support files...")
            self._create_readme(game_dir, metadata)
            self._create_requirements(game_dir)
            print("      Created: README.md, requirements.txt")

            # 8. Validate import (syntax check)
            print("\n   🔍 Step 7: Validating code...")
            is_valid, error = self._validate_code(game_file)

            if is_valid:
                print("      ✅ Code validation passed")
            else:
                print(f"      ⚠️  Code validation warning: {error}")

            print(f"\n   ✅ Assembly Complete!")
            print(f"\n   To run the game:")
            print(f"      cd {game_dir.relative_to(output_dir.parent)}")
            print(f"      python game.py")

            return {
                'success': True,
                'game_dir': game_dir,
                'entry_point': game_file,
                'assets_count': asset_count,
                'code_valid': is_valid
            }

        except Exception as e:
            print(f"\n   ❌ Assembly failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'success': False,
                'error': str(e)
            }

    def _extract_metadata(self, benchmark_data: Dict[str, Any]) -> GameMetadata:
        """Extract metadata from a benchmark definition."""

        meta = benchmark_data.get('meta', {})
        state_template = benchmark_data.get('state_template', {})

        # Pull game name and description
        name = meta.get('name', 'Unknown Game')
        description = meta.get('description', '')

        # Infer game type
        game_type = 'unknown'
        grid_size = None

        # Check for card game
        if 'card' in description.lower() or 'memory' in description.lower():
            game_type = 'card_game'
            if 'board' in state_template and 'size' in state_template['board']:
                grid_size = tuple(state_template['board']['size'])

        # Check for grid-based game
        elif 'grid' in description.lower() or 'tile' in description.lower():
            game_type = 'grid_based'
            # Try to derive grid size from state_template
            if 'globals' in state_template and 'grid_size' in state_template['globals']:
                size = state_template['globals']['grid_size']
                grid_size = (size, size) if isinstance(size, int) else tuple(size)

        # Extract controls
        controls = []
        if 'transition' in benchmark_data and 'actions' in benchmark_data['transition']:
            actions = benchmark_data['transition']['actions']
            controls = [action.get('name', '') for action in actions]

        return GameMetadata(
            name=name,
            description=description,
            game_type=game_type,
            grid_size=grid_size,
            controls=controls
        )

    def _extract_metadata_from_analysis(self, analysis: Dict[str, Any]) -> GameMetadata:
        """Extract metadata from instruction-mode analysis."""

        game_design = analysis.get('game_design', {})

        name = analysis.get('game_name', 'Custom Game')
        description = game_design.get('core_mechanics', analysis.get('user_instruction', ''))
        game_type = analysis.get('game_type', 'unknown')

        # Pull grid size if present
        grid_size = None
        difficulty_levels = game_design.get('difficulty_levels', [])
        if difficulty_levels and len(difficulty_levels) > 0:
            # Use mid difficulty parameters
            mid_level = difficulty_levels[len(difficulty_levels) // 2]
            params = mid_level.get('parameters', {})
            if 'grid_size' in params:
                size_str = params['grid_size']
                if 'x' in size_str:
                    parts = size_str.split('x')
                    grid_size = (int(parts[0]), int(parts[1]))

        # Extract controls
        controls = []
        action_space = game_design.get('action_space', [])
        for action in action_space:
            action_name = action.get('action', '')
            action_desc = action.get('description', '')
            if action_name:
                controls.append(f"{action_name}: {action_desc}")

        return GameMetadata(
            name=name,
            description=description,
            game_type=game_type,
            grid_size=grid_size,
            controls=controls if controls else ['Arrow keys: Move', 'R: Reset', 'ESC: Quit']
        )

    def _convert_analysis_to_benchmark_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert analysis into benchmark_data format for code generation."""

        game_design = analysis.get('game_design', {})

        # Build a simplified benchmark_data structure
        benchmark_data = {
            'meta': {
                'name': analysis.get('game_name', 'Custom Game'),
                'description': game_design.get('core_mechanics', ''),
                'game_type': analysis.get('game_type', 'unknown')
            },
            'state_template': {
                'globals': {}
            },
            'transition': {
                'actions': []
            },
            'termination': {},
            'game_design': game_design,  # keep full game design info
            'visual_theme': analysis.get('visual_theme', ''),
            'art_style': analysis.get('art_style', ''),
            'required_assets': analysis.get('required_assets', [])
        }

        # Add state space info
        state_space = game_design.get('state_space', {})
        if state_space:
            benchmark_data['state_template']['globals'] = state_space

        # Add action space info
        action_space = game_design.get('action_space', [])
        for action in action_space:
            benchmark_data['transition']['actions'].append({
                'name': action.get('action', ''),
                'description': action.get('description', '')
            })

        # Add difficulty levels
        difficulty_levels = game_design.get('difficulty_levels', [])
        if difficulty_levels:
            benchmark_data['difficulty_levels'] = difficulty_levels

        return benchmark_data

    def _copy_assets(
        self,
        assets: Dict[str, str],
        assets_dir: Path
    ) -> int:
        """Copy assets into the game directory."""

        count = 0

        for asset_id, image_b64 in assets.items():
            asset_path = assets_dir / f"{asset_id}.png"

            # Decode and save
            img_bytes = base64.b64decode(image_b64)
            with open(asset_path, 'wb') as f:
                f.write(img_bytes)

            count += 1

        return count

    def _copy_benchmark_levels(self, benchmark_root: Path, game_dir: Path) -> int:
        """Copy benchmark levels and config files."""

        copied_yaml = 0
        for folder_name in ("levels", "val_levels"):
            source_dir = benchmark_root / folder_name
            if source_dir.is_dir():
                target_dir = game_dir / folder_name
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                copied_yaml += sum(1 for _ in target_dir.glob("*.yaml"))

        benchmark_config = benchmark_root / "config.yaml"
        if benchmark_config.exists():
            shutil.copy2(benchmark_config, game_dir / "config.yaml")

        return copied_yaml

    def _gather_benchmark_references(self, benchmark_root: Path) -> List[Path]:
        """Collect benchmark source files for code-generation context."""

        references: List[Path] = []
        candidates = [
            "env_main.py",
            "env_main_use.py",
            "env_obs.py",
            "env_generate.py",
            "env_code.py",
            "agent_instruction.txt",
            "action_space.txt",
            "env_desc.txt",
            "config.yaml"
        ]

        for relative in candidates:
            path = benchmark_root / relative
            if path.exists() and path not in references:
                references.append(path)

        for folder_name in ("levels", "val_levels"):
            folder = benchmark_root / folder_name
            if not folder.is_dir():
                continue
            for idx, level_file in enumerate(sorted(folder.glob("*.yaml"))):
                if level_file not in references:
                    references.append(level_file)
                if idx >= 1:
                    break

        return references

    def _collect_environment_artifacts(
        self,
        game_dir: Path,
        benchmark_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}

        globals_state = benchmark_data.get('state_template', {}).get('globals', {})
        actions = [action.get('name') for action in benchmark_data.get('transition', {}).get('actions', []) if action]

        if globals_state:
            artifacts['globals'] = {
                'grid_size': globals_state.get('grid_size'),
                'start_pos': globals_state.get('start_pos'),
                'exit_pos': globals_state.get('exit_pos'),
                'item_categories': globals_state.get('item_categories'),
                'dimensions': globals_state.get('dimensions'),
                'max_steps': globals_state.get('max_steps')
            }

        if actions:
            artifacts['actions'] = actions

        termination = benchmark_data.get('termination', {})
        if termination:
            artifacts['termination'] = {
                'max_steps': termination.get('max_steps')
            }

        level_sets: Dict[str, List[Dict[str, Any]]] = {}
        for folder_name in ("levels", "val_levels"):
            folder_path = game_dir / folder_name
            if not folder_path.is_dir():
                continue

            level_entries: List[Dict[str, Any]] = []
            for level_file in sorted(folder_path.glob("*.yaml")):
                entry: Dict[str, Any] = {'file': f"{folder_name}/{level_file.name}"}
                try:
                    with open(level_file, 'r', encoding='utf-8') as handle:
                        level_data = yaml.safe_load(handle) or {}

                    level_globals = level_data.get('globals', {})
                    maze_state = level_data.get('maze', {})

                    if level_globals:
                        entry['grid_size'] = level_globals.get('grid_size')
                        entry['start_pos'] = level_globals.get('start_pos')
                        entry['exit_pos'] = level_globals.get('exit_pos')
                        entry['max_steps'] = level_globals.get('max_steps')
                        entry['item_categories'] = level_globals.get('item_categories')
                        entry['dimensions'] = level_globals.get('dimensions')

                    probabilities = maze_state.get('wall_probabilities', {}) or {}
                    prob_values: List[float] = []
                    for value in probabilities.values():
                        try:
                            prob_values.append(float(value))
                        except (TypeError, ValueError):
                            continue

                    if prob_values:
                        entry['wall_probability_range'] = [round(min(prob_values), 3), round(max(prob_values), 3)]
                        entry['wall_probability_cells'] = len(prob_values)

                    collapsed = maze_state.get('collapsed_walls', {}) or {}
                    if collapsed:
                        entry['collapsed_walls'] = len(collapsed)

                    entry['guaranteed_open'] = self._identify_zero_probability_cells(probabilities, level_globals)

                    agent_state = level_data.get('agent', {})
                    if agent_state:
                        agent_entry: Dict[str, Any] = {}
                        for key in ('inventory', 'ledgers', 'total_profit', 'accumulated_rewards', 'discoveries'):
                            if key in agent_state:
                                agent_entry[key] = agent_state[key]
                        if agent_entry:
                            agent_entry['episode_complete'] = bool(level_data.get('episode_complete', False))
                            agent_entry['timestep'] = level_data.get('timestep', 0)
                            entry['agent'] = agent_entry

                    market_state = level_data.get('market', {})
                    if market_state:
                        market_entry: Dict[str, Any] = {}
                        for key in ('embargo_risks', 'hedge_status', 'exchange_rates'):
                            if key in market_state:
                                market_entry[key] = market_state[key]
                        exchange_matrices = market_state.get('exchange_matrices')
                        if exchange_matrices:
                            market_entry['exchange_matrices'] = exchange_matrices
                        if market_entry:
                            entry['market'] = market_entry

                except Exception as exc:
                    entry['load_error'] = str(exc)

                level_entries.append(entry)

            if level_entries:
                level_sets[folder_name] = level_entries

        if level_sets:
            artifacts['levels'] = level_sets

        return artifacts

    @staticmethod
    def _identify_zero_probability_cells(
        probabilities: Dict[str, Any],
        level_globals: Dict[str, Any]
    ) -> List[List[int]]:
        zero_cells: List[List[int]] = []
        if not probabilities:
            return zero_cells

        important_cells = []
        if level_globals:
            for key in ('start_pos', 'exit_pos'):
                value = level_globals.get(key)
                if isinstance(value, list) and len(value) == 2:
                    important_cells.append(value)

        for cell in important_cells:
            key = f"{cell[0]},{cell[1]}"
            try:
                if float(probabilities.get(key, 1.0)) == 0.0:
                    zero_cells.append(cell)
            except (TypeError, ValueError):
                continue

        return zero_cells

    def _create_readme(self, game_dir: Path, metadata: GameMetadata):
        """Create README file."""

        readme_content = f"""# {metadata.name}

{metadata.description}

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python game.py
```

## Controls

{chr(10).join(f'- {control}' for control in (metadata.controls or ['ESC - Quit', 'R - Restart']))}

## Game Info

- Type: {metadata.game_type}
{f'- Grid Size: {metadata.grid_size[0]}x{metadata.grid_size[1]}' if metadata.grid_size else ''}

## Credits

- Assets: AI-generated via Claude Code visualizer pipeline
- Code: Auto-generated pygame implementation

---

*This game was automatically assembled from AI-generated assets.*
"""

        with open(game_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def _create_requirements(self, game_dir: Path):
        """Create requirements.txt."""

        requirements = """pygame>=2.5.0
"""

        with open(game_dir / "requirements.txt", 'w', encoding='utf-8') as f:
            f.write(requirements)

    def _validate_code(self, code_file: Path) -> tuple[bool, Optional[str]]:
        """Validate code syntax."""

        try:
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read()

            # Check syntax
            compile(code, str(code_file), 'exec')
            return True, None

        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)


# Test helper
async def test_assembly():
    """Test game assembly."""
    # Add manual test code here if needed
    pass


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_assembly())
