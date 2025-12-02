"""
Assembly Agent - level assembly and visualization.
Assemble generated assets into full levels with preview and QA.
"""

import base64
import yaml
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from base.engine.async_llm import AsyncLLM


class LevelRenderer:
    """Level renderer - render YAML layouts into PNG images."""

    def __init__(self, tile_size: int = 256):
        """
        Args:
            tile_size: Pixel size of a single tile
        """
        self.tile_size = tile_size

    def render_level(
        self,
        level_data: Dict[str, Any],
        assets: Dict[str, str],  # {asset_id: image_base64}
        asset_mapping: Dict[str, str]  # {symbol: asset_id} e.g. {'#': 'wall', '.': 'floor'}
    ) -> Dict[str, Any]:
        """
        Render a full level.

        Args:
            level_data: Level data loaded from YAML
            assets: Generated asset dict
            asset_mapping: Symbol-to-asset mapping

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'width': int,
                'height': int,
                'grid_size': tuple,
                'missing_assets': list
            }
        """

        try:
            # Extract grid layout
            grid = self._extract_grid(level_data)
            if not grid:
                return {'success': False, 'error': 'No grid found in level data'}

            rows = len(grid)
            cols = len(grid[0]) if rows > 0 else 0

            if rows == 0 or cols == 0:
                return {'success': False, 'error': 'Empty grid'}

            print(f"   📐 Grid size: {cols}x{rows}")

            # Create canvas
            canvas_width = cols * self.tile_size
            canvas_height = rows * self.tile_size
            canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

            # Track missing assets
            missing_assets = set()

            # Render cell by cell
            for row_idx, row in enumerate(grid):
                for col_idx, symbol in enumerate(row):
                    # Skip blanks
                    if symbol.strip() == '':
                        continue

                    # Look up asset
                    asset_id = asset_mapping.get(symbol)

                    if not asset_id:
                        missing_assets.add(f"No mapping for symbol '{symbol}'")
                        continue

                    if asset_id not in assets:
                        missing_assets.add(f"Missing asset: {asset_id}")
                        continue

                    # Load asset image
                    asset_img = self._decode_image(assets[asset_id])

                    # Ensure correct size
                    if asset_img.size != (self.tile_size, self.tile_size):
                        asset_img = asset_img.resize(
                            (self.tile_size, self.tile_size),
                            Image.Resampling.LANCZOS
                        )

                    # Paste onto canvas
                    x = col_idx * self.tile_size
                    y = row_idx * self.tile_size
                    canvas.paste(asset_img, (x, y), asset_img)

            # Encode as base64
            canvas_b64 = self._encode_image(canvas)

            print(f"   ✅ Rendered: {canvas_width}x{canvas_height}px ({cols}x{rows} tiles)")
            if missing_assets:
                print(f"   ⚠️  Missing {len(missing_assets)} assets")

            return {
                'success': True,
                'image_base64': canvas_b64,
                'width': canvas_width,
                'height': canvas_height,
                'grid_size': (cols, rows),
                'missing_assets': list(missing_assets)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def render_sprite_showcase(
        self,
        assets: Dict[str, str],
        asset_specs: List[Dict[str, Any]],
        cols: int = 4
    ) -> Dict[str, Any]:
        """
        （）

        Args:
            assets: 
            asset_specs: （ strategy）
            cols: 

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'width': int,
                'height': int
            }
        """

        try:
            asset_count = len(assets)
            if asset_count == 0:
                return {'success': False, 'error': 'No assets provided'}

            # 
            rows = (asset_count + cols - 1) // cols

            # cell: tile_size + padding + label
            cell_size = self.tile_size + 40  # 40px
            padding = 10

            canvas_width = cols * (cell_size + padding) + padding
            canvas_height = rows * (cell_size + padding) + padding

            # （）
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            # ，
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

            # 
            for idx, (asset_id, asset_b64) in enumerate(assets.items()):
                row = idx // cols
                col = idx % cols

                x = padding + col * (cell_size + padding)
                y = padding + row * (cell_size + padding)

                # 
                asset_img = self._decode_image(asset_b64)

                # Resize if needed
                if asset_img.size != (self.tile_size, self.tile_size):
                    asset_img = asset_img.resize(
                        (self.tile_size, self.tile_size),
                        Image.Resampling.LANCZOS
                    )

                # RGB（RGBA）
                if asset_img.mode == 'RGBA':
                    # 
                    bg = self._create_checkerboard(self.tile_size, self.tile_size)
                    bg.paste(asset_img, (0, 0), asset_img)
                    asset_img = bg

                # 
                canvas.paste(asset_img, (x, y))

                # 
                draw.rectangle(
                    [x, y, x + self.tile_size, y + self.tile_size],
                    outline=(200, 200, 200),
                    width=1
                )

                # 
                label = asset_id[:20]  # 
                label_y = y + self.tile_size + 5
                draw.text((x + 5, label_y), label, fill=(0, 0, 0), font=font)

            canvas_b64 = self._encode_image(canvas)

            print(f"   ✅ Sprite showcase: {canvas_width}x{canvas_height}px ({asset_count} assets)")

            return {
                'success': True,
                'image_base64': canvas_b64,
                'width': canvas_width,
                'height': canvas_height,
                'asset_count': asset_count
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_grid(self, level_data: Dict[str, Any]) -> List[List[str]]:
        """
         YAML 

        ：
        1. grid: ["####", "#..#", "####"]
        2. layout: 
        3. map: 
        4. state_template.board.size: [4, 4] ()
        """

        #  1: 
        for field in ['grid', 'layout', 'map', 'level']:
            if field in level_data:
                grid_raw = level_data[field]

                # 
                if isinstance(grid_raw, list) and all(isinstance(s, str) for s in grid_raw):
                    return [list(row) for row in grid_raw]

                # （）
                if isinstance(grid_raw, str):
                    lines = grid_raw.strip().split('\n')
                    return [list(line) for line in lines]

        #  2:  state_template （）
        if 'state_template' in level_data:
            state = level_data['state_template']

            #  board.size
            if 'board' in state and 'size' in state['board']:
                size = state['board']['size']
                if isinstance(size, list) and len(size) == 2:
                    cols, rows = size
                    # （ 'C'  card）
                    return [['C'] * cols for _ in range(rows)]

            #  globals.grid_size
            if 'globals' in state and 'grid_size' in state['globals']:
                grid_size = state['globals']['grid_size']
                if isinstance(grid_size, int):
                    #  NxN 
                    return [['C'] * grid_size for _ in range(grid_size)]

        #  3:  meta 
        if 'meta' in level_data:
            meta = level_data['meta']
            #  meta 
            for size_key in ['grid_size', 'board_size', 'size']:
                if size_key in meta:
                    size_val = meta[size_key]
                    if isinstance(size_val, int):
                        return [['C'] * size_val for _ in range(size_val)]
                    elif isinstance(size_val, list) and len(size_val) == 2:
                        cols, rows = size_val
                        return [['C'] * cols for _ in range(rows)]

        return []

    def _create_checkerboard(self, width: int, height: int, cell_size: int = 8) -> Image.Image:
        """（）"""

        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                if (x // cell_size + y // cell_size) % 2 == 1:
                    draw.rectangle(
                        [x, y, x + cell_size, y + cell_size],
                        fill=(220, 220, 220)
                    )

        return img

    def _decode_image(self, image_b64: str) -> Image.Image:
        """ base64 """
        img_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(img_bytes))

    def _encode_image(self, img: Image.Image) -> str:
        """ base64"""
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()


class VisualQAAnalyzer:
    """ QA  - """

    def __init__(self, llm_name: str = "claude-sonnet-4-5"):
        """
        Args:
            llm_name:  LLM 
        """
        self.llm = AsyncLLM(llm_name)

    async def analyze_level(
        self,
        level_image_b64: str,
        showcase_image_b64: str,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ，

        Args:
            level_image_b64: 
            showcase_image_b64: 
            strategy: （art style）

        Returns:
            {
                'success': bool,
                'issues': [
                    {
                        'severity': 'critical|warning|info',
                        'category': 'style_inconsistency|contrast|missing|quality',
                        'description': str,
                        'suggestion': str
                    }
                ],
                'overall_quality': 'excellent|good|fair|poor',
                'summary': str
            }
        """

        print("\n   🔍 Visual QA Analysis...")

        #  prompt
        art_style = strategy.get('rendering_approach', {}).get('art_style', 'Unknown')
        theme = strategy.get('visual_theme', 'Unknown')

        content = [
            {
                "type": "text",
                "text": "# Image 1: Complete Level Preview"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{level_image_b64}"}
            },
            {
                "type": "text",
                "text": "# Image 2: Asset Showcase (All Generated Assets)"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{showcase_image_b64}"}
            },
            {
                "type": "text",
                "text": self._build_qa_prompt(art_style, theme)
            }
        ]

        try:
            response = await self.llm(content)

            # 
            issues = self._parse_qa_response(response)

            # 
            overall = self._calculate_quality(issues)

            print(f"   📊 Overall Quality: {overall.upper()}")
            print(f"   🔍 Issues Found: {len(issues)}")

            for issue in issues:
                icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'warning' else "🔵"
                print(f"   {icon} [{issue['category']}] {issue['description'][:60]}...")

            return {
                'success': True,
                'issues': issues,
                'overall_quality': overall,
                'summary': self._create_summary(issues, overall)
            }

        except Exception as e:
            print(f"   ❌ QA Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'issues': [],
                'overall_quality': 'unknown'
            }

    def _build_qa_prompt(self, art_style: str, theme: str) -> str:
        """ QA  prompt"""

        return f"""You are a game art quality control expert. Analyze the generated game assets for visual issues.

**Target Specifications:**
- Art Style: {art_style}
- Theme: {theme}

**Your Task:**
Examine both images and identify visual problems in these categories:

1. **Style Inconsistency** (CRITICAL)
   - Do all assets follow the same art style?
   - Are there assets that look like they're from different games?
   - Is the visual language consistent?

2. **Contrast Issues** (CRITICAL)
   - Can the player/characters be clearly distinguished from the background?
   - Are there visibility problems (e.g., dark sprites on dark backgrounds)?
   - Is important gameplay information visible?

3. **Missing/Broken Assets** (CRITICAL)
   - Are there any obvious placeholder or missing textures?
   - Are there visual glitches or artifacts?

4. **Quality Issues** (WARNING)
   - Do assets match the intended art style?
   - Is the resolution/detail appropriate?
   - Are there any aesthetic problems?

5. **Composition Issues** (INFO)
   - Does the overall level look balanced?
   - Is there visual hierarchy?
   - Does it look appealing?

**Output Format:**
For each issue found, output in this JSON array format:

```json
[
  {{
    "severity": "critical|warning|info",
    "category": "style_inconsistency|contrast|missing|quality|composition",
    "description": "Brief description of the problem",
    "suggestion": "How to fix it"
  }}
]
```

If no issues found, output: `[]`

**Important:**
- Be objective and specific
- Focus on problems that affect gameplay or visual coherence
- Provide actionable suggestions
- If everything looks good, say so!

Output ONLY the JSON array, no other text:"""

    def _parse_qa_response(self, response: str) -> List[Dict[str, Any]]:
        """ LLM  QA """

        import re

        #  JSON 
        json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            try:
                issues = json.loads(json_match.group(1))
                return issues if isinstance(issues, list) else []
            except json.JSONDecodeError:
                pass

        #  JSON 
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                issues = json.loads(json_match.group(0))
                return issues if isinstance(issues, list) else []
            except json.JSONDecodeError:
                pass

        # ""
        if any(phrase in response.lower() for phrase in ['no issues', 'looks good', 'everything is fine', 'no problems']):
            return []

        # Fallback: 
        return []

    def _calculate_quality(self, issues: List[Dict[str, Any]]) -> str:
        """"""

        if not issues:
            return 'excellent'

        critical_count = sum(1 for i in issues if i['severity'] == 'critical')
        warning_count = sum(1 for i in issues if i['severity'] == 'warning')

        if critical_count >= 3:
            return 'poor'
        elif critical_count >= 1:
            return 'fair'
        elif warning_count >= 3:
            return 'fair'
        else:
            return 'good'

    def _create_summary(self, issues: List[Dict[str, Any]], quality: str) -> str:
        """"""

        if not issues:
            return "No visual issues detected. All assets look consistent and appropriate."

        critical = [i for i in issues if i['severity'] == 'critical']
        warnings = [i for i in issues if i['severity'] == 'warning']

        summary = f"Overall quality: {quality.upper()}. "

        if critical:
            summary += f"{len(critical)} critical issue(s) found: "
            summary += ", ".join(i['category'] for i in critical[:3])
            if len(critical) > 3:
                summary += f" and {len(critical) - 3} more"
            summary += ". "

        if warnings:
            summary += f"{len(warnings)} warning(s). "

        return summary


class AssemblyAgent:
    """ Agent -  QA"""

    def __init__(
        self,
        tile_size: int = 256,
        llm_name: str = "claude-sonnet-4-5"
    ):
        """
        Args:
            tile_size: Tile 
            llm_name: QA  LLM
        """
        self.renderer = LevelRenderer(tile_size)
        self.qa_analyzer = VisualQAAnalyzer(llm_name)

    async def assemble_and_verify(
        self,
        benchmark_path: Path,
        strategy: Dict[str, Any],
        generated_assets: Dict[str, str],  # {asset_id: image_b64}
        output_dir: Path
    ) -> Dict[str, Any]:
        """
         QA

        Args:
            benchmark_path: Benchmark YAML 
            strategy: 
            generated_assets: 
            output_dir: 

        Returns:
            {
                'success': bool,
                'level_preview_path': Path,
                'showcase_path': Path,
                'qa_report': Dict,
                'requires_iteration': bool
            }
        """

        print(f"\n📦 Assembly & Verification")
        print(f"   Benchmark: {benchmark_path.name}")
        print(f"   Assets: {len(generated_assets)}")

        # 1.  benchmark
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            level_data = yaml.safe_load(f)

        # 2. （ strategy ）
        asset_mapping = self._create_asset_mapping(strategy, level_data)

        # 3. 
        print("\n   🎨 Rendering level preview...")
        level_result = self.renderer.render_level(
            level_data,
            generated_assets,
            asset_mapping
        )

        if not level_result['success']:
            return {
                'success': False,
                'error': f"Level rendering failed: {level_result.get('error')}"
            }

        # 4. 
        print("\n   🎨 Rendering asset showcase...")
        asset_specs = strategy.get('assets', [])
        showcase_result = self.renderer.render_sprite_showcase(
            generated_assets,
            asset_specs
        )

        if not showcase_result['success']:
            return {
                'success': False,
                'error': f"Showcase rendering failed: {showcase_result.get('error')}"
            }

        # 5. 
        level_preview_path = output_dir / "level_preview.png"
        showcase_path = output_dir / "asset_showcase.png"

        self._save_image(level_result['image_base64'], level_preview_path)
        self._save_image(showcase_result['image_base64'], showcase_path)

        print(f"\n   💾 Saved preview: {level_preview_path.name}")
        print(f"   💾 Saved showcase: {showcase_path.name}")

        # 6.  QA
        qa_result = await self.qa_analyzer.analyze_level(
            level_result['image_base64'],
            showcase_result['image_base64'],
            strategy
        )

        # 7.  QA 
        qa_report_path = output_dir / "qa_report.json"
        with open(qa_report_path, 'w', encoding='utf-8') as f:
            json.dump(qa_result, f, indent=2, ensure_ascii=False)

        print(f"   💾 Saved QA report: {qa_report_path.name}")

        # 8. 
        requires_iteration = False
        if qa_result.get('success'):
            critical_issues = [
                i for i in qa_result['issues']
                if i['severity'] == 'critical'
            ]
            requires_iteration = len(critical_issues) > 0

        return {
            'success': True,
            'level_preview_path': level_preview_path,
            'showcase_path': showcase_path,
            'qa_report_path': qa_report_path,
            'qa_result': qa_result,
            'requires_iteration': requires_iteration,
            'missing_assets': level_result.get('missing_assets', [])
        }

    def _create_asset_mapping(
        self,
        strategy: Dict[str, Any],
        level_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        

        ：
        1. strategy  symbol_mapping（）
        2. level_data  legend（）
        3. （）
        """

        mapping = {}

        #  1:  strategy 
        if 'symbol_mapping' in strategy:
            mapping.update(strategy['symbol_mapping'])

        #  2:  level_data  legend 
        if 'legend' in level_data:
            legend = level_data['legend']
            for symbol, description in legend.items():
                #  assets 
                for asset in strategy.get('assets', []):
                    asset_id = asset.get('id', asset.get('name', ''))
                    asset_name = asset.get('name', '').lower()
                    desc_lower = description.lower()

                    # 
                    if asset_name in desc_lower or desc_lower in asset_name:
                        mapping[symbol] = asset_id
                        break

        #  3: 
        if not mapping:
            # 
            default_mappings = {
                '#': 'wall',
                '.': 'floor',
                '@': 'player',
                'E': 'enemy',
                'T': 'treasure',
                'G': 'goal'
            }

            # 
            asset_ids = {asset.get('id', asset.get('name')) for asset in strategy.get('assets', [])}

            for symbol, default_id in default_mappings.items():
                # 
                for asset_id in asset_ids:
                    if default_id in asset_id.lower():
                        mapping[symbol] = asset_id
                        break

        return mapping

    def _save_image(self, image_b64: str, path: Path):
        """ base64 """
        img_bytes = base64.b64decode(image_b64)
        with open(path, 'wb') as f:
            f.write(img_bytes)


# 
async def test_assembly():
    """ Assembly Agent"""

    # 
    test_level = {
        'grid': [
            '####',
            '#..#',
            '#@.#',
            '####'
        ],
        'legend': {
            '#': 'wall',
            '.': 'floor',
            '@': 'player'
        }
    }

    test_strategy = {
        'visual_theme': 'Dungeon',
        'rendering_approach': {
            'art_style': 'pixel art'
        },
        'assets': [
            {'id': 'wall', 'name': 'Stone Wall'},
            {'id': 'floor', 'name': 'Stone Floor'},
            {'id': 'player', 'name': 'Hero'}
        ]
    }

    # （）
    def create_test_asset(color):
        img = Image.new('RGBA', (256, 256), color)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    test_assets = {
        'wall': create_test_asset((100, 100, 100, 255)),  # 
        'floor': create_test_asset((200, 200, 200, 255)),  # 
        'player': create_test_asset((255, 0, 0, 255))  # 
    }

    # 
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_level, f)
        temp_yaml = Path(f.name)

    temp_output = Path(tempfile.mkdtemp())

    # 
    agent = AssemblyAgent()
    result = await agent.assemble_and_verify(
        temp_yaml,
        test_strategy,
        test_assets,
        temp_output
    )

    print(f"\n✅ Test completed!")
    print(f"   Output: {temp_output}")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   QA Quality: {result['qa_result']['overall_quality']}")
        print(f"   Issues: {len(result['qa_result']['issues'])}")

    # 
    temp_yaml.unlink()


if __name__ == '__main__':
    import asyncio
    asyncio.run(test_assembly())
