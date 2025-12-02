"""
Visual Refinement Agent
Claude analysis + Gemini editing for image cleanup and enhancement.

Key features:
- Full history tracking (images and metadata per step)
- Context-aware decisions (sees the full processing flow)
- Up to 3 regeneration attempts with AI processing loop
- High-performance background removal
"""

import base64
import json
import re
import asyncio
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from base.engine.async_llm import AsyncLLM
from visualizer.agents.image_gen_agent import ImageGenAgent


@dataclass
class ProcessingStep:
    """Record of a single processing step."""
    step_number: int
    action_type: str
    action_params: Dict[str, Any]
    before_image_b64: str
    after_image_b64: str
    before_info: Dict[str, Any]
    after_info: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self):
        """Convert to dict without base64 images to save memory."""
        return {
            'step_number': self.step_number,
            'action_type': self.action_type,
            'action_params': self.action_params,
            'before_info': self.before_info,
            'after_info': self.after_info,
            'success': self.success,
            'error_message': self.error_message,
            'notes': self.notes
        }

    def get_summary(self) -> str:
        """Return a human-readable step summary."""
        before = self.before_info
        after = self.after_info

        summary = f"Step {self.step_number}: {self.action_type}"
        summary += f"\n  Before: {before['width']}x{before['height']}px, transparency={before['color_analysis'].get('transparency_percentage', 0)*100:.0f}%"
        summary += f"\n  After:  {after['width']}x{after['height']}px, transparency={after['color_analysis'].get('transparency_percentage', 0)*100:.0f}%"

        if not self.success:
            summary += f"\n  ❌ Failed: {self.error_message}"
        elif self.notes:
            summary += f"\n  ℹ️  {self.notes}"

        return summary


class ProcessingHistory:
    """Manage processing history."""

    def __init__(self):
        self.steps: List[ProcessingStep] = []
        self.current_image_b64: str = None

    def add_step(self, step: ProcessingStep):
        """Add a processing step."""
        self.steps.append(step)
        if step.success:
            self.current_image_b64 = step.after_image_b64

    def get_summary(self, max_steps: int = 5) -> str:
        """Get a summary of recent steps."""
        if not self.steps:
            return "No processing steps yet."

        recent_steps = self.steps[-max_steps:]
        summary = f"Processing History ({len(self.steps)} total steps, showing recent {len(recent_steps)}):\n"
        summary += "=" * 60 + "\n"

        for step in recent_steps:
            summary += step.get_summary() + "\n"
            summary += "-" * 60 + "\n"

        return summary

    def get_latest_images(self, count: int = 3) -> List[Dict[str, str]]:
        """Return the most recent images for multimodal input."""
        images = []
        for step in self.steps[-count:]:
            images.append({
                'step': step.step_number,
                'action': step.action_type,
                'image_b64': step.after_image_b64
            })
        return images


class VisualRefinementTools:
    """Image analysis and editing toolbox."""
    
    def __init__(self):
        """Initialize tools."""
        self.use_scipy = False
        try:
            import scipy
            self.use_scipy = True
        except ImportError:
            pass

    def _decode_image(self, image_b64: str) -> Image.Image:
        """Decode a base64 image."""
        img_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(img_bytes))

    def _encode_image(self, img: Image.Image) -> str:
        """Encode a PIL image as base64."""
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def get_image_info(self, image_b64: str) -> Dict[str, Any]:
        """Get detailed information about an image."""
        img = self._decode_image(image_b64)
        img_bytes = base64.b64decode(image_b64)

        # Analyze colors
        color_analysis = self._analyze_colors(img)

        return {
            "width": img.width,
            "height": img.height,
            "format": img.format or "PNG",
            "mode": img.mode,
            "has_transparency": img.mode in ("RGBA", "LA"),
            "file_size_kb": round(len(img_bytes) / 1024, 1),
            "color_analysis": color_analysis
        }

    def _analyze_colors(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze image color characteristics."""
        img_array = np.array(img)

        has_white_bg = False
        transparency_pct = 0.0

        if img.mode == "RGBA":
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]

            white_mask = np.all(rgb > 240, axis=2)
            has_white_bg = np.sum(white_mask) > (img.width * img.height * 0.1)

            transparent_pixels = np.sum(alpha == 0)
            transparency_pct = round(transparent_pixels / (img.width * img.height), 2)
        elif img.mode == "RGB":
            rgb = img_array
            white_mask = np.all(rgb > 240, axis=2)
            has_white_bg = np.sum(white_mask) > (img.width * img.height * 0.1)

        return {
            "has_white_background": bool(has_white_bg),
            "transparency_percentage": float(transparency_pct)
        }

    def composite_on_color(
        self,
        image_b64: str,
        background_color: tuple = (30, 30, 30, 255)
    ) -> str:
        """Composite onto a solid background to visualize transparency."""
        img = self._decode_image(image_b64)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        bg = Image.new('RGBA', img.size, background_color)
        bg.paste(img, (0, 0), img)
        return self._encode_image(bg)

    def composite_on_checkerboard(
        self,
        image_b64: str,
        tile_size: int = 32,
        light_color: tuple = (220, 220, 220, 255),
        dark_color: tuple = (160, 160, 160, 255)
    ) -> str:
        """Composite onto a checkerboard background to inspect transparency."""
        img = self._decode_image(image_b64)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        width, height = img.size
        checker = Image.new('RGBA', (width, height), light_color)

        tile = Image.new('RGBA', (tile_size * 2, tile_size * 2), light_color)
        dark_tile = Image.new('RGBA', (tile_size, tile_size), dark_color)
        tile.paste(dark_tile, (0, 0))
        tile.paste(dark_tile, (tile_size, tile_size))

        for y in range(0, height, tile_size * 2):
            for x in range(0, width, tile_size * 2):
                checker.paste(tile, (x, y))

        checker.paste(img, (0, 0), img)
        return self._encode_image(checker)

    def _detect_content_regions_fast(self, gray: np.ndarray, threshold: float = 10.0) -> np.ndarray:
        """
        🚀 Fast content-region detection using convolution instead of generic_filter.
        """
        if not self.use_scipy:
            return np.ones_like(gray, dtype=bool)
        
        from scipy import signal, ndimage
        
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        local_mean = signal.convolve2d(gray, kernel, mode='same', boundary='symm')
        local_mean_sq = signal.convolve2d(gray**2, kernel, mode='same', boundary='symm')
        
        local_var = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        content_mask = local_std > threshold
        
        struct = ndimage.generate_binary_structure(2, 1)
        content_mask = ndimage.binary_dilation(content_mask, structure=struct, iterations=3)
        content_mask = ndimage.binary_fill_holes(content_mask)
        content_mask = ndimage.binary_erosion(content_mask, structure=struct, iterations=2)
        
        return content_mask
    
    def _generate_smart_samples(
        self,
        width: int,
        height: int,
        content_mask: np.ndarray,
        sample_inner: bool = True
    ) -> List[tuple]:
        """🚀 Generate sample points efficiently to reduce redundancy."""
        sample_points = []
        seen = set()
        
        step = max(3, min(width, height) // 20)
        
        for x in range(0, width, step):
            if (x, 0) not in seen:
                sample_points.append((x, 0))
                seen.add((x, 0))
            if (x, height-1) not in seen:
                sample_points.append((x, height-1))
                seen.add((x, height-1))
        
        for y in range(0, height, step):
            if (0, y) not in seen:
                sample_points.append((0, y))
                seen.add((0, y))
            if (width-1, y) not in seen:
                sample_points.append((width-1, y))
                seen.add((width-1, y))
        
        if sample_inner:
            grid_step = max(width // 8, height // 8, 20)
            for y in range(grid_step, height - grid_step, grid_step):
                for x in range(grid_step, width - grid_step, grid_step):
                    if not content_mask[y, x]:
                        pt = (x, y)
                        if pt not in seen:
                            sample_points.append(pt)
                            seen.add(pt)
        
        return sample_points
    
    def _flood_fill_optimized(
        self,
        rgb: np.ndarray,
        seed_point: tuple,
        processed: np.ndarray,
        content_mask: np.ndarray,
        tolerance: int,
        max_pixels: int
    ) -> np.ndarray:
        """🚀 Optimized flood fill with early termination and content protection."""
        height, width = rgb.shape[:2]
        sy, sx = seed_point
        
        if processed[sy, sx] or content_mask[sy, sx]:
            return np.zeros((height, width), dtype=bool)
        
        seed_color = rgb[sy, sx]
        region_mask = np.zeros((height, width), dtype=bool)
        queue = deque([(sy, sx)])
        region_mask[sy, sx] = True
        processed[sy, sx] = True
        pixel_count = 1
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue and pixel_count < max_pixels:
            y, x = queue.popleft()
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                
                if processed[ny, nx] or content_mask[ny, nx]:
                    continue
                
                color_diff = np.abs(rgb[ny, nx] - seed_color).max()
                
                if color_diff <= tolerance:
                    processed[ny, nx] = True
                    region_mask[ny, nx] = True
                    queue.append((ny, nx))
                    pixel_count += 1
                    
                    if pixel_count > max_pixels:
                        break
        
        return region_mask

    def remove_background(
        self,
        image_b64: str,
        tolerance: int = 30,
        aggressive: bool = False
    ) -> Dict[str, Any]:
        """
        🔥 General background removal (optimized).

        Features:
        - Fast content detection (convolution)
        - Smart sampling (reduced redundancy)
        - Optimized flood fill (early exit)
        - Vectorized edge cleanup
        """
        try:
            img = self._decode_image(image_b64)
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            rgb = img_array[:, :, :3].astype(np.int16)
            alpha = img_array[:, :, 3].copy()
            
            total_pixels = height * width
            max_region_size = int(total_pixels * 0.85)
            
            # Step 1: Fast content detection
            gray = np.mean(rgb, axis=2)
            content_mask = self._detect_content_regions_fast(gray, threshold=10.0)
            
            # Step 2: Smart sampling
            sample_points = self._generate_smart_samples(
                width, height, content_mask, sample_inner=aggressive
            )
            
            # Step 3: Detect background type
            corner_colors = [rgb[0, 0], rgb[0, width-1], rgb[height-1, 0], rgb[height-1, width-1]]
            color_variance = np.std(corner_colors, axis=0).mean()
            bg_type = "gradient" if color_variance > 15 else "solid"
            
            print(f"      🎨 Background type: {bg_type}, {len(sample_points)} sample points")
            
            # Step 4: Optimized flood fill
            remove_mask = np.zeros((height, width), dtype=bool)
            processed = np.zeros((height, width), dtype=bool)
            regions_removed = 0
            
            for seed_x, seed_y in sample_points:
                sy = max(0, min(height - 1, seed_y))
                sx = max(0, min(width - 1, seed_x))
                
                region_mask = self._flood_fill_optimized(
                    rgb, (sy, sx), processed, content_mask,
                    tolerance, max_region_size
                )
                
                region_size = int(region_mask.sum())
                if region_size > 0:
                    remove_mask |= region_mask
                    regions_removed += 1
            
            pixels_removed = int(remove_mask.sum())
            
            # Step 5: Vectorized edge cleanup
            if pixels_removed > 0:
                alpha[remove_mask] = 0
                
                if self.use_scipy:
                    from scipy import ndimage
                    subject_mask = alpha > 10
                    
                    if subject_mask.any():
                        struct = np.ones((3, 3), dtype=bool)
                        dilated = ndimage.binary_dilation(subject_mask, structure=struct, iterations=2)
                        edge_region = dilated & ~subject_mask
                        
                        edge_y, edge_x = np.where(edge_region)
                        if len(edge_y) > 0:
                            avg_bg = np.mean(corner_colors, axis=0)
                            edge_colors = rgb[edge_y, edge_x]
                            color_diffs = np.abs(edge_colors - avg_bg).max(axis=1)
                            
                            to_remove = color_diffs <= 15
                            if to_remove.any():
                                alpha[edge_y[to_remove], edge_x[to_remove]] = 0
                                pixels_removed += int(to_remove.sum())
                
                img_array[:, :, 3] = alpha
                result_img = Image.fromarray(img_array, mode='RGBA')
                result_b64 = self._encode_image(result_img)
            else:
                result_b64 = image_b64
            
            print(f"      ✅ Removed {bg_type} background: {pixels_removed:,} pixels ({regions_removed} regions)")
            
            return {
                "success": pixels_removed > 0,
                "transparent_image_b64": result_b64,
                "pixels_made_transparent": pixels_removed,
                "background_type": bg_type,
                "regions_removed": regions_removed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def smart_resize(
        self,
        image_b64: str,
        target_size: int,
        method: str = "LANCZOS"
    ) -> Dict[str, Any]:
        """
        🔥 Smart resize: choose best strategy to avoid distortion.

        Logic:
        1) If square → direct resize
        2) If not square → resize to fit long edge + center pad
        """
        try:
            img = self._decode_image(image_b64)
            original_size = (img.width, img.height)
            
            resample_map = {
                "LANCZOS": Image.Resampling.LANCZOS,
                "NEAREST": Image.Resampling.NEAREST,
                "BILINEAR": Image.Resampling.BILINEAR,
                "BICUBIC": Image.Resampling.BICUBIC
            }
            resample = resample_map.get(method.upper(), Image.Resampling.LANCZOS)
            
            aspect_ratio = img.width / img.height
            is_square = 0.95 <= aspect_ratio <= 1.05
            
            if is_square:
                resized = img.resize((target_size, target_size), resample=resample)
                strategy = "direct"
                print(f"      ✅ Direct resize: {original_size} → {target_size}x{target_size}")
            else:
                scale = target_size / max(img.width, img.height)
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                fitted = img.resize((new_w, new_h), resample=resample)
                
                square = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
                paste_x = (target_size - new_w) // 2
                paste_y = (target_size - new_h) // 2
                square.paste(fitted, (paste_x, paste_y), fitted if fitted.mode == 'RGBA' else None)
                
                resized = square
                strategy = "fit_and_pad"
                print(f"      ✅ Fit+Pad: {original_size} → {new_w}x{new_h} → {target_size}x{target_size} (centered)")
            
            result_b64 = self._encode_image(resized)
            
            return {
                "success": True,
                "resized_image_b64": result_b64,
                "original_size": original_size,
                "new_size": (target_size, target_size),
                "strategy_used": strategy
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def auto_crop_content(
        self,
        image_b64: str,
        padding: int = 0
    ) -> Dict[str, Any]:
        """
        🔥 Auto-crop transparent/white edges, keep only content.

        Uses smart subject detection instead of plain getbbox().
        """
        try:
            img = self._decode_image(image_b64)
            original_size = (img.width, img.height)

            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                img_array = np.array(img)
                rgb = img_array[:, :, :3]
                white_mask = np.all(rgb >= 240, axis=2)
                alpha = img_array[:, :, 3].copy()
                alpha[white_mask] = 0
                img_array[:, :, 3] = alpha
                img = Image.fromarray(img_array, mode='RGBA')

            bbox = img.getbbox()
            detection_method = "traditional_bbox"

            if bbox is None:
                return {
                    "success": False,
                    "error": "Image is completely transparent"
                }

            # Smart padding
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            image_area = img.width * img.height
            subject_ratio = bbox_area / image_area if image_area > 0 else 0
            
            if subject_ratio < 0.3:
                smart_padding = max(padding, 8)
            elif subject_ratio > 0.7:
                smart_padding = max(padding, 2)
            else:
                smart_padding = max(padding, 5)

            left = max(0, bbox[0] - smart_padding)
            top = max(0, bbox[1] - smart_padding)
            right = min(img.width, bbox[2] + smart_padding)
            bottom = min(img.height, bbox[3] + smart_padding)

            cropped = img.crop((left, top, right, bottom))
            cropped_b64 = self._encode_image(cropped)

            print(f"      ✅ Auto-cropped: {original_size} → {cropped.size} (method: {detection_method}, padding: {smart_padding}px)")

            return {
                "success": True,
                "cropped_image_b64": cropped_b64,
                "original_size": original_size,
                "content_box": (left, top, right, bottom),
                "new_size": cropped.size,
                "detection_method": detection_method
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_tile_from_grid(
        self,
        image_b64: str,
        grid_size: tuple = (2, 2),
        tile_position: tuple = (0, 0)
    ) -> Dict[str, Any]:
        """Extract a single tile from a tile grid."""
        try:
            img = self._decode_image(image_b64)

            cols, rows = grid_size
            col, row = tile_position

            if col >= cols or row >= rows:
                return {
                    "success": False,
                    "error": f"Tile position {tile_position} out of bounds for grid {grid_size}"
                }

            tile_width = img.width // cols
            tile_height = img.height // rows

            left = col * tile_width
            top = row * tile_height
            right = left + tile_width
            bottom = top + tile_height

            tile = img.crop((left, top, right, bottom))
            tile_b64 = self._encode_image(tile)

            print(f"      ✅ Extracted tile [{col}, {row}] from {grid_size} grid: {tile.size}")

            return {
                "success": True,
                "tile_image_b64": tile_b64,
                "tile_size": tile.size
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def edit_with_gemini(
        self,
        image_b64: str,
        edit_instruction: str,
        gemini_agent: ImageGenAgent
    ) -> Dict[str, Any]:
        """Edit an image with Gemini."""
        prompt = f"""Image editing task:

{edit_instruction}

CRITICAL REQUIREMENTS:
- Preserve the original visual style and art direction
- Only make the requested changes, nothing more
- Maintain the same art style (pixel art / hand-drawn / etc.)
- If the original has transparency, preserve it
- Output high quality result suitable for game use
- White background for easy extraction if needed
"""

        print(f"      Gemini editing: {edit_instruction}...")

        result = await gemini_agent.generate_image_to_image(
            prompt=prompt,
            reference_images=[image_b64],
            preserve_history=False
        )

        return {
            "success": result.get('success', False),
            "edited_image_b64": result.get('image_base64'),
            "instruction_used": prompt,
            "error": result.get('error')
        }


class RefinementAgent:
    """Refinement agent powered by Claude analysis + Gemini editing."""

    def __init__(
        self,
        analyzer_llm: str = "claude-sonnet-4-5",
        editor_model: str = "gemini-2.5-flash-image-preview"
    ):
        self.analyzer = AsyncLLM(analyzer_llm)
        self.editor_agent = ImageGenAgent(editor_model)
        self.tools = VisualRefinementTools()

    async def refine_asset(
        self,
        raw_image_b64: str,
        asset_spec: Dict[str, Any],
        target_size: int = 64,
        reference_assets: List[str] = None,
        processing_guidance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main refinement pipeline with up to three regeneration attempts.

        Args:
            raw_image_b64: Original generated image
            asset_spec: Asset specification
            target_size: Target size (default 64x64)
            reference_assets: Other generated assets for context
            processing_guidance: Processing guidance string

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'dimensions': tuple,
                'iterations': int,
                'regeneration_attempts': int,
                'processing_history': list
            }
        """

        print(f"\n   🔧 Visual Refinement Agent (v2 - Multi-attempt)")
        print(f"      Asset: {asset_spec['name']}")
        print(f"      Type: {'Tileable' if asset_spec.get('is_tileable') else 'Sprite'}")
        print(f"      Target: {target_size}x{target_size}px")
        if processing_guidance:
            print(f"      Guidance: {processing_guidance}")

        # 🔥 Outer loop: up to 3 regeneration attempts
        max_regeneration_attempts = 3
        regenerated_image_b64 = None
        
        for regeneration_attempt in range(max_regeneration_attempts):
            if regeneration_attempt > 0:
                print(f"\n   🔄 === Regeneration Attempt {regeneration_attempt + 1}/{max_regeneration_attempts} ===")
            
            # Decide which image to start with for this attempt
            current_raw_image = raw_image_b64 if regeneration_attempt == 0 else regenerated_image_b64
            
            # Reset history per regeneration attempt
            history = ProcessingHistory()

            # Record initial state
            initial_info = self.tools.get_image_info(current_raw_image)
            initial_step = ProcessingStep(
                step_number=0,
                action_type="initial",
                action_params={},
                before_image_b64=current_raw_image,
                after_image_b64=current_raw_image,
                before_info=initial_info,
                after_info=initial_info,
                success=True,
                notes=f"Raw generated image: {initial_info['width']}x{initial_info['height']}px" + 
                      (f" (Regeneration #{regeneration_attempt + 1})" if regeneration_attempt > 0 else "")
            )
            history.add_step(initial_step)

            modifications_log: List[str] = []
            iteration = 0
            max_iterations = 10

            # 🔥 Inner loop: AI processing flow
            while iteration < max_iterations:
                iteration += 1
                print(f"\n      --- Iteration {iteration} ---")

                # Claude analyzes and decides next action
                action = await self._analyze_and_decide(
                    history=history,
                    asset_spec=asset_spec,
                    target_size=target_size,
                    iteration=iteration,
                    processing_guidance=processing_guidance
                )

                current_image_b64 = history.current_image_b64
                current_info = self.tools.get_image_info(current_image_b64)

                if action['type'] == 'complete':
                    print(f"      ✅ Complete: {action['reason']}")
                    break

                # Execute tool actions
                elif action['type'] == 'remove_background':
                    tolerance = action.get('tolerance', 30)
                    aggressive = action.get('aggressive', False)
                    
                    bg_result = self.tools.remove_background(
                        current_image_b64,
                        tolerance=tolerance,
                        aggressive=aggressive
                    )
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='remove_background',
                        action_params={'tolerance': tolerance, 'aggressive': aggressive},
                        before_image_b64=current_image_b64,
                        after_image_b64=bg_result.get('transparent_image_b64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(bg_result.get('transparent_image_b64', current_image_b64)) if bg_result['success'] else current_info,
                        success=bg_result['success'],
                        error_message=bg_result.get('error') if not bg_result['success'] else None,
                        notes=f"BG type: {bg_result.get('background_type')}, removed {bg_result.get('pixels_removed', 0)} px" if bg_result.get('success') else None
                    )
                    history.add_step(step)
                    
                    if bg_result['success']:
                        modifications_log.append(f"Removed {bg_result['background_type']} background")
                    else:
                        print(f"      ❌ Background removal failed: {bg_result.get('error')}")

                elif action['type'] == 'smart_resize':
                    method = action.get('method', 'LANCZOS')
                    
                    art_style = asset_spec.get('prompt_strategy', {}).get('base_prompt', '').lower()
                    if method == 'LANCZOS' and ('pixel art' in art_style or 'pixelated' in art_style):
                        method = 'NEAREST'
                        print(f"      💡 Auto-selected NEAREST for pixel art")
                    
                    resize_result = self.tools.smart_resize(
                        current_image_b64,
                        target_size,
                        method=method
                    )
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='smart_resize',
                        action_params={'target_size': target_size, 'method': method},
                        before_image_b64=current_image_b64,
                        after_image_b64=resize_result.get('resized_image_b64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(resize_result.get('resized_image_b64', current_image_b64)) if resize_result['success'] else current_info,
                        success=resize_result['success'],
                        error_message=resize_result.get('error') if not resize_result['success'] else None,
                        notes=f"Strategy: {resize_result.get('strategy_used')}" if resize_result.get('success') else None
                    )
                    history.add_step(step)
                    
                    if resize_result['success']:
                        modifications_log.append(f"Smart resized to {target_size}x{target_size} ({resize_result['strategy_used']})")
                    else:
                        print(f"      ❌ Smart resize failed: {resize_result.get('error')}")

                elif action['type'] == 'auto_crop':
                    padding = action.get('padding', 0)
                    crop_result = self.tools.auto_crop_content(current_image_b64, padding)
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='auto_crop',
                        action_params={'padding': padding},
                        before_image_b64=current_image_b64,
                        after_image_b64=crop_result.get('cropped_image_b64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(crop_result.get('cropped_image_b64', current_image_b64)) if crop_result['success'] else current_info,
                        success=crop_result['success'],
                        error_message=crop_result.get('error') if not crop_result['success'] else None
                    )
                    history.add_step(step)
                    
                    if crop_result['success']:
                        modifications_log.append(f"Auto-cropped to content")
                    else:
                        print(f"      ❌ Auto crop failed: {crop_result.get('error')}")

                elif action['type'] == 'extract_tile':
                    grid_size = action.get('grid_size', [2, 2])
                    tile_pos = action.get('tile_pos', [0, 0])
                    
                    extract_result = self.tools.extract_tile_from_grid(
                        current_image_b64,
                        grid_size=tuple(grid_size),
                        tile_position=tuple(tile_pos)
                    )
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='extract_tile',
                        action_params={'grid_size': grid_size, 'tile_pos': tile_pos},
                        before_image_b64=current_image_b64,
                        after_image_b64=extract_result.get('tile_image_b64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(extract_result.get('tile_image_b64', current_image_b64)) if extract_result['success'] else current_info,
                        success=extract_result['success'],
                        error_message=extract_result.get('error') if not extract_result['success'] else None
                    )
                    history.add_step(step)
                    
                    if extract_result['success']:
                        modifications_log.append(f"Extracted tile at {tile_pos} from {grid_size} grid")
                    else:
                        print(f"      ❌ Tile extraction failed: {extract_result.get('error')}")

                elif action['type'] == 'regenerate':
                    reason = action.get('reason', 'Need better version for processing')
                    print(f"      🔄 Regenerating: {reason}")
                    
                    regen_result = await self._regenerate_with_gemini(
                        current_image_b64=current_image_b64,
                        asset_spec=asset_spec,
                        instruction=reason
                    )
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='regenerate',
                        action_params={'reason': reason},
                        before_image_b64=current_image_b64,
                        after_image_b64=regen_result.get('image_base64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(regen_result.get('image_base64', current_image_b64)) if regen_result.get('success') else current_info,
                        success=regen_result.get('success', False),
                        error_message=regen_result.get('error') if not regen_result.get('success') else None,
                        notes="Regenerated with Gemini"
                    )
                    history.add_step(step)
                    
                    if regen_result.get('success'):
                        modifications_log.append(f"Regenerated: {reason}")
                    else:
                        print(f"      ❌ Regeneration failed: {regen_result.get('error')}")

                elif action['type'] == 'edit':
                    instruction = action.get('instruction', '')
                    print(f"      🎨 AI editing: {instruction}")
                    
                    edit_result = await self.tools.edit_with_gemini(
                        current_image_b64,
                        instruction,
                        self.editor_agent
                    )
                    
                    step = ProcessingStep(
                        step_number=iteration,
                        action_type='edit',
                        action_params={'instruction': instruction},
                        before_image_b64=current_image_b64,
                        after_image_b64=edit_result.get('edited_image_b64', current_image_b64),
                        before_info=current_info,
                        after_info=self.tools.get_image_info(edit_result.get('edited_image_b64', current_image_b64)) if edit_result.get('success') else current_info,
                        success=edit_result.get('success', False),
                        error_message=edit_result.get('error') if not edit_result.get('success') else None,
                        notes="Edited with Gemini"
                    )
                    history.add_step(step)
                    
                    if edit_result.get('success'):
                        modifications_log.append(f"AI edited: {instruction[:50]}...")
                    else:
                        print(f"      ❌ AI edit failed: {edit_result.get('error')}")

                else:
                    print(f"      ⚠️  Unknown action: {action['type']}")
                    break

            # 🔥 End of processing loop; get final result
            final_image_b64 = history.current_image_b64 or current_image_b64
            final_info = self.tools.get_image_info(final_image_b64)

            print(f"\n      📊 Final: {final_info['width']}x{final_info['height']}px, {final_info['file_size_kb']}KB")
            print(f"      🔄 Iterations: {iteration}")
            print(f"      ✅ Transparency: {final_info['color_analysis'].get('transparency_percentage', 0)*100:.0f}%")

            # 🔥 Quality check: run on first two attempts, skip on final
            skip_quality_check = (regeneration_attempt >= max_regeneration_attempts - 1)

            if not skip_quality_check and iteration < max_iterations:
                print(f"\n      🔍 AI Quality Check (attempt {regeneration_attempt + 1}/{max_regeneration_attempts})...")
                quality_check = await self._final_quality_check(
                    image_b64=final_image_b64,
                    asset_spec=asset_spec,
                    target_size=target_size
                )
                
                if not quality_check['is_usable'] and quality_check['action'] == 'regenerate':
                    print(f"      ⚠️  Quality issue: {quality_check['reason']}")
                    print(f"      🔄 Triggering regeneration...")
                    
                    regen_result = await self._regenerate_with_gemini(
                        current_image_b64=final_image_b64,
                        asset_spec=asset_spec,
                        instruction=quality_check['regeneration_instruction']
                    )
                    
                    if regen_result.get('success'):
                        regenerated_image_b64 = regen_result['image_base64']
                        print(f"      ✅ Regenerated successfully, starting new processing cycle...")
                        # Continue outer loop
                        continue
                    else:
                        print(f"      ❌ Regeneration failed, using current result")
                        break
                else:
                    # Quality check passed or advised to keep
                    print(f"      ✅ Quality check passed")
                    break
            else:
                if skip_quality_check:
                    print(f"\n      ⚠️  Max regeneration attempts reached ({max_regeneration_attempts}), skipping quality check")
                    print(f"      📦 Using final result as-is")
                break

        # Return final result
        return {
            'success': True,
            'image_base64': final_image_b64,
            'dimensions': (final_info['width'], final_info['height']),
            'iterations': iteration,
            'regeneration_attempts': regeneration_attempt + 1,
            'modifications': modifications_log,
            'processing_history': [step.to_dict() for step in history.steps]
        }

    async def _regenerate_with_gemini(
        self,
        current_image_b64: str,
        asset_spec: Dict[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """Regenerate/fix an image using Gemini."""
        base_prompt = asset_spec.get('prompt_strategy', {}).get('base_prompt', '')

        prompt = f"""🔧 **ASSET REGENERATION FOR GAME ENGINE**

**Original Specification:**
{base_prompt}

**Problem Identified:**
{instruction}

**Your Task:**
Create an IMPROVED version of this asset that:
1. **Keeps the SAME visual content** - same subject, same style, same details
2. **Fixes the technical issues** identified above
3. **Makes it EASY to process** automatically

**CRITICAL Technical Requirements:**
- Render on **SOLID white background (#FFFFFF)** - NO gradients, NO shadows on background
- Subject **centered, filling 70-75% of canvas** - leave clear padding on all sides
- **NO glow/bloom/halo effects** extending beyond subject
- **Clean, sharp edges** for automatic background removal
- Subject should be **isolated and complete** (no cropping)

**Quality Standards:**
- Match the original art style (pixel art / illustrated / realistic / etc.)
- Preserve all important visual details
- High quality rendering suitable for game use
- Proper lighting and shading (within the subject, not on background)

**Output Goal:**
A technically perfect version that automatic tools can easily:
1. Remove background (pure white → transparent)
2. Resize without quality loss
3. Composite into game scenes

Generate the improved asset now."""

        print(f"      📤 Calling Gemini for regeneration...")

        result = await self.editor_agent.generate_image_to_image(
            prompt=prompt,
            reference_images=[current_image_b64],
            preserve_history=False
        )

        return result

    async def _final_quality_check(
        self,
        image_b64: str,
        asset_spec: Dict[str, Any],
        target_size: int
    ) -> Dict[str, Any]:
        """AI-driven final quality check."""
        img_info = self.tools.get_image_info(image_b64)
        
        try:
            on_dark = self.tools.composite_on_color(image_b64, (50, 50, 50, 255))
            on_checker = self.tools.composite_on_checkerboard(image_b64)
        except:
            on_dark = image_b64
            on_checker = image_b64
        
        content = [
            {
                "type": "text",
                "text": f"""🔍 **FINAL QUALITY CHECK FOR GAME ASSET**

You are a game asset quality inspector. Determine if this processed asset is USABLE in a game engine.

**Asset Specification:**
- Name: {asset_spec['name']}
- Type: {'Tileable texture' if asset_spec.get('is_tileable') else 'Sprite/Icon'}
- Target size: {target_size}x{target_size}px
- Purpose: {asset_spec.get('prompt_strategy', {}).get('base_prompt', 'N/A')[:200]}

**Technical Information:**
- Current size: {img_info['width']}x{img_info['height']}px
- Transparency: {img_info['color_analysis'].get('transparency_percentage', 0)*100:.0f}%
- Has white background: {img_info['color_analysis'].get('has_white_background')}

**Image to inspect:**
"""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            },
            {
                "type": "text",
                "text": "\n**On dark background (transparency check):**"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{on_dark}"}
            },
            {
                "type": "text",
                "text": "\n**On checkerboard (transparency check):**"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{on_checker}"}
            },
            {
                "type": "text",
                "text": f"""

**Your Task:**
Inspect this asset and determine if it's USABLE for game development.

**Critical Issues that make it UNUSABLE:**
1. **Content overfill**: Subject fills >90% of canvas (no padding for rendering)
2. **Size wrong**: Not {target_size}x{target_size} pixels
3. **Background not removed**: Still has white/colored background (check helper views)
4. **Subject damaged**: Main content has missing pixels, artifacts, or distortion
5. **Wrong content**: Doesn't match the asset specification
6. **Quality too low**: Blurry, pixelated (if not pixel art), or corrupted

**Decision Logic:**
IF asset has ANY critical issue → **NOT USABLE**
  - IF issue is from BAD GENERATION (wrong content, quality, or damaged subject):
    → ACTION: **regenerate** (call Gemini to create better version)
  - IF issue is from BAD PROCESSING (size, background, overfill):
    → ACTION: **reprocess** (try different tools)

IF asset is acceptable → **USABLE**
  → ACTION: **ok**

**Output Format (JSON):**
```json
{{
  "is_usable": true/false,
  "reason": "Brief explanation of your decision",
  "action": "ok" | "regenerate" | "reprocess",
  "suggested_fix": "What needs to be fixed",
  "regeneration_instruction": "Detailed instruction for Gemini IF action=regenerate"
}}
```

Inspect the images carefully and output your decision in JSON format.
"""
            }
        ]
        
        try:
            response = await self.analyzer(content)
            result = self._parse_quality_check(response)
            return result
            
        except Exception as e:
            print(f"      ⚠️  Quality check error: {e}")
            return {
                'is_usable': True,
                'reason': f'Quality check failed: {e}',
                'action': 'ok',
                'suggested_fix': 'Unable to verify',
                'regeneration_instruction': ''
            }
    
    def _parse_quality_check(self, response: str) -> Dict[str, Any]:
        """Parse Claude's quality-check result."""
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                return result
            except json.JSONDecodeError:
                pass
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return result
            except json.JSONDecodeError:
                pass
        
        return {
            'is_usable': True,
            'reason': 'Failed to parse quality check',
            'action': 'ok',
            'suggested_fix': '',
            'regeneration_instruction': ''
        }

    async def _analyze_and_decide(
        self,
        history: ProcessingHistory,
        asset_spec: Dict[str, Any],
        target_size: int,
        iteration: int,
        processing_guidance: Optional[str] = None
    ) -> Dict[str, Any]:
        """Claude analyzes the image and chooses the next action."""
        current_image_b64 = history.current_image_b64
        img_info = self.tools.get_image_info(current_image_b64)

        print(f"      📊 Current: {img_info['width']}x{img_info['height']}px, transparency={img_info['color_analysis'].get('transparency_percentage', 0)*100:.0f}%")

        content = []

        content.append({
            "type": "text",
            "text": f"**Processing History:**\n{history.get_summary(max_steps=5)}"
        })

        recent_images = history.get_latest_images(count=min(3, len(history.steps)))
        for img_data in recent_images:
            content.append({
                "type": "text",
                "text": f"\n**Step {img_data['step']} - {img_data['action']}:**"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data['image_b64']}"}
            })

        content.append({
            "type": "text",
            "text": f"\n**Current Image (Step {history.steps[-1].step_number}):**"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{current_image_b64}"}
        })

        content.append({
            "type": "text",
            "text": self._build_analysis_prompt(
                asset_spec=asset_spec,
                target_size=target_size,
                img_info=img_info,
                iteration=iteration,
                processing_guidance=processing_guidance
            )
        })

        try:
            response = await self.analyzer(content)
            action = self._parse_decision(response)
            return action

        except Exception as e:
            print(f"      ❌ Analysis error: {e}")
            return {'type': 'complete', 'reason': f'Error: {e}'}

    def _build_analysis_prompt(
        self,
        asset_spec: Dict[str, Any],
        target_size: int,
        img_info: Dict[str, Any],
        iteration: int,
        processing_guidance: Optional[str] = None
    ) -> str:
        """Construct the analysis prompt for Claude."""
        is_tileable = asset_spec.get('is_tileable', False)
        asset_name = asset_spec.get('name', 'Unknown')
        base_prompt = asset_spec.get('prompt_strategy', {}).get('base_prompt', '')
        guidance_text = processing_guidance or asset_spec.get('processing_guidance') or "No additional post-processing instructions provided."

        return f"""You are a game asset quality control specialist. Your goal is to prepare this image for use as a game asset.

**FINAL GOAL:**
1. ✅ Size: EXACTLY {target_size}x{target_size} pixels
2. ✅ Background: FULLY TRANSPARENT (no white background)
3. ✅ Content: Centered in the canvas
4. ✅ Quality: Clean, suitable for game use

**Asset Requirements:**
- Name: {asset_name}
- Type: {"Tileable texture (single repeating unit)" if is_tileable else "Sprite (isolated game object/character)"}
- Description: {base_prompt}
- Post-processing guidance: {guidance_text}

**Current Image Info:**
- Size: {img_info['width']}x{img_info['height']} pixels
- Has transparency: {img_info['has_transparency']}
- Has white background: {img_info['color_analysis'].get('has_white_background')}
- Transparency: {img_info['color_analysis'].get('transparency_percentage', 0)*100:.0f}%

**Current Iteration:** {iteration}/10

**Available Tools:**

1. **complete** - Mark as finished
   {{"action": "complete", "reason": "..."}}

2. **remove_background** - Universal background remover
   {{"action": "remove_background", "tolerance": 30, "aggressive": false}}
   - tolerance: 0-255 (higher = more permissive, default 30)
   - aggressive: true = internal sampling for UI elements

3. **smart_resize** - Universal resizer (auto-chooses best strategy)
   {{"action": "smart_resize", "target_size": {target_size}, "method": "LANCZOS"}}
   - method: "LANCZOS" (smooth) | "NEAREST" (pixel art)

4. **auto_crop** - Crop to content bounds + optional padding
   {{"action": "auto_crop", "padding": 0}}

5. **extract_tile** - Extract single tile from grid/sheet
   {{"action": "extract_tile", "grid_size": [cols, rows], "tile_pos": [x, y]}}

6. **regenerate** - AI: Regenerate for easier processing
   {{"action": "regenerate", "reason": "..."}}
   - Use when tools fail repeatedly

7. **edit** - AI: Complex edits (Gemini API)
   {{"action": "edit", "instruction": "..."}}

**Workflow:**
1. Extract tile (if multi-tile) → extract_tile
2. Remove background → remove_background
3. Crop to content → auto_crop
4. Resize to target → smart_resize
5. Verify → complete

**Rules:**
- Try local tools first (2-5), AI tools (6-7) only when local fails
- Process step-by-step, one action per iteration
- Mark complete when size={target_size}x{target_size}, transparent background, centered

Output JSON only:
```json
{{"action": "complete|remove_background|smart_resize|auto_crop|extract_tile|regenerate|edit", ...}}
```
"""

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse Claude's JSON decision response."""
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(1))
                if 'action' in decision:
                    decision['type'] = decision['action']
                return decision
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group(0))
                if 'action' in decision:
                    decision['type'] = decision['action']
                return decision
            except json.JSONDecodeError:
                pass

        print(f"      ⚠️  Failed to parse decision, defaulting to complete")
        return {'type': 'complete', 'reason': 'Failed to parse decision'}


# Backward-compatible alias
VisualRefinementAgent = RefinementAgent


if __name__ == '__main__':
    print("Visual Refinement Agent v2 - Ready")
    print("Features: 3x regeneration attempts, high-performance background removal, smart toolset")
