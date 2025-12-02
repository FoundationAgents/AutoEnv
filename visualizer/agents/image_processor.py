"""
Deterministic Image Processor (non-AI).

Steps:
1. Remove pure white background → make transparent
2. Crop to content bounds
3. Scale proportionally to target size
4. Center on a square canvas
"""

import base64
from io import BytesIO
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np


class ImageProcessor:
    """Deterministic image processing utility."""

    def __init__(self, target_size: int = 256, background_threshold: int = 250):
        """
        Args:
            target_size: Target canvas size (square)
            background_threshold: White threshold (RGB values above treated as white)
        """
        self.target_size = target_size
        self.background_threshold = background_threshold

    def process_asset(
        self,
        image_base64: str,
        asset_name: str = "asset"
    ) -> Dict[str, Any]:
        """
        Process a single asset through deterministic cleanup.

        Args:
            image_base64: Base64-encoded image
            asset_name: Name for logging

        Returns:
            {
                'success': bool,
                'image_base64': str,  # processed image
                'dimensions': (width, height),
                'steps': List[str],  # processing steps
                'error': str  # if failed
            }
        """
        steps = []
        
        try:
            # 1) Decode
            image = self._decode_image(image_base64)
            original_size = image.size
            steps.append(f"Original size: {original_size[0]}x{original_size[1]}px")

            # 2) Ensure RGBA
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
                steps.append("Converted to RGBA")

            # 3) Remove white background
            image = self._remove_white_background(image)
            steps.append(f"Removed white background (threshold: {self.background_threshold})")

            # 4) Crop transparent edges
            image = self._crop_transparent_edges(image)
            cropped_size = image.size
            steps.append(f"Cropped to: {cropped_size[0]}x{cropped_size[1]}px")

            # 5) Resize with aspect ratio preserved
            image = self._resize_keep_aspect(image, self.target_size)
            resized_size = image.size
            steps.append(f"Resized to: {resized_size[0]}x{resized_size[1]}px")

            # 6) Center on square canvas
            image = self._center_on_canvas(image, self.target_size)
            final_size = image.size
            steps.append(f"Final canvas: {final_size[0]}x{final_size[1]}px")

            # 7) Encode to base64
            processed_base64 = self._encode_image(image)

            return {
                'success': True,
                'image_base64': processed_base64,
                'dimensions': final_size,
                'steps': steps
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': steps
            }

    def _decode_image(self, image_base64: str) -> Image.Image:
        """base64"""
        # data URL
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))

    def _encode_image(self, image: Image.Image, format: str = 'PNG') -> str:
        """base64"""
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _remove_white_background(self, image: Image.Image) -> Image.Image:
        """
        Remove white background and soften near-white edges.

        Strategy:
        - Pixels with RGB >= threshold become fully transparent
        - Near-white pixels are partially faded
        """
        # To numpy
        img_array = np.array(image)
        
        # Split channels
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3].copy()
        
        # Calculate whiteness
        whiteness = rgb.mean(axis=2)
        
        # Fully white pixels (>= threshold)
        is_white = (rgb[:, :, 0] >= self.background_threshold) & \
                   (rgb[:, :, 1] >= self.background_threshold) & \
                   (rgb[:, :, 2] >= self.background_threshold)
        
        # Make fully transparent
        alpha[is_white] = 0
        
        # Near-white pixels in [threshold-20, threshold)
        near_white = (whiteness >= self.background_threshold - 20) & \
                     (whiteness < self.background_threshold) & \
                     (~is_white)
        
        if near_white.any():
            # Fade proportionally
            fade_factor = (whiteness[near_white] - (self.background_threshold - 20)) / 20
            alpha[near_white] = (alpha[near_white] * (1 - fade_factor)).astype(np.uint8)
        
        # Reassemble
        img_array[:, :, 3] = alpha
        
        return Image.fromarray(img_array, 'RGBA')

    def _crop_transparent_edges(self, image: Image.Image) -> Image.Image:
        """
        Crop transparent edges with a small padding.
        """
        # Alpha channel
        alpha = np.array(image.split()[3])
        
        # Non-transparent pixels
        non_transparent = np.where(alpha > 10)  # threshold 10 to ignore noise
        
        if len(non_transparent[0]) == 0:
            # Fully transparent: return tiny transparent image
            return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        
        # Bounding box
        top = non_transparent[0].min()
        bottom = non_transparent[0].max()
        left = non_transparent[1].min()
        right = non_transparent[1].max()
        
        # Add small padding
        padding = 2
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(image.height - 1, bottom + padding)
        right = min(image.width - 1, right + padding)
        
        return image.crop((left, top, right + 1, bottom + 1))

    def _resize_keep_aspect(self, image: Image.Image, max_size: int) -> Image.Image:
        """
        Resize proportionally so the long edge is max_size using LANCZOS.
        """
        width, height = image.size
        
        # No upscaling if already small
        if width <= max_size and height <= max_size:
            return image
        
        # Compute new size
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        # Ensure at least 1 pixel
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _center_on_canvas(self, image: Image.Image, canvas_size: int) -> Image.Image:
        """
        Center the image on a transparent square canvas.
        """
        # Create canvas
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        
        # Compute position
        width, height = image.size
        x = (canvas_size - width) // 2
        y = (canvas_size - height) // 2
        
        # Paste with alpha
        canvas.paste(image, (x, y), image)
        
        return canvas

    def get_image_info(self, image_base64: str) -> Dict[str, Any]:
        """
        Get basic image info.
        
        Returns:
            {
                'size': (width, height),
                'mode': str,
                'has_transparency': bool,
                'transparency_percentage': float,
                'dominant_colors': List[Tuple[int, int, int]]
            }
        """
        try:
            image = self._decode_image(image_base64)
            
            info = {
                'size': image.size,
                'mode': image.mode
            }
            
            # Transparency info
            if image.mode == 'RGBA':
                alpha = np.array(image.split()[3])
                transparent_pixels = (alpha < 10).sum()
                total_pixels = alpha.size
                
                info['has_transparency'] = transparent_pixels > 0
                info['transparency_percentage'] = transparent_pixels / total_pixels
            else:
                info['has_transparency'] = False
                info['transparency_percentage'] = 0.0
            
            return info
            
        except Exception as e:
            return {'error': str(e)}


# 
def test_processor():
    """Manual test for the image processor."""
    processor = ImageProcessor(target_size=256)
    
    # Create a test image (white background + red circle)
    test_image = Image.new('RGB', (400, 300), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([100, 50, 300, 250], fill='red')
    
    # 
    buffer = BytesIO()
    test_image.save(buffer, format='PNG')
    buffer.seek(0)
    test_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    # 
    result = processor.process_asset(test_base64, "test_circle")
    
    if result['success']:
        print("✅ ")
        print(f"   : {result['dimensions']}")
        print("   :")
        for step in result['steps']:
            print(f"     - {step}")
        
        # 
        output_data = base64.b64decode(result['image_base64'])
        with open('test_output.png', 'wb') as f:
            f.write(output_data)
        print("   : test_output.png")
    else:
        print(f"❌ : {result['error']}")


if __name__ == '__main__':
    test_processor()
