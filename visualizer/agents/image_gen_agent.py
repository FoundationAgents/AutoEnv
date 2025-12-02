"""
Image Generation Agent
Uses Gemini-style image generation APIs on top of AsyncLLM.
"""

import base64
import re
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
from io import BytesIO

import sys
import os
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from base.engine.async_llm import AsyncLLM, LLMsConfig


class ImageGenAgent:
    """Image generator supporting text-to-image and image-to-image."""

    def __init__(self, model_name: str = "gemini-2.5-flash-image-preview"):
        """
        Initialize the image generator.

        Args:
            model_name: Model name from model_config.yaml
        """
        self.model_name = model_name
        self.llm = AsyncLLM(model_name)  # AsyncLLM will resolve config by name

        # Conversation history (for style consistency)
        self.conversation_history = []
        self.last_generated_image_b64 = None

    async def generate_text_to_image(
        self,
        prompt: str,
        preserve_history: bool = True
    ) -> Dict[str, Any]:
        """
        Text-to-image generation.

        Args:
            prompt: Prompt to generate
            preserve_history: Whether to record conversation history

        Returns:
            {
                'success': bool,
                'image_base64': str,  # base64-encoded image
                'prompt': str
            }
        """
        try:
            # Call LLM (Gemini-style generation)
            response = await self.llm(prompt)

            # Extract image (format: ![image](data:image/png;base64,...))
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    'success': False,
                    'error': 'No image found in response',
                    'raw_response': response
                }

            # Save to history
            if preserve_history:
                self.last_generated_image_b64 = image_b64
                self.conversation_history.append({
                    'type': 'text_to_image',
                    'prompt': prompt,
                    'image_b64': image_b64
                })

            return {
                'success': True,
                'image_base64': image_b64,
                'prompt': prompt
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt
            }

    async def generate_image_to_image(
        self,
        prompt: str,
        reference_images: List[str],  # base64 encoded images
        preserve_history: bool = True
    ) -> Dict[str, Any]:
        """
        Image-to-image generation (with style references).

        Args:
            prompt: Prompt to generate
            reference_images: List of base64-encoded reference images
            preserve_history: Whether to record conversation history

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'prompt': str
            }
        """
        try:
            # Build multimodal message with references
            # We use OpenAI-compatible format for images
            message_with_images = self._build_image_message(prompt, reference_images)

            # Call LLM
            response = await self.llm(message_with_images)

            # Extract generated image
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    'success': False,
                    'error': 'No image found in response',
                    'raw_response': response
                }

            # Save to history
            if preserve_history:
                self.last_generated_image_b64 = image_b64
                self.conversation_history.append({
                    'type': 'image_to_image',
                    'prompt': prompt,
                    'reference_count': len(reference_images),
                    'image_b64': image_b64
                })

            return {
                'success': True,
                'image_base64': image_b64,
                'prompt': prompt
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt
            }

    def _build_image_message(self, text: str, images: List[str]) -> list:
        """
        Build a multimodal message with images and text.

        Uses an OpenAI Vision-compatible format.

        Args:
            text: Text prompt
            images: Base64-encoded images

        Returns:
            Content list combining images and text
        """
        content = []

        # 1) Add all reference images first
        for i, image_b64 in enumerate(images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }
            })

        # 2) Append text prompt (images first, text after)
        content.append({
            "type": "text",
            "text": f"""Above are {len(images)} reference image(s) showing the target visual style.

{text}

IMPORTANT: Use the reference images as a STYLE GUIDE. Match:
- Visual style and art technique (pixel art/hand-drawn/3D/etc.)
- Color palette and saturation levels
- Level of detail and complexity
- Rendering approach (shading, outlines, textures)
- Overall aesthetic consistency

The new image MUST look like it comes from the SAME GAME as the references.
Generate the new asset maintaining perfect visual consistency.
"""
        })

        return content

    def _extract_image_from_response(self, response: str) -> Optional[str]:
        """Extract base64 image from an LLM response."""

        # Format example: ![image](data:image/png;base64,...)
        match = re.search(r'data:image/[^;]+;base64,([^)]+)', response)
        if match:
            return match.group(1)

        return None

    def decode_image(self, image_b64: str) -> Image.Image:
        """Decode a base64 image into a PIL Image."""
        img_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(img_bytes))

    def save_image(self, image_b64: str, output_path: str):
        """Save a base64 image to disk."""
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = self.decode_image(image_b64)
        img.save(str(output_path))
        print(f"💾 Saved image: {output_path}")


class StyleAnchorGenerator:
    """Style anchor generator for the first reference asset."""

    def __init__(self, image_gen_agent: ImageGenAgent):
        self.agent = image_gen_agent

    async def generate_anchor(
        self,
        theme: str,
        asset_type: str = "floor_tile"
    ) -> Dict[str, Any]:
        """
        Generate the style anchor asset.

        Args:
            theme: Visual theme (e.g., "medieval underground dungeon, pixel art")
            asset_type: Anchor asset type

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'theme': str,
                'asset_type': str
            }
        """
        prompt = f"""
Create a high-quality game asset that will serve as the STYLE ANCHOR for an entire asset set.

Theme: {theme}
Asset Type: {asset_type}

This is the FIRST and MOST IMPORTANT asset. All other assets will reference this one for visual consistency.

Requirements:
1. Rich in visual detail to clearly establish the art style
2. Exemplify the theme perfectly
3. High quality and well-crafted
4. If tileable texture, ensure seamless edges
5. White or transparent background for easy extraction

This asset will be used as the visual reference for all subsequent generations.
"""

        print(f"\n🎨 Generating Style Anchor...")
        print(f"   Theme: {theme}")
        print(f"   Asset: {asset_type}")

        result = await self.agent.generate_text_to_image(prompt)

        if result['success']:
            print(f"   ✅ Style anchor generated successfully")
            return {
                'success': True,
                'image_base64': result['image_base64'],
                'theme': theme,
                'asset_type': asset_type,
                'prompt': prompt
            }
        else:
            print(f"   ❌ Failed: {result.get('error')}")
            return result


class StyleConsistentGenerator:
    """Generate assets that stay consistent with a style anchor."""

    def __init__(self, image_gen_agent: ImageGenAgent):
        self.agent = image_gen_agent

    async def generate_with_style_reference(
        self,
        asset_description: str,
        style_anchor_b64: str,
        theme: str,
        previous_assets: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a new asset using the style anchor as reference.

        Args:
            asset_description: Asset description
            style_anchor_b64: Style anchor image (base64)
            theme: Theme string
            previous_assets: Previously generated assets (optional extra references)

        Returns:
            {
                'success': bool,
                'image_base64': str,
                'description': str
            }
        """
        prompt = f"""
Create a game asset that EXACTLY MATCHES the visual style of the reference image.

Theme: {theme}
New Asset: {asset_description}

CRITICAL STYLE MATCHING REQUIREMENTS:
1. MUST use the same art style (pixel art / hand-drawn / 3D / etc.)
2. MUST use the same color palette and color saturation
3. MUST match the level of detail and complexity
4. MUST use the same rendering technique (shading, outlines, textures)
5. Should look like it comes from the SAME GAME as the reference

Visual Consistency Check:
- If reference is pixel art → new asset MUST be pixel art
- If reference has outlines → new asset MUST have similar outlines
- If reference has specific color tone → new asset MUST match that tone

Generate: {asset_description}

White or transparent background for extraction.
"""

        print(f"\n🖌️  Generating asset with style reference...")
        print(f"   Asset: {asset_description}")

        # Prepare reference images
        reference_images = [style_anchor_b64]
        if previous_assets:
            # Take up to the two most recent for extra context
            reference_images.extend(previous_assets[-2:])

        # Use image-to-image generation
        result = await self.agent.generate_image_to_image(
            prompt=prompt,
            reference_images=reference_images
        )

        if result['success']:
            print(f"   ✅ Generated successfully")
        else:
            print(f"   ❌ Failed: {result.get('error')}")

        return result


# Example usage
async def demo_style_anchor_generation():
    """Demo: generate a style anchor and a follow-up asset."""

    # 1) Initialize
    image_agent = ImageGenAgent("gemini-2.5-flash-image-preview")
    anchor_gen = StyleAnchorGenerator(image_agent)

    # 2) Generate style anchor
    theme = "medieval underground dungeon, dark pixel art style, atmospheric"

    anchor_result = await anchor_gen.generate_anchor(
        theme=theme,
        asset_type="ancient stone floor tile"
    )

    if anchor_result['success']:
        # Save anchor
        image_agent.save_image(
            anchor_result['image_base64'],
            "output/style_anchor.png"
        )

        # 3) Generate another asset based on the anchor
        style_gen = StyleConsistentGenerator(image_agent)

        wall_result = await style_gen.generate_with_style_reference(
            asset_description="ancient stone wall, dark gray, medieval dungeon style",
            style_anchor_b64=anchor_result['image_base64'],
            theme=theme
        )

        if wall_result['success']:
            image_agent.save_image(
                wall_result['image_base64'],
                "output/wall_tile.png"
            )


if __name__ == '__main__':
    asyncio.run(demo_style_anchor_generation())
