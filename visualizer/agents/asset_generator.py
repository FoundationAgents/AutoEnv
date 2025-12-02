"""
Adaptive Asset Generator
Dynamically generate assets based on a strategy.
"""

from typing import Dict, List, Any

# Concise technical constraints to avoid prompt bloat
MANDATORY_TECH_REQUIREMENTS = (
    "\n\n🚨 CRITICAL TECHNICAL REQUIREMENTS:\n"
    "1. Background: SOLID COLOR for max contrast with subject\n"
    "   - Dark subject → WHITE (#FFF) background\n"
    "   - Light subject → BLACK (#000) background\n"
    "   - Colorful subject → OPPOSITE color background\n"
    "   - NO gradients/shadows\n"
    "2. Edges: NO outer glow/bloom/halo effects\n"
    "3. Composition: ONE centered subject, 70-85% canvas fill\n"
    "4. Clean edges for easy background removal\n"
)

from visualizer.agents.image_gen_agent import ImageGenAgent


class AdaptiveAssetGenerator:
    """Adaptive asset generator driven by strategy."""

    def __init__(self, image_agent: ImageGenAgent):
        self.image_agent = image_agent

    async def generate_by_strategy(
        self,
        asset_spec: Dict[str, Any],
        strategy: Dict[str, Any],
        reference_images: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an asset according to the strategy rules.

        No hardcoded prompt templates; follows strategy guidance only.

        Args:
            asset_spec: Asset specification
            strategy: Full strategy dict
            reference_images: List of reference images (base64)

        Returns:
            Result dict {'success': bool, 'image_base64': str, ...}
        """

        prompt_strategy = asset_spec.get('prompt_strategy', {})
        base_prompt = prompt_strategy.get('base_prompt', '')
        processing_guidance = asset_spec.get('processing_guidance')

        # If strategy defines a prompt builder, use it
        if 'approach' in prompt_strategy:
            # Let another AI build the final prompt from the approach
            final_prompt = await self._build_prompt_by_approach(
                asset_spec,
                strategy
            )
        else:
            final_prompt = base_prompt

        # 🔥 Enhance prompt with layout/size guidance
        final_prompt = self._enhance_prompt_with_layout_guidance(
            final_prompt,
            asset_spec
        )

        # Append processing guidance if provided
        if processing_guidance:
            trimmed_prompt = final_prompt.strip()
            guidance_suffix = (
                "\n\nIMPORTANT POST-PROCESSING REQUIREMENTS: "
                f"{processing_guidance.strip()}"
            )
            final_prompt = (trimmed_prompt + guidance_suffix) if trimmed_prompt else guidance_suffix

        # 🔥 Force technical requirements
        if "CRITICAL TECHNICAL REQUIREMENTS" not in final_prompt:
            final_prompt = (final_prompt.rstrip() + MANDATORY_TECH_REQUIREMENTS)

        # Choose generation method based on references
        if reference_images:
            result = await self.image_agent.generate_image_to_image(
                prompt=final_prompt,
                reference_images=reference_images
            )
        else:
            result = await self.image_agent.generate_text_to_image(
                prompt=final_prompt
            )

        return result

    def _enhance_prompt_with_layout_guidance(
        self,
        base_prompt: str,
        asset_spec: Dict[str, Any]
    ) -> str:
        """
        🔥 Enhance prompt with precise layout and sizing guidance.

        This keeps outputs usable in-game.
        """
        
        is_tileable = asset_spec.get('is_tileable', False)
        asset_type = asset_spec.get('asset_type', 'object')
        
        # Build layout guidance
        layout_guidance = "\n\n📐 LAYOUT REQUIREMENTS:\n"
        
        if is_tileable:
            layout_guidance += (
                "- This is a TILEABLE texture - generate a SINGLE repeating unit\n"
                "- Subject should fill 90-95% of canvas (edge-to-edge for seamless tiling)\n"
                "- Ensure all edges can seamlessly connect (no visible seams)\n"
                "- DO NOT show multiple tiles or a grid - just ONE tile\n"
            )
        else:
            layout_guidance += (
                "- This is an isolated game object - generate ONE centered subject\n"
                "- Subject should fill 70-85% of canvas\n"
                "- Leave even padding (10-15%) on all sides for processing\n"
                "- Center the subject both horizontally and vertically\n"
            )
        
        # Add type-specific guidance
        if asset_type in ('character', 'player', 'enemy'):
            layout_guidance += (
                "- Character should face forward or specified direction\n"
                "- Show complete character (head to feet if standing)\n"
                "- Pose should be clear and game-ready\n"
            )
        elif asset_type in ('tile', 'terrain'):
            layout_guidance += (
                "- Show the tile from top-down or isometric view as specified\n"
                "- Fill frame consistently (important for grid alignment)\n"
            )
        elif asset_type == 'ui':
            layout_guidance += (
                "- UI element should be clearly visible and readable\n"
                "- Avoid decorative elements that extend beyond main subject\n"
            )

        # Insert between base prompt and technical requirements
        return base_prompt.strip() + layout_guidance

    async def _build_prompt_by_approach(
        self,
        asset_spec: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> str:
        """
        Build a prompt dynamically based on the strategy's approach.

        Placeholder: delegated to another AI in future.
        """

        # A prompt-builder AI could go here; for now return base_prompt

        return asset_spec.get('prompt_strategy', {}).get('base_prompt', '')
