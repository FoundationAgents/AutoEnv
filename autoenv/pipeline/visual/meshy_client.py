"""
Meshy API Client for Image-to-3D conversion.

Uses the Meshy AI API to convert 2D images to 3D models.
API Documentation: https://docs.meshy.ai/en/api/image-to-3d
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import aiohttp


class MeshyClient:
    """Async client for Meshy Image-to-3D API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.meshy.ai/v1",
    ):
        """
        Initialize Meshy client.

        Args:
            api_key: Meshy API key
            base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def image_to_data_uri(self, image_path: Path) -> str:
        """
        Convert image file to Data URI format.

        Args:
            image_path: Path to the image file

        Returns:
            Data URI string (data:image/png;base64,...)
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        return f"data:{mime_type};base64,{b64}"

    def base64_to_data_uri(self, b64: str, mime_type: str = "image/png") -> str:
        """
        Convert base64 string to Data URI format.

        Args:
            b64: Base64 encoded image string
            mime_type: MIME type of the image

        Returns:
            Data URI string
        """
        # Remove existing data URI prefix if present
        if b64.startswith("data:"):
            return b64
        return f"data:{mime_type};base64,{b64}"

    async def create_image_to_3d_task(
        self,
        image_data_uri: str,
        topology: str = "quad",
        target_polycount: int = 10000,
        should_remesh: bool = True,
        enable_pbr: bool = True,
    ) -> dict[str, Any]:
        """
        Create an Image-to-3D task.

        Args:
            image_data_uri: Data URI of the input image
            topology: Mesh topology ("quad" or "triangle")
            target_polycount: Target polygon count (keep low for faster processing)
            should_remesh: Whether to remesh the output
            enable_pbr: Enable PBR textures

        Returns:
            API response with task ID
        """
        url = f"{self.base_url}/image-to-3d"
        payload = {
            "image_url": image_data_uri,
            "topology": topology,
            "target_polycount": target_polycount,
            "should_remesh": should_remesh,
            "enable_pbr": enable_pbr,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as resp:
                if resp.status != 200 and resp.status != 202:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {error_text}",
                    }
                data = await resp.json()
                return {
                    "success": True,
                    "task_id": data.get("result"),
                    "response": data,
                }

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """
        Get the status of an Image-to-3D task.

        Args:
            task_id: The task ID to check

        Returns:
            Task status and result
        """
        url = f"{self.base_url}/image-to-3d/{task_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {error_text}",
                    }
                data = await resp.json()
                return {
                    "success": True,
                    "status": data.get("status"),
                    "progress": data.get("progress", 0),
                    "model_urls": data.get("model_urls", {}),
                    "thumbnail_url": data.get("thumbnail_url"),
                    "response": data,
                }

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float = 600,  # 10 minutes default timeout
        poll_interval: float = 5.0,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Wait for an Image-to-3D task to complete.

        Args:
            task_id: The task ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks
            progress_callback: Optional callback for progress updates

        Returns:
            Final task result
        """
        start_time = time.time()
        last_progress = -1

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "task_id": task_id,
                }

            status = await self.get_task_status(task_id)
            if not status["success"]:
                return status

            current_status = status.get("status", "")
            progress = status.get("progress", 0)

            # Report progress if changed
            if progress != last_progress and progress_callback:
                progress_callback(task_id, current_status, progress)
            last_progress = progress

            if current_status == "SUCCEEDED":
                return {
                    "success": True,
                    "status": "SUCCEEDED",
                    "model_urls": status.get("model_urls", {}),
                    "thumbnail_url": status.get("thumbnail_url"),
                    "task_id": task_id,
                    "elapsed_time": elapsed,
                }
            elif current_status == "FAILED":
                return {
                    "success": False,
                    "status": "FAILED",
                    "error": status.get("response", {}).get("message", "Task failed"),
                    "task_id": task_id,
                }
            elif current_status == "EXPIRED":
                return {
                    "success": False,
                    "status": "EXPIRED",
                    "error": "Task expired",
                    "task_id": task_id,
                }

            await asyncio.sleep(poll_interval)

    async def image_to_3d(
        self,
        image_path: Path | None = None,
        image_base64: str | None = None,
        timeout: float = 600,
        progress_callback=None,
        target_polycount: int = 10000,
    ) -> dict[str, Any]:
        """
        Convert image to 3D model (full workflow).

        Args:
            image_path: Path to input image (provide either this or image_base64)
            image_base64: Base64 encoded image (provide either this or image_path)
            timeout: Maximum time to wait for completion
            progress_callback: Optional callback for progress updates
            target_polycount: Target polygon count

        Returns:
            Result with model URLs or error
        """
        # Convert to Data URI
        if image_path:
            data_uri = self.image_to_data_uri(image_path)
        elif image_base64:
            data_uri = self.base64_to_data_uri(image_base64)
        else:
            return {"success": False, "error": "Provide image_path or image_base64"}

        # Create task
        task_result = await self.create_image_to_3d_task(
            image_data_uri=data_uri,
            target_polycount=target_polycount,
        )

        if not task_result["success"]:
            return task_result

        task_id = task_result["task_id"]

        # Wait for completion
        return await self.wait_for_task(
            task_id=task_id,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    async def download_model(
        self,
        model_url: str,
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Download a 3D model file.

        Args:
            model_url: URL of the model file
            output_path: Path to save the model

        Returns:
            Result with success status
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as resp:
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": f"HTTP {resp.status}",
                        }
                    content = await resp.read()
                    output_path.write_bytes(content)

            return {
                "success": True,
                "path": str(output_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
