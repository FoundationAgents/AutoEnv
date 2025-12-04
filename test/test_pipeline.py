"""测试 AutoEnv Pipeline"""

import asyncio
import sys
from pathlib import Path

from autoenv.pipeline import AutoEnvPipeline


async def main():
    # 创建 pipeline
    pipeline = AutoEnvPipeline.create_default(
        image_model="gemini-2.5-flash-image",
        llm_name="gemini-2.5-flash",
    )

    print("\nRunning pipeline...")
    # 直接运行
    ctx = await pipeline.run(instruction="A 2D grid maze game",
                             output_dir=Path("test_output_2"))

    print(f"\nResult: success={ctx.success}, error={ctx.error}")


if __name__ == "__main__":
    asyncio.run(main())

