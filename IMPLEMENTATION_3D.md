# 3D 场景生成 - 实现说明

## 实现内容

### 1. 新增节点：ThreeJSAssemblyNode

**位置**: `autoenv/pipeline/visual/nodes.py`

**功能**:

- 替代 3D 模式下的 AssemblyNode (2D 用 pygame，3D 用 three.js)
- 从 three.js 模板生成可交互 HTML 场景
- 自动生成模型加载、定位、动画代码
- 可选 agent 增强（添加游戏逻辑）

**核心方法**:

```python
_generate_threejs_scene()  # 生成HTML文件
_generate_positioning_code()  # 模型定位逻辑（圆形布局）
_generate_animation_code()  # 简单旋转动画
_build_enhancement_prompt()  # Agent提示（可选）
```

### 2. Three.js 模板

**位置**: `autoenv/pipeline/visual/threejs_template.html`

**特性**:

- 使用 three.js CDN (v0.170.0)
- OrbitControls (鼠标控制相机)
- GLTFLoader (加载.glb 模型)
- 完整光照系统 (环境光 + 方向光 + 阴影)
- 地面平面 + 网格辅助
- 键盘 WASD 移动
- 加载进度提示

**可替换占位符**:

- `{MODEL_COUNT}`: 模型数量
- `{MODEL_PATHS_JSON}`: 模型路径映射
- `{MODEL_POSITIONING_CODE}`: JS 定位代码
- `{ANIMATION_CODE}`: JS 动画代码

### 3. Pipeline 更新

**位置**: `autoenv/pipeline/visual/pipeline.py`

**变更**:

```python
# 2D模式 (dimension="2d")
Analyzer → Strategist → AssetGenerator → BackgroundRemoval → AssemblyNode (pygame)

# 3D模式 (dimension="3d")
Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Image3DConvert → ThreeJSAssemblyNode (three.js)
```

**新增 agent**: `threejs_agent` (step_limit=60, cost_limit=12.0)

## 功能对比分析

### 2D Assembly (Pygame)

| 方面 | 实现                  |
| ---- | --------------------- |
| 输出 | game.py (Python 脚本) |
| 资源 | .png 图片             |
| 渲染 | 2D sprite blitting    |
| 运行 | `python game.py`      |
| 交互 | 键盘/鼠标事件         |
| 物理 | 简单 2D 碰撞          |

### 3D Assembly (Three.js)

| 方面 | 实现                    |
| ---- | ----------------------- |
| 输出 | index.html (Web 应用)   |
| 资源 | .glb 3D 模型            |
| 渲染 | WebGL (GPU 加速)        |
| 运行 | 浏览器 / HTTP 服务器    |
| 交互 | 轨道控制器 + 键盘       |
| 物理 | 可集成 Rapier/Cannon.js |

## 验证方法

### 1. 运行 3D 生成

```python
# test_3d_generation.py
from pathlib import Path
from autoenv.pipeline.visual.pipeline import VisualPipeline

async def main():
    pipeline = VisualPipeline.create_default(
        image_model="gemini-2.5-flash-image",
        dimension="3d",
        meshy_api_key="msy_vfeMrK0HGYuJir4zK74nfEz2ddn5mNYUrzdU",
        max_3d_assets=3
    )

    await pipeline.run(
        instruction="A 3D puzzle game with boxes and a player",
        output_dir=Path("workspace/test_3d")
    )

import asyncio
asyncio.run(main())
```

### 2. 查看生成结果

```bash
cd workspace/test_3d/game
python -m http.server 8000
# 浏览器打开: http://localhost:8000
```

### 3. 验证项

- [ ] HTML 文件包含所有模型路径
- [ ] 模型正确加载并显示
- [ ] 相机控制正常工作
- [ ] 光照和阴影正确
- [ ] 模型位置合理分布
- [ ] 动画流畅运行

## 与 2D Assembly 的一致性

### 相同点

1. **输入**: 都依赖 strategy.json 和 generated_assets
2. **流程**: Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Assembly
3. **输出结构**: game/目录 + 资源子目录
4. **可运行**: 都生成完整可执行的游戏/场景

### 不同点

1. **资源格式**: PNG vs GLB
2. **技术栈**: Python/Pygame vs JavaScript/WebGL
3. **运行环境**: Python 解释器 vs 浏览器
4. **3D 节点**: 2D 无需 Image3DConvert，3D 必须

## 功能完整性评估

✅ **已实现**:

- 基础 three.js 场景生成
- 自动模型加载和定位
- 光照系统
- 相机控制
- 简单动画
- Agent 增强接口

⚠️ **可改进**:

- 模型定位算法（当前为简单圆形布局）
- 碰撞检测
- 游戏逻辑生成（依赖 agent）
- VR/AR 支持
- 物理引擎集成

🔄 **与 2D 对等**:

- 2D: 生成可玩的 pygame 游戏（有游戏循环、输入处理、碰撞）
- 3D: 生成可交互的 three.js 场景（有渲染循环、轨道控制、模型展示）
- **结论**: 功能对等，但 3D 侧重展示，2D 侧重游戏逻辑

## 总结

### 设计决策

1. **替换而非扩展**: 3D 模式完全替换 AssemblyNode，而非扩展它

   - **理由**: Pygame 和 Three.js 技术栈完全不同，强行统一会增加复杂度

2. **模板 + 动态生成**: 使用 HTML 模板 + 动态填充

   - **理由**: 保证基础场景可用，agent 增强为可选

3. **Agent 可选**: ThreeJSAssemblyNode 先生成基础可用 HTML，agent 增强为可选步骤
   - **理由**: 避免 agent 生成错误导致整个场景不可用

### 与 2D Assembly 对比

| 维度     | 2D Assembly     | 3D Assembly     | 一致性             |
| -------- | --------------- | --------------- | ------------------ |
| 可运行性 | ✅              | ✅              | ✓ 都生成可运行程序 |
| 资源使用 | ✅              | ✅              | ✓ 都使用生成的资源 |
| 交互性   | ✅ 完整游戏逻辑 | ⚠️ 基础场景展示 | △ 2D 更完善        |
| 扩展性   | ⚠️ Pygame 限制  | ✅ Web 生态丰富 | △ 各有优劣         |

**最终答案**: 3D 的 ThreeJSAssembly 与 2D 的 Assembly **功能对等但实现不同**：

- 都生成可运行、可交互的程序
- 都使用 pipeline 生成的资源
- 2D 侧重完整游戏逻辑，3D 侧重场景展示和扩展性
- 3D 通过 agent 增强可达到 2D 的交互复杂度
