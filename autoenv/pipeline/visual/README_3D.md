# 3D Scene Generation with Three.js

## 概述

在 3D 模式下，AutoEnv pipeline 生成基于 three.js 的可交互 3D 场景，而非 2D 的 pygame 游戏。

## Pipeline 对比

### 2D 模式 (dimension="2d")

```
Analyzer → Strategist → AssetGenerator → BackgroundRemoval → AssemblyNode
输出: game/game.py + game/assets/*.png (Pygame游戏)
```

### 3D 模式 (dimension="3d")

```
Analyzer → Strategist → AssetGenerator → BackgroundRemoval → Image3DConvert → ThreeJSAssemblyNode
输出: game/index.html + game/models/*.glb (Three.js场景)
```

## 功能对比

| 功能     | 2D (Pygame)     | 3D (Three.js)    |
| -------- | --------------- | ---------------- |
| 渲染引擎 | Pygame (Python) | Three.js (WebGL) |
| 资源格式 | PNG 图片        | GLB 3D 模型      |
| 运行环境 | Python 解释器   | 浏览器           |
| 相机控制 | 2D 平面移动     | 3D 轨道/FPS 控制 |
| 光照     | 无/简单         | 完整光照系统     |
| 物理     | 简单碰撞        | 可集成物理引擎   |
| 交互性   | 键盘/鼠标       | 键盘/鼠标/VR     |

## 使用方法

### 生成 3D 场景

```python
from autoenv.pipeline.visual.pipeline import VisualPipeline

pipeline = VisualPipeline.create_default(
    image_model="gemini-2.5-flash-image",
    dimension="3d",  # 启用3D模式
    meshy_api_key="your-meshy-key",
    max_3d_assets=4
)

await pipeline.run(
    instruction="A 3D Sokoban puzzle game",
    output_dir=Path("output/game_3d")
)
```

### 查看生成的 3D 场景

1. **本地 HTTP 服务器** (推荐):

   ```bash
   cd output/game_3d/game
   python -m http.server 8000
   # 打开浏览器访问: http://localhost:8000
   ```

2. **直接打开** (可能受 CORS 限制):
   ```bash
   open output/game_3d/game/index.html
   ```

## 生成文件结构

```
output/game_3d/
├── analysis.json          # 需求分析
├── strategy.json          # 可视化策略
├── assets/                # 原始2D图片
│   ├── player.png
│   ├── box.png
│   └── ...
├── models_3d/             # 转换后的3D模型
│   ├── player.glb
│   ├── box.glb
│   └── ...
├── 3d_timing_log.json    # 转换性能日志
└── game/                  # 可运行的游戏
    ├── index.html         # Three.js场景入口
    └── models/            # 3D模型副本
        ├── player.glb
        ├── box.glb
        └── ...
```

## Three.js 场景功能

### 内置功能

1. **相机控制**:

   - 左键拖动: 旋转视角
   - 右键拖动: 平移
   - 滚轮: 缩放
   - WASD 键: 移动相机目标

2. **光照系统**:

   - 环境光 (AmbientLight)
   - 方向光 (DirectionalLight) + 阴影

3. **模型加载**:

   - GLTFLoader 异步加载所有.glb 模型
   - 进度提示
   - 自动启用阴影

4. **场景布局**:
   - 地面平面
   - 网格辅助
   - 基于策略的自动模型定位

### Agent 增强 (可选)

如果启用了`ThreeJSAssemblyNode`的 agent，会自动:

- 添加游戏逻辑 (移动、碰撞、目标检测)
- 优化模型位置和布局
- 实现动画系统
- 添加游戏状态管理

## 自定义开发

### 修改模板

编辑 `autoenv/pipeline/visual/threejs_template.html`:

```javascript
// 自定义模型位置
if (assetId === "player") {
  model.position.set(0, 0, 0);
  model.scale.setScalar(1.5);
}

// 自定义动画
if (models["box"]) {
  models["box"].rotation.y += 0.01; // 旋转动画
}
```

### 集成物理引擎

```javascript
import RAPIER from "https://cdn.skypack.dev/@dimforge/rapier3d-compat";

// 在animate()中更新物理
world.step();
```

### 添加 VR 支持

```javascript
import { VRButton } from "three/addons/webxr/VRButton.js";
renderer.xr.enabled = true;
document.body.appendChild(VRButton.createButton(renderer));
```

## 性能优化

1. **减少多边形数**: 调整`target_polycount`参数

   ```python
   pipeline = VisualPipeline.create_default(
       dimension="3d",
       target_polycount=5000  # 默认10000
   )
   ```

2. **限制转换数量**: 设置`max_3d_assets`

   ```python
   max_3d_assets=3  # 只转换前3个重要资源
   ```

3. **并行转换**: Image3DConvertNode 自动并行处理所有模型

## 已知限制

1. **Meshy API 配额**: 免费版有转换次数限制
2. **模型质量**: 取决于输入图片质量和 Meshy 转换效果
3. **浏览器兼容性**: 需支持 WebGL 2.0
4. **文件大小**: GLB 模型比 PNG 大，注意带宽

## 故障排除

### 模型加载失败

- 检查`models/`目录是否有.glb 文件
- 确认使用 HTTP 服务器而非直接打开 HTML (CORS 限制)
- 查看浏览器控制台错误信息

### 性能问题

- 降低`target_polycount`
- 减少`max_3d_assets`数量
- 简化模型材质和纹理

### Agent 生成的代码有问题

- ThreeJSAssemblyNode 首先生成基础可用模板
- Agent 增强是可选的，可禁用 agent 参数
- 手动修改 index.html 进行调试
