# ============================================================
# 2D RENDERER
# Purpose: Render SemanticView to 2D image using PIL/Pillow
# ============================================================

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import math

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageFont = None
    np = None

from base.render.base_renderer import (
    BaseRenderer, RenderConfig, BackgroundColor, BackgroundImage
)
from base.env.semantic_view import SemanticView, ObjectData, Transform


class Renderer2D(BaseRenderer):
    """
    2D 图像渲染器。
    
    将 SemanticView 转换为 PIL Image。使用像素坐标系统，
    支持完整的 Transform 变换（包括透视效果）。
    
    渲染流程：
    1. 计算场景边界（基于所有物体位置）
    2. 创建画布并绘制背景（纯色或图片）
    3. 按 z_index 排序物体，依次渲染每个物体
    4. 根据 view_region 裁剪最终输出
    
    素材文件命名约定：
    - {asset_path}/{obj_id}.png
    - 素材使用原始尺寸，或由 ObjectData.size 指定
    
    See test/test_interaction_2d.py for complete usage examples.
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        初始化 2D 渲染器。
        
        Args:
            config: 渲染配置
            
        Raises:
            ImportError: 如果 PIL/Pillow 未安装
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "Renderer2D requires PIL/Pillow and numpy. "
                "Install with: pip install Pillow numpy"
            )
        super().__init__(config)
        self._default_font = None
    
    def render(self, semantic_view: SemanticView) -> "Image.Image":
        """
        渲染 SemanticView 为 PIL Image。
        
        流程：先渲染完整场景，再根据 view_region 裁剪。
        
        Args:
            semantic_view: 语义视图数据
            
        Returns:
            PIL Image 对象
        """
        # 1. 计算场景尺寸和画布尺寸
        scene_bounds = self._calculate_scene_bounds(semantic_view)
        canvas_size = self._determine_canvas_size(semantic_view, scene_bounds)
        
        # 2. 创建画布并绘制背景
        canvas = self._create_background(canvas_size)
        
        # 3. 获取标准化的物体列表并按层级排序
        objects = semantic_view.get_normalized_objects()
        sorted_objects = self._sort_by_layer(objects)
        
        # 4. 绘制每个物体
        for obj in sorted_objects:
            self._draw_object(canvas, obj)
        
        # 5. 根据 view_region 裁剪输出
        output = self._crop_to_view_region(canvas, semantic_view.view_region)
        
        return output
    
    def render_with_overlay(self, semantic_view: SemanticView,
                            overlay_text: Optional[str] = None,
                            overlay_position: Tuple[int, int] = (10, 10),
                            overlay_font_size: int = 12) -> "Image.Image":
        """
        渲染带有文字覆盖层的图像。
        
        Args:
            semantic_view: 语义视图数据
            overlay_text: 覆盖文字
            overlay_position: 文字位置
            font_size: 字体大小（默认 12）
            
        Returns:
            带文字覆盖的 PIL Image
        """
        image = self.render(semantic_view)
        
        if overlay_text:
            draw = ImageDraw.Draw(image)
            font = self._get_font(overlay_font_size)
            
            # 绘制半透明背景
            text_bbox = draw.textbbox(overlay_position, overlay_text, font=font)
            padding = 5
            bg_rect = (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding
            )
            draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
            
            draw.text(overlay_position, overlay_text, fill=(255, 255, 255), font=font)
        
        return image
    
    def _calculate_scene_bounds(self, semantic_view: SemanticView) -> Dict[str, float]:
        """
        计算场景边界（所有物体的包围盒）。
        
        Returns:
            包含 min_x, min_y, max_x, max_y 的字典
        """
        objects = semantic_view.get_normalized_objects()
        
        if not objects:
            return {"min_x": 0, "min_y": 0, "max_x": 800, "max_y": 600}
        
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for obj in objects:
            x, y = self._extract_position(obj.pos)
            
            # 计算物体尺寸
            if obj.size:
                w, h = obj.size
            else:
                # 尝试获取素材尺寸
                sprite = self.load_asset(obj.id)
                if sprite:
                    w, h = sprite.size
                else:
                    w, h = 32, 32  # 默认尺寸
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}
    
    def _determine_canvas_size(self, semantic_view: SemanticView, 
                                scene_bounds: Dict[str, float]) -> Tuple[int, int]:
        """确定画布尺寸"""
        # 优先使用配置的分辨率
        if self.config.resolution:
            return self.config.resolution
        
        # 从 view_region 推断
        view_region = semantic_view.view_region
        if isinstance(view_region, dict):
            if "width" in view_region and "height" in view_region:
                return (int(view_region["width"]), int(view_region["height"]))
        
        # 基于场景边界
        width = int(scene_bounds["max_x"] - scene_bounds["min_x"])
        height = int(scene_bounds["max_y"] - scene_bounds["min_y"])
        
        return (max(width, 100), max(height, 100))
    
    def _create_background(self, size: Tuple[int, int]) -> "Image.Image":
        """创建背景图像（纯色或图片）"""
        width, height = size
        
        background = self.config.background
        
        if isinstance(background, BackgroundImage):
            # 图片背景
            return self._create_image_background(size, background)
        else:
            # 纯色背景（默认）
            color = self._parse_color(background.color)
            return Image.new("RGBA", (width, height), color)
    
    def _create_image_background(self, size: Tuple[int, int], 
                                  bg_config: BackgroundImage) -> "Image.Image":
        """创建图片背景"""
        width, height = size
        
        # 加载背景图片
        try:
            bg_img = Image.open(bg_config.path)
            if bg_img.mode != "RGBA":
                bg_img = bg_img.convert("RGBA")
        except Exception:
            # 加载失败，使用黑色背景
            return Image.new("RGBA", (width, height), (0, 0, 0, 255))
        
        # 根据 mode 处理图片
        mode = bg_config.mode
        
        if mode == "stretch":
            # 拉伸填充
            return bg_img.resize((width, height), Image.Resampling.LANCZOS)
        
        elif mode == "tile":
            # 平铺
            canvas = Image.new("RGBA", (width, height))
            bg_w, bg_h = bg_img.size
            for y in range(0, height, bg_h):
                for x in range(0, width, bg_w):
                    canvas.paste(bg_img, (x, y))
            return canvas
        
        elif mode == "center":
            # 居中（不缩放）
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            bg_w, bg_h = bg_img.size
            x = (width - bg_w) // 2
            y = (height - bg_h) // 2
            canvas.paste(bg_img, (x, y), bg_img)
            return canvas
        
        elif mode == "cover":
            # 覆盖（保持比例，填满画布）
            bg_w, bg_h = bg_img.size
            scale = max(width / bg_w, height / bg_h)
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            bg_img = bg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # 居中裁剪
            x = (new_w - width) // 2
            y = (new_h - height) // 2
            return bg_img.crop((x, y, x + width, y + height))
        
        elif mode == "contain":
            # 包含（保持比例，完整显示）
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            bg_w, bg_h = bg_img.size
            scale = min(width / bg_w, height / bg_h)
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            bg_img = bg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x = (width - new_w) // 2
            y = (height - new_h) // 2
            canvas.paste(bg_img, (x, y), bg_img)
            return canvas
        
        return bg_img.resize((width, height), Image.Resampling.LANCZOS)
    
    def _sort_by_layer(self, objects: List[ObjectData]) -> List[ObjectData]:
        """按层级排序物体（低层级先绘制）"""
        return sorted(objects, key=lambda obj: obj.z_index)
    
    def _draw_object(self, canvas: "Image.Image", obj: ObjectData) -> None:
        """绘制单个物体"""
        # 提取位置（像素坐标）
        screen_x, screen_y = self._extract_position(obj.pos)
        
        # 加载素材
        sprite = self.load_asset(obj.id)
        if sprite is None:
            # 使用默认占位符
            size = obj.size or (32, 32)
            self._draw_placeholder(canvas, int(screen_x), int(screen_y), obj.id, size)
            return
        
        # 确定目标尺寸
        if obj.size:
            target_w, target_h = obj.size
            if sprite.size != (target_w, target_h):
                sprite = sprite.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 应用变换（缩放、透视、旋转）
        sprite = self._transform_sprite(sprite, obj)
        
        # 绘制到画布
        try:
            pos = (int(screen_x), int(screen_y))
            if sprite.mode == "RGBA":
                canvas.paste(sprite, pos, sprite)
            else:
                canvas.paste(sprite, pos)
        except Exception:
            pass  # 忽略绘制错误
    
    def _extract_position(self, pos: Any) -> Tuple[float, float]:
        """提取 2D 位置坐标"""
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return float(pos[0]), float(pos[1])
        elif isinstance(pos, dict):
            return float(pos.get("x", 0)), float(pos.get("y", 0))
        return 0.0, 0.0
    
    def _transform_sprite(self, sprite: "Image.Image", obj: ObjectData) -> "Image.Image":
        """
        应用完整变换，包括透视效果。
        
        变换顺序：缩放 → 透视（X/Y旋转） → 平面旋转（Z旋转）
        """
        transform = obj.transform or Transform()
        
        # 1. 应用缩放
        if transform.scale_x != 1.0 or transform.scale_y != 1.0:
            new_w = int(sprite.width * transform.scale_x)
            new_h = int(sprite.height * transform.scale_y)
            if new_w > 0 and new_h > 0:
                sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 2. 应用 X/Y 轴透视旋转
        if transform.rotation_x != 0.0 or transform.rotation_y != 0.0:
            sprite = self._apply_perspective(sprite, transform.rotation_x, transform.rotation_y)
        
        # 3. 应用 Z 轴旋转（平面旋转）
        if transform.rotation_z != 0.0:
            sprite = sprite.rotate(
                -transform.rotation_z, 
                expand=True, 
                resample=Image.Resampling.BILINEAR
            )
        
        return sprite
    
    def _apply_perspective(self, sprite: "Image.Image", 
                           rot_x: float, rot_y: float) -> "Image.Image":
        """
        应用 X/Y 轴旋转产生透视效果。
        
        Args:
            rot_x: 绕 X 轴旋转角度（正值：顶部远离，底部靠近）
            rot_y: 绕 Y 轴旋转角度（正值：左侧远离，右侧靠近）
            
        Returns:
            变换后的图像
        """
        w, h = sprite.size
        
        # 将角度转换为弧度
        rx = math.radians(rot_x)
        ry = math.radians(rot_y)
        
        # 计算透视强度因子（限制最大角度以避免极端变形）
        max_angle = math.radians(60)
        rx = max(-max_angle, min(max_angle, rx))
        ry = max(-max_angle, min(max_angle, ry))
        
        fx = math.tan(ry) * 0.3  # Y轴旋转影响水平透视
        fy = math.tan(rx) * 0.3  # X轴旋转影响垂直透视
        
        # 定义原始四角点和目标四角点
        # 原始: 左上(0,0), 右上(w,0), 右下(w,h), 左下(0,h)
        src_points = [(0, 0), (w, 0), (w, h), (0, h)]
        
        # 计算透视后的点
        # rotation_y > 0: 左侧远离（缩小），右侧靠近（放大）
        # rotation_x > 0: 顶部远离（缩小），底部靠近（放大）
        dst_points = [
            (w * max(0, fx), h * max(0, fy)),           # 左上
            (w * (1 - max(0, -fx)), h * max(0, fy)),    # 右上
            (w * (1 - max(0, fx)), h * (1 - max(0, -fy))),  # 右下
            (w * max(0, -fx), h * (1 - max(0, -fy)))    # 左下
        ]
        
        try:
            coeffs = self._find_perspective_coeffs(src_points, dst_points)
            return sprite.transform(
                (w, h), Image.Transform.PERSPECTIVE, coeffs,
                resample=Image.Resampling.BICUBIC
            )
        except Exception:
            # 如果透视变换失败，返回原图
            return sprite
    
    def _find_perspective_coeffs(self, src_points: List[Tuple[float, float]], 
                                  dst_points: List[Tuple[float, float]]) -> Tuple:
        """
        计算透视变换的 8 个系数。
        
        使用最小二乘法求解透视变换矩阵。
        """
        matrix = []
        for (sx, sy), (dx, dy) in zip(src_points, dst_points):
            matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
            matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        
        A = np.array(matrix, dtype=np.float64)
        B = np.array([coord for point in src_points for coord in point], dtype=np.float64)
        
        coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return tuple(coeffs)
    
    def _crop_to_view_region(self, canvas: "Image.Image", 
                              view_region: Any) -> "Image.Image":
        """
        根据 view_region 裁剪画布并应用形状遮罩。
        
        支持的 view_region 格式：
        - {"x": int, "y": int, "width": int, "height": int} - 矩形
        - {"shape_type": "circle", "center": (x, y), "radius": r} - 圆形遮罩
        - {"shape_type": "sector", "center": (x, y), "radius": r, "angle_start": a1, "angle_end": a2} - 扇形遮罩
        - {"shape_type": "ring", "center": (x, y), "outer_radius": r1, "inner_radius": r2} - 环形遮罩
        - {"shape_type": "polygon", "vertices": [...]} - 多边形遮罩
        - None: 返回完整画布
        """
        if view_region is None:
            return canvas
        
        if not isinstance(view_region, dict):
            return canvas
        
        canvas_w, canvas_h = canvas.size
        
        # 检查形状类型
        shape_type = view_region.get("shape_type")
        
        # 根据形状类型应用不同的遮罩
        if shape_type == "circle":
            return self._apply_circular_mask(canvas, view_region)
        elif shape_type == "sector":
            return self._apply_sector_mask(canvas, view_region)
        elif shape_type == "ring":
            return self._apply_ring_mask(canvas, view_region)
        elif shape_type == "polygon":
            return self._apply_polygon_mask(canvas, view_region)
        elif shape_type == "rectangle":
            # 矩形视野使用标准裁剪
            return self._apply_rectangular_crop(canvas, view_region)
        
        # 兼容旧格式：{x, y, width, height}
        if "width" in view_region and "height" in view_region:
            return self._apply_rectangular_crop(canvas, view_region)
        
        # 兼容旧格式：{center, radius}
        if "center" in view_region and "radius" in view_region:
            return self._apply_circular_mask(canvas, view_region)
        
        return canvas
    
    def _apply_rectangular_crop(self, canvas: "Image.Image", 
                                  view_region: Dict[str, Any]) -> "Image.Image":
        """应用矩形裁剪"""
        canvas_w, canvas_h = canvas.size
        
        x = int(view_region.get("x", 0))
        y = int(view_region.get("y", 0))
        w = int(view_region.get("width", canvas_w))
        h = int(view_region.get("height", canvas_h))
        
        # 边界检查
        x = max(0, min(x, canvas_w))
        y = max(0, min(y, canvas_h))
        x2 = min(x + w, canvas_w)
        y2 = min(y + h, canvas_h)
        
        if x2 > x and y2 > y:
            return canvas.crop((x, y, x2, y2))
        return canvas
    
    def _apply_circular_mask(self, canvas: "Image.Image", 
                              view_region: Dict[str, Any]) -> "Image.Image":
        """
        应用圆形遮罩。
        
        圆形外的区域将被设为透明（或黑色）。
        """
        center = view_region.get("center")
        radius = view_region.get("radius")
        
        if center is None or radius is None:
            return canvas
        
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        
        # 计算裁剪区域（圆的包围盒）
        canvas_w, canvas_h = canvas.size
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(canvas_w, int(cx + r))
        y2 = min(canvas_h, int(cy + r))
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        # 裁剪到包围盒
        cropped = canvas.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size
        
        # 创建圆形遮罩
        mask = Image.new('L', (crop_w, crop_h), 0)
        draw = ImageDraw.Draw(mask)
        
        # 圆心相对于裁剪区域的位置
        local_cx = cx - x1
        local_cy = cy - y1
        
        # 绘制圆形（白色区域为可见）
        draw.ellipse(
            [local_cx - r, local_cy - r, local_cx + r, local_cy + r],
            fill=255
        )
        
        # 创建输出图像（黑色背景）
        result = Image.new('RGBA', (crop_w, crop_h), (0, 0, 0, 255))
        result.paste(cropped, mask=mask)
        
        return result
    
    def _apply_sector_mask(self, canvas: "Image.Image", 
                            view_region: Dict[str, Any]) -> "Image.Image":
        """
        应用扇形遮罩。
        
        扇形外的区域将被设为透明（或黑色）。
        """
        center = view_region.get("center")
        radius = view_region.get("radius")
        angle_start = view_region.get("angle_start", 0)
        angle_end = view_region.get("angle_end", 360)
        
        if center is None or radius is None:
            return canvas
        
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        
        # 计算裁剪区域（扇形的包围盒）
        canvas_w, canvas_h = canvas.size
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(canvas_w, int(cx + r))
        y2 = min(canvas_h, int(cy + r))
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        # 裁剪到包围盒
        cropped = canvas.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size
        
        # 创建扇形遮罩
        mask = Image.new('L', (crop_w, crop_h), 0)
        draw = ImageDraw.Draw(mask)
        
        # 圆心相对于裁剪区域的位置
        local_cx = cx - x1
        local_cy = cy - y1
        
        # 使用 pieslice 绘制扇形
        # 注意：PIL 的角度是从3点钟方向顺时针计算的
        # 而我们的系统是从12点钟方向（正上方）计算的
        # 需要转换：PIL角度 = 90 - 我们的角度
        pil_start = 90 - angle_end
        pil_end = 90 - angle_start
        
        draw.pieslice(
            [local_cx - r, local_cy - r, local_cx + r, local_cy + r],
            start=pil_start,
            end=pil_end,
            fill=255
        )
        
        # 创建输出图像（黑色背景）
        result = Image.new('RGBA', (crop_w, crop_h), (0, 0, 0, 255))
        result.paste(cropped, mask=mask)
        
        return result
    
    def _apply_ring_mask(self, canvas: "Image.Image", 
                          view_region: Dict[str, Any]) -> "Image.Image":
        """
        应用环形遮罩（圆环）。
        
        环形外和内圆区域将被设为透明（或黑色）。
        """
        center = view_region.get("center")
        outer_radius = view_region.get("outer_radius")
        inner_radius = view_region.get("inner_radius", 0)
        
        if center is None or outer_radius is None:
            return canvas
        
        cx, cy = float(center[0]), float(center[1])
        r_outer = float(outer_radius)
        r_inner = float(inner_radius)
        
        # 计算裁剪区域（外圆的包围盒）
        canvas_w, canvas_h = canvas.size
        x1 = max(0, int(cx - r_outer))
        y1 = max(0, int(cy - r_outer))
        x2 = min(canvas_w, int(cx + r_outer))
        y2 = min(canvas_h, int(cy + r_outer))
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        # 裁剪到包围盒
        cropped = canvas.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size
        
        # 创建环形遮罩
        mask = Image.new('L', (crop_w, crop_h), 0)
        draw = ImageDraw.Draw(mask)
        
        # 圆心相对于裁剪区域的位置
        local_cx = cx - x1
        local_cy = cy - y1
        
        # 先绘制外圆（白色）
        draw.ellipse(
            [local_cx - r_outer, local_cy - r_outer, 
             local_cx + r_outer, local_cy + r_outer],
            fill=255
        )
        
        # 再绘制内圆（黑色，挖空）
        if r_inner > 0:
            draw.ellipse(
                [local_cx - r_inner, local_cy - r_inner, 
                 local_cx + r_inner, local_cy + r_inner],
                fill=0
            )
        
        # 创建输出图像（黑色背景）
        result = Image.new('RGBA', (crop_w, crop_h), (0, 0, 0, 255))
        result.paste(cropped, mask=mask)
        
        return result
    
    def _apply_polygon_mask(self, canvas: "Image.Image", 
                             view_region: Dict[str, Any]) -> "Image.Image":
        """
        应用多边形遮罩。
        
        多边形外的区域将被设为透明（或黑色）。
        """
        vertices = view_region.get("vertices")
        
        if not vertices or len(vertices) < 3:
            return canvas
        
        # 计算多边形的包围盒
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        canvas_w, canvas_h = canvas.size
        x1 = max(0, int(min_x))
        y1 = max(0, int(min_y))
        x2 = min(canvas_w, int(max_x))
        y2 = min(canvas_h, int(max_y))
        
        if x2 <= x1 or y2 <= y1:
            return canvas
        
        # 裁剪到包围盒
        cropped = canvas.crop((x1, y1, x2, y2))
        crop_w, crop_h = cropped.size
        
        # 创建多边形遮罩
        mask = Image.new('L', (crop_w, crop_h), 0)
        draw = ImageDraw.Draw(mask)
        
        # 将顶点转换为相对于裁剪区域的坐标
        local_vertices = [(v[0] - x1, v[1] - y1) for v in vertices]
        
        # 绘制多边形（白色区域为可见）
        draw.polygon(local_vertices, fill=255)
        
        # 创建输出图像（黑色背景）
        result = Image.new('RGBA', (crop_w, crop_h), (0, 0, 0, 255))
        result.paste(cropped, mask=mask)
        
        return result
    
    def _draw_placeholder(self, canvas: "Image.Image", x: int, y: int, 
                          obj_id: str, size: Tuple[int, int]) -> None:
        """绘制占位符（当素材不存在时）"""
        draw = ImageDraw.Draw(canvas)
        w, h = size
        
        # 绘制彩色方块
        color = self._id_to_color(obj_id)
        draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color, outline=(0, 0, 0))
        
        # 绘制 ID 首字母
        if obj_id:
            font_size = min(w, h) // 2
            font = self._get_font(max(8, font_size))
            text = obj_id[0].upper()
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_x = x + (w - text_w) // 2
            text_y = y + (h - text_h) // 2
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    def _id_to_color(self, obj_id: str) -> Tuple[int, int, int, int]:
        """根据 ID 生成确定性颜色"""
        if not obj_id:
            return (128, 128, 128, 255)
        
        hash_val = hash(obj_id)
        r = (hash_val & 0xFF0000) >> 16
        g = (hash_val & 0x00FF00) >> 8
        b = hash_val & 0x0000FF
        
        # 确保颜色不太暗
        r = max(r, 64)
        g = max(g, 64)
        b = max(b, 64)
        
        return (r, g, b, 255)
    
    def _parse_color(self, color: str) -> Tuple[int, int, int, int]:
        """解析颜色字符串"""
        if color.startswith("#"):
            color = color[1:]
            if len(color) == 6:
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
                return (r, g, b, 255)
            elif len(color) == 8:
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
                a = int(color[6:8], 16)
                return (r, g, b, a)
        return (0, 0, 0, 255)
    
    def _get_font(self, size: int = 12) -> "ImageFont.FreeTypeFont":
        """获取字体"""
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (IOError, OSError):
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except (IOError, OSError):
                return ImageFont.load_default()
    
    def _load_asset_impl(self, obj_id: str) -> Optional["Image.Image"]:
        """加载图片素材"""
        # 尝试多种文件扩展名
        extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp"]
        
        for ext in extensions:
            path = self.get_asset_path(obj_id, ext)
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    # 转换为 RGBA 以支持透明度
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    return img
                except Exception:
                    continue
        
        return None
