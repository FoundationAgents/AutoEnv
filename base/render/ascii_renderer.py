# ============================================================
# ASCII RENDERER
# Purpose: Render SemanticView to ASCII string representation
# ============================================================

from typing import Any, Dict, List, Optional, Tuple, Union
from base.render.base_renderer import BaseRenderer, RenderConfig
from base.env.semantic_view import SemanticView, ObjectData


class ASCIIRenderer(BaseRenderer):
    """
    ASCII 字符渲染器。
    
    将 SemanticView 转换为 ASCII 字符串表示。
    适用于基于文本的游戏和调试输出。
    
    使用 RenderConfig 中的以下字段：
    - symbol_map: 物体 ID 到字符的映射
    - grid_size: 网格尺寸
    - default_symbol: 未映射物体的默认符号
    - empty_symbol: 空格子的符号
    
    See test/test_interaction_ascii.py for complete usage examples.
    """
    
    def render(self, semantic_view: SemanticView) -> str:
        """
        渲染 SemanticView 为 ASCII 字符串。
        
        Args:
            semantic_view: 语义视图数据
            
        Returns:
            ASCII 字符串表示
        """
        # 确定网格尺寸
        grid_size = self._determine_grid_size(semantic_view)
        if grid_size is None:
            return self._render_list_format(semantic_view)
        
        width, height = grid_size
        
        # 初始化网格
        grid = [[self.config.empty_symbol for _ in range(width)] for _ in range(height)]
        
        # 获取标准化的物体列表
        objects = semantic_view.get_normalized_objects()
        
        # 按 z_index 排序（低层级先放置，高层级覆盖）
        sorted_objects = sorted(objects, key=lambda obj: obj.z_index)
        
        # 放置物体
        for obj in sorted_objects:
            x, y = self._extract_2d_pos(obj.pos)
            if x is None or y is None:
                continue
            
            # 边界检查
            if 0 <= x < width and 0 <= y < height:
                symbol = self._get_symbol(obj)
                grid[y][x] = symbol
        
        # 转换为字符串
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)
    
    def render_with_info(self, semantic_view: SemanticView, 
                         header: Optional[str] = None,
                         footer: Optional[str] = None) -> str:
        """
        渲染带有额外信息的输出。
        
        Args:
            semantic_view: 语义视图数据
            header: 头部信息（显示在网格上方）
            footer: 尾部信息（显示在网格下方）
            
        Returns:
            带头尾信息的 ASCII 字符串
        """
        parts = []
        
        if header:
            parts.append(header)
            parts.append("")
        
        parts.append(self.render(semantic_view))
        
        if footer:
            parts.append("")
            parts.append(footer)
        
        return "\n".join(parts)
    
    def _determine_grid_size(self, semantic_view: SemanticView) -> Optional[Tuple[int, int]]:
        """
        确定网格尺寸。
        
        优先使用配置中的 grid_size，否则从 view_region 推断。
        """
        if self.config.grid_size:
            return self.config.grid_size
        
        view_region = semantic_view.view_region
        if view_region is None:
            return None
        
        # 尝试从 view_region 推断尺寸
        if isinstance(view_region, dict):
            # 格式: {width: int, height: int}
            if "width" in view_region and "height" in view_region:
                return (int(view_region["width"]), int(view_region["height"]))
            
            # 格式: {size: [width, height]}
            if "size" in view_region and isinstance(view_region["size"], (list, tuple)):
                size = view_region["size"]
                return (int(size[0]), int(size[1]))
            
            # 格式: {center: [x, y], radius: int} - 计算边界框
            if "center" in view_region and "radius" in view_region:
                radius = int(view_region["radius"])
                size = radius * 2 + 1
                return (size, size)
        
        # 从物体位置推断
        objects = semantic_view.get_normalized_objects()
        if objects:
            max_x, max_y = 0, 0
            for obj in objects:
                x, y = self._extract_2d_pos(obj.pos)
                if x is not None and y is not None:
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
            if max_x > 0 or max_y > 0:
                return (max_x + 1, max_y + 1)
        
        return None
    
    def _extract_2d_pos(self, pos: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        从位置数据中提取 2D 坐标。
        
        支持多种格式：
        - [x, y] / (x, y)
        - {"x": x, "y": y}
        - [x, y, z] (忽略 z)
        """
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
        
        if isinstance(pos, dict):
            x = pos.get("x")
            y = pos.get("y")
            if x is not None and y is not None:
                return int(x), int(y)
        
        return None, None
    
    def _get_symbol(self, obj: ObjectData) -> str:
        """
        获取物体的显示符号。
        
        查找顺序：
        1. 物体额外字段中的 symbol
        2. symbol_map 中的精确匹配
        3. symbol_map 中的前缀匹配（如 "wall_1" 匹配 "wall"）
        4. 默认符号
        """
        obj_id = obj.id
        
        # 检查额外字段中是否有 symbol
        if hasattr(obj, 'model_extra') and obj.model_extra:
            if "symbol" in obj.model_extra:
                return str(obj.model_extra["symbol"])
        
        # 精确匹配
        if obj_id in self.config.symbol_map:
            return self.config.symbol_map[obj_id]
        
        # 前缀匹配
        for prefix, symbol in self.config.symbol_map.items():
            if obj_id.startswith(prefix):
                return symbol
        
        # 默认符号
        return self.config.default_symbol
    
    def _render_list_format(self, semantic_view: SemanticView) -> str:
        """
        无法确定网格时，渲染为列表格式。
        """
        objects = semantic_view.get_normalized_objects()
        lines = ["Objects:"]
        for obj in objects:
            symbol = self._get_symbol(obj)
            lines.append(f"  [{symbol}] {obj.id} at {obj.pos}")
        return "\n".join(lines)
    
    def _load_asset_impl(self, obj_id: str) -> Any:
        """
        ASCII 渲染器不需要加载素材文件，直接返回 symbol_map 中的映射。
        """
        return self.config.symbol_map.get(obj_id, self.config.default_symbol)
    
    def get_legend(self) -> str:
        """
        获取符号图例说明。
        
        Returns:
            符号映射的文字说明
        """
        if not self.config.symbol_map:
            return "No symbol mapping defined."
        
        lines = ["Legend:"]
        for obj_id, symbol in self.config.symbol_map.items():
            lines.append(f"  {symbol} = {obj_id}")
        lines.append(f"  {self.config.empty_symbol} = empty")
        lines.append(f"  {self.config.default_symbol} = unknown")
        return "\n".join(lines)
