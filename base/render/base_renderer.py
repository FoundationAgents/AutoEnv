# ============================================================
# RENDERER BASE CLASS (Pydantic)
# Purpose: Abstract base class for rendering SemanticView to output
# ============================================================

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Tuple, Union, Literal
import os


class BackgroundColor(BaseModel):
    """纯色背景配置"""
    type: Literal["color"] = "color"
    color: str = Field(default="#000000", description="背景颜色（十六进制）")


class BackgroundImage(BaseModel):
    """图片背景配置"""
    type: Literal["image"] = "image"
    path: str = Field(..., description="背景图片路径")
    mode: Literal["stretch", "tile", "center", "cover", "contain"] = Field(
        default="cover", description="图片填充模式"
    )


# 背景类型联合
Background = Union[BackgroundColor, BackgroundImage]


class RenderConfig(BaseModel):
    """
    渲染器配置类 - 不预设游戏风格。
    
    使用像素坐标系统，物体尺寸由 ObjectData.size 决定。
    
    Attributes:
        asset_path: 素材目录路径
        resolution: 输出分辨率 (width, height)
        background: 背景配置（纯色或图片）
        symbol_map: ASCII 符号映射（ASCIIRenderer 使用）
        grid_size: 网格尺寸（ASCII 渲染使用）
        default_symbol: 未映射物体的默认 ASCII 符号
        empty_symbol: 空格子的 ASCII 符号
    
    Example:
        >>> # 纯色背景
        >>> config = RenderConfig(
        ...     asset_path="./assets",
        ...     resolution=(1920, 1080),
        ...     background=BackgroundColor(color="#1a1a2e")
        ... )
        >>> 
        >>> # 图片背景
        >>> config = RenderConfig(
        ...     asset_path="./assets",
        ...     resolution=(1920, 1080),
        ...     background=BackgroundImage(path="./bg.png", mode="cover")
        ... )
    """
    
    asset_path: str = Field(default="./assets", description="素材目录路径")
    resolution: Tuple[int, int] = Field(
        default=(800, 600), description="输出分辨率 (width, height)"
    )
    background: Background = Field(
        default_factory=lambda: BackgroundColor(color="#000000"),
        description="背景配置（纯色或图片）"
    )
    
    # 素材映射配置
    asset_mapping: Dict[str, str] = Field(
        default_factory=dict, 
        description="素材映射，key 为物体 ID 或类型前缀，value 为素材文件名（不含扩展名）"
    )
    
    # ASCII 渲染专用
    symbol_map: Dict[str, str] = Field(
        default_factory=dict, description="ASCII 符号映射，key 为物体 ID，value 为显示字符"
    )
    default_symbol: str = Field(default="?", description="未映射物体的默认 ASCII 符号")
    empty_symbol: str = Field(default=".", description="空格子的 ASCII 符号")
    grid_size: Optional[Tuple[int, int]] = Field(
        default=None, description="ASCII 网格尺寸 (width, height)"
    )
    
    model_config = {"extra": "allow"}  # 允许扩展自定义字段


class BaseRenderer(ABC):
    """
    渲染器抽象基类。
    
    负责将 SemanticView 转换为最终输出（str/Image）。
    框架提供三个具体实现：ASCIIRenderer、Renderer2D、Renderer3D。
    
    渲染流程：
    1. 创建画布（基于 resolution 或动态计算）
    2. 绘制背景（纯色或图片）
    3. 按 z_index 排序物体，依次渲染
    4. 根据 view_region 裁剪输出
    
    Attributes:
        config: 渲染配置
        _asset_cache: 素材缓存字典
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        初始化渲染器。
        
        Args:
            config: 渲染配置，如果为 None 则使用默认配置
        """
        self.config = config or RenderConfig()
        self._asset_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def render(self, semantic_view: "SemanticView") -> Any:
        """
        渲染主方法。
        
        将 SemanticView 转换为最终输出。
        
        Args:
            semantic_view: 语义视图数据
            
        Returns:
            渲染结果，类型取决于具体实现：
            - ASCIIRenderer: str
            - Renderer2D: PIL.Image
            - Renderer3D: PIL.Image
        """
        pass
    
    def load_asset(self, obj_id: str) -> Any:
        """
        根据 ID 加载素材（带缓存）。
        
        Args:
            obj_id: 物体 ID
            
        Returns:
            加载的素材，类型取决于具体实现
        """
        if obj_id not in self._asset_cache:
            asset = self._load_asset_impl(obj_id)
            if asset is not None:
                self._asset_cache[obj_id] = asset
            return asset
        return self._asset_cache[obj_id]
    
    @abstractmethod
    def _load_asset_impl(self, obj_id: str) -> Any:
        """
        实际加载素材的实现。
        
        Args:
            obj_id: 物体 ID
            
        Returns:
            加载的素材，如果不存在返回 None
        """
        pass
    
    def clear_cache(self) -> None:
        """清空素材缓存"""
        self._asset_cache.clear()
    
    def get_asset_path(self, obj_id: str, extension: str = "") -> str:
        """
        获取素材文件的完整路径。
        
        支持素材映射：
        1. 精确匹配：asset_mapping["player"] -> "horse"
        2. 前缀匹配：asset_mapping["treasure_"] -> "crown"（匹配 treasure_0, treasure_1 等）
        
        Args:
            obj_id: 物体 ID
            extension: 文件扩展名（如 ".png", ".gltf"）
            
        Returns:
            素材文件的完整路径
        """
        # 获取映射后的素材名称
        asset_name = self._resolve_asset_name(obj_id)
        return os.path.join(self.config.asset_path, f"{asset_name}{extension}")
    
    def _resolve_asset_name(self, obj_id: str) -> str:
        """
        解析素材名称，支持精确匹配和前缀匹配。
        
        Args:
            obj_id: 物体 ID
            
        Returns:
            映射后的素材名称，如果没有映射则返回原 obj_id
        """
        mapping = self.config.asset_mapping
        
        # 1. 精确匹配
        if obj_id in mapping:
            return mapping[obj_id]
        
        # 2. 前缀匹配（查找以 "_" 结尾的前缀键）
        for prefix, asset_name in mapping.items():
            if prefix.endswith("_") and obj_id.startswith(prefix):
                return asset_name
        
        # 3. 无匹配，返回原 ID
        return obj_id


# 类型提示的延迟导入
from base.env.semantic_view import SemanticView
