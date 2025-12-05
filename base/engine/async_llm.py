# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Zhaoyang & didi
# @Desc    :
import os
import re
import yaml

from openai import AsyncOpenAI

from pathlib import Path
from typing import Dict, Optional, Any
from base.engine.cost_monitor import record_cost
from base.engine.logs import logger, LogLevel

class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.top_p = config.get("top_p", 1)

class LLMsConfig:
    """Configuration manager for multiple LLM configurations"""
    
    _instance = None  # For singleton pattern if needed
    _default_config = None
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with an optional configuration dictionary"""
        self.configs = config_dict or {}
    
    @classmethod
    def default(cls):
        """Get or create a default configuration from YAML file"""
        if cls._default_config is None:
            config_data: Optional[Dict[str, Any]] = None

            config_paths = [
                Path("config/global_config.yaml"),
                Path("config/global_config2.yaml"),
                Path("./config/global_config.yaml"),
                Path("config/model_config.yaml"),
            ]

            config_file = next((path for path in config_paths if path.exists() and path.stat().st_size > 0), None)

            if config_file is not None:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = cls._load_config_from_env()

            if not config_data:
                raise FileNotFoundError(
                    "No default configuration file found in the expected locations and no environment-based fallback is configured."
                )

            if "models" in config_data:
                config_data = config_data["models"] or {}

            cls._default_config = cls(config_data)

        return cls._default_config

    @classmethod
    def _load_config_from_env(cls) -> Optional[Dict[str, Any]]:
        """Build configuration from environment variables when no YAML file is present."""
        inline_config = os.getenv("AUTOENV_MODEL_CONFIG_JSON")
        if inline_config:
            try:
                data = yaml.safe_load(inline_config)
            except yaml.YAMLError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    "Failed to parse AUTOENV_MODEL_CONFIG_JSON; falling back to explicit env vars.",
                )
            else:
                if isinstance(data, dict):
                    return data.get("models", data) or {}

        api_key = os.getenv("AUTOENV_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        base_url = (
            os.getenv("AUTOENV_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    f"Invalid float value for {name}: {raw!r}; using {default}.",
                )
                return default

        temperature = _get_float("AUTOENV_OPENAI_TEMPERATURE", 1)
        top_p = _get_float("AUTOENV_OPENAI_TOP_P", 1)

        models_env = os.getenv("AUTOENV_OPENAI_MODELS", "o3")
        models = [m.strip() for m in models_env.split(",") if m.strip()]

        if not models:
            models = ["o3"]

        config: Dict[str, Any] = {}
        for model_name in models:
            normalized = model_name.upper().replace('-','_').replace('/','_')
            env_key_name = f"AUTOENV_{normalized}_API_KEY"
            env_base_name = f"AUTOENV_{normalized}_BASE_URL"

            model_api_key = os.getenv(env_key_name, api_key)
            model_base_url = os.getenv(env_base_name, base_url)

            config[model_name] = {
                "api_key": model_api_key,
                "base_url": model_base_url,
                "temperature": temperature,
                "top_p": top_p,
            }

        return config
    
    def get(self, llm_name: str) -> LLMConfig:
        """Get the configuration for a specific LLM by name"""
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        
        config = self.configs[llm_name]
        
        # Create a config dictionary with the expected keys for LLMConfig
        llm_config = {
            "model": llm_name,  # Use the key as the model name
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),  # Map api_key to key
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1)  # Add top_p parameter
        }
        
        # Create and return an LLMConfig instance with the specified configuration
        return LLMConfig(llm_config)
    
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add or update a configuration"""
        self.configs[name] = config
    
    def get_all_names(self) -> list:
        """Get names of all available LLM configurations"""
        return list(self.configs.keys())
    
class ModelPricing:
    """Pricing information for different models in USD per 1K tokens"""
    PRICES = {
        # openai: https://platform.openai.com/docs/pricing
        # anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "o3": {"input": 0.002, "output": 0.008},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5": {"input": 0.00125, "output": 0.01},
        "gpt-5-mini": {"input":0.00025, "output": 0.002},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "moonshotai/kimi-k2": {"input": 0.000296, "output": 0.001185}, 
        "deepseek/deepseek-chat-v3.1": {"input":0.00025 , "output":0.001},
        "deepseek-chat": {"input":0.00025 , "output":0.001},
        "z-ai/glm-4.5": {"input": 0.00033, "output": 0.00132},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-4-5-haiku": {"input": 0.00088, "output": 0.0044},
        "claude-4-sonnet-20250514": {"input": 0.003, "output": 0.015},
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.000252},
        "gemini-2.5-flash-image": {"input": 0.0003, "output": 0.03},
    }


    @classmethod
    def get_price(cls, model_name, token_type):
        """Get the price per 1K tokens for a specific model and token type (input/output)"""
        # Try to find exact match first
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        
        # Try to find a partial match (e.g., if model name contains version numbers)
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        
        # Return default pricing if no match found
        return 0

class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    
    def add_usage(self, model, input_tokens, output_tokens):
        """Add token usage for a specific API call"""
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        
        return usage_record
    
    def get_summary(self):
        """Get a summary of token usage and costs"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }

class AsyncLLM:
    def __init__(self, config, system_msg:str = None, max_completion_tokens:int = None):
        """
        Initialize the AsyncLLM with a configuration
        
        Args:
            config: Either an LLMConfig instance or a string representing the LLM name
                   If a string is provided, it will be looked up in the default configuration
            system_msg: Optional system message to include in all prompts
            max_tokens: Optional maximum number of tokens to generate
        """
        # Handle the case where config is a string (LLM name)
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        
        # At this point, config should be an LLMConfig instance
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker()
        self.max_completion_tokens = max_completion_tokens
        
    async def __call__(self, prompt, max_tokens=None):
        """Send a prompt to the LLM. Accepts text (str) or multimodal payloads (list)."""
        message = []
        if self.sys_msg is not None:
            message.append({
                "content": self.sys_msg,
                "role": "system"
            })

        # Support plain text prompts and multimodal payloads (list)
        if isinstance(prompt, str):
            message.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            message.append({"role": "user", "content": prompt})
        else:
            raise ValueError(f"prompt must be str or list, got {type(prompt)}")

        # Prefer to use the max_tokens argument passed to the function; if it is None, use the instance variable.
        tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens

        # Claude models via Bedrock don't support both temperature and top_p
        is_claude = "claude" in self.config.model.lower()
        sampling_params = (
            {"temperature": self.config.temperature}
            if is_claude
            else {"temperature": self.config.temperature, "top_p": self.config.top_p}
        )

        if tokens_to_use is not None and "o3" in self.config.model:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                max_completion_tokens=tokens_to_use,
                **sampling_params,
            )
        # Only gpt-series support max_completion_tokens.
        elif tokens_to_use is not None and "o3" not in self.config.model:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                max_tokens=tokens_to_use,
                **sampling_params,
            )
        else:
            response = await self.aclient.chat.completions.create(
                model=self.config.model,
                messages=message,
                **sampling_params,
            )

        # Extract token usage from response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Track token usage and calculate cost
        usage_record = self.usage_tracker.add_usage(
            self.config.model,
            input_tokens,
            output_tokens
        )

        # Report to global cost monitor if active
        record_cost(self.config.model, input_tokens, output_tokens, usage_record["total_cost"])

        # Return text or multimodal content (API returns content as provided)
        ret = response.choices[0].message.content
        logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")

        return ret
    
    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        return self.usage_tracker.get_summary()

    async def generate_text_to_image(self, prompt: str) -> dict[str, Any]:
        """
        Text-to-image generation.

        Args:
            prompt: Text prompt for image generation

        Returns:
            {
                'success': bool,
                'image_base64': str | None,
                'prompt': str,
                'error': str | None
            }
        """
        try:
            response = await self(prompt)
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }

            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }

    async def generate_image_to_image(
        self, prompt: str, reference_images: list[str]
    ) -> dict[str, Any]:
        """
        Image-to-image generation with style references.

        Args:
            prompt: Text prompt for image generation
            reference_images: List of base64-encoded reference images

        Returns:
            {
                'success': bool,
                'image_base64': str | None,
                'prompt': str,
                'error': str | None
            }
        """
        try:
            content = []
            for img_b64 in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
            content.append({"type": "text", "text": prompt})

            response = await self(content)
            image_b64 = self._extract_image_from_response(response)

            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }

            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }

    def _extract_image_from_response(self, response: str) -> str | None:
        """Extract base64 image from LLM response."""
        if not isinstance(response, str):
            return None
        match = re.search(r"data:image/[^;]+;base64,([^)]+)", response)
        return match.group(1) if match else None


def create_llm_instance(llm_config) -> AsyncLLM:
    """
    Create an AsyncLLM instance using the provided configuration
    
    Args:
        llm_config: Either an LLMConfig instance, a dictionary of configuration values,
                            or a string representing the LLM name to look up in default config
    
    Returns:
        An instance of AsyncLLM configured according to the provided parameters
    """
    # Case 1: llm_config is already an LLMConfig instance
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config)
    
    # Case 2: llm_config is a string (LLM name)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config)  # AsyncLLM constructor handles lookup
    
    # Case 3: llm_config is a dictionary
    elif isinstance(llm_config, dict):
        # Create an LLMConfig instance from the dictionary
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config)
    
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
