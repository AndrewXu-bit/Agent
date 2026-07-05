"""
配置管理模块 - 从 YAML 文件加载和管理应用配置。

支持环境变量替换、配置验证和运行时访问。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """应用配置管理器。
    
    从 config.yaml 加载配置，支持环境变量替换。
    
    Attributes:
        data: 配置数据字典。
        config_path: 配置文件路径。
    """
    
    _instance: Optional[Config] = None
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器。
        
        Args:
            config_path: 配置文件路径，默认为项目根目录的 config.yaml。
        """
        if config_path is None:
            # 自动查找项目根目录的 config.yaml
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_path = str(project_root / "config.yaml")
        
        self.config_path = config_path
        self.data = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载并解析 YAML 配置文件。
        
        Args:
            config_path: 配置文件路径。
            
        Returns:
            配置数据字典。
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        
        # 处理环境变量替换
        return self._resolve_env_vars(raw_data)
    
    def _resolve_env_vars(self, data: Any) -> Any:
        """递归解析配置中的环境变量引用。
        
        支持 ${VAR_NAME} 格式的变量引用。
        
        Args:
            data: 待解析的数据。
            
        Returns:
            解析后的数据。
        """
        if isinstance(data, dict):
            return {k: self._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(item) for item in data]
        elif isinstance(data, str):
            # 匹配 ${VAR_NAME} 模式
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, data)
            if matches:
                result = data
                for var_name in matches:
                    env_value = os.getenv(var_name, "")
                    result = result.replace(f"${{{var_name}}}", env_value)
                return result
            return data
        else:
            return data
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径。
        
        Args:
            key: 配置键，如 "llm.provider" 或 "agent.name"。
            default: 默认值。
            
        Returns:
            配置值。
        """
        keys = key.split(".")
        value = self.data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取 LLM 配置。
        
        Returns:
            LLM 配置字典。
        """
        return self.data.get("llm", {})
    
    def get_agent_config(self) -> Dict[str, Any]:
        """获取 Agent 配置。
        
        Returns:
            Agent 配置字典。
        """
        return self.data.get("agent", {})
    
    def get_tools_config(self) -> Dict[str, Any]:
        """获取工具配置。
        
        Returns:
            工具配置字典。
        """
        return self.data.get("tools", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置。
        
        Returns:
            日志配置字典。
        """
        return self.data.get("logging", {})
    
    def reload(self) -> None:
        """重新加载配置文件。"""
        self.data = self._load_config(self.config_path)
    
    @classmethod
    def instance(cls, config_path: Optional[str] = None) -> Config:
        """获取配置单例实例。
        
        Args:
            config_path: 配置文件路径。
            
        Returns:
            Config 实例。
        """
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """重置单例（用于测试）。"""
        cls._instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置实例的快捷函数。
    
    Args:
        config_path: 配置文件路径。
        
    Returns:
        Config 实例。
    """
    return Config.instance(config_path)
