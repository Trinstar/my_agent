import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器，负责加载和提供配置项访问"""
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:

        if config_path is None:
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config.yaml"

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        return self._config

    @property
    def config_dict(self) -> Dict[str, Any]:
        """get config dictionary"""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键（如'database.uri'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_mcp_tools_config(self) -> Dict[str, Dict[str, str]]:
        """获取MCP服务器配置"""
        res = self.get('mcp_tools', {})
        if res is None:
            raise ValueError(
                "MCP tools configuration is missing in config file.")
        return res

    def get_func_tools_config(self) -> Dict[str, Any]:
        """获取功能工具配置"""
        res = self.get('func_tools', {})
        if res is None:
            raise ValueError(
                "Function tools configuration is missing in config file.")
        return res

    def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        res = self.get('database', {})
        if res is None:
            raise ValueError(
                "Database configuration is missing in config file.")
        return res

    def get_rag_config(self) -> Dict[str, Any]:
        """获取RAG配置"""
        res = self.get('rag', {})
        if res is None:
            raise ValueError("RAG configuration is missing in config file.")
        return res


config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config_manager
