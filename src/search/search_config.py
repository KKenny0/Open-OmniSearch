"""
Singleton Pattern for Configuration
"""
from typing import Dict, Any
from threading import Lock


class SearchConfig:
    """Singleton configuration class for search settings"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config: Dict[str, Any] = {
                'max_retries': 5,
                'max_results': 5,
                'safesearch': 'Off',
                'timeout': 30
            }
            self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        with self._lock:
            self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        with self._lock:
            self._config.update(config_dict)
