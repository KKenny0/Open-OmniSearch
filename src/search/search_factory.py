"""
Factory pattern for Agent Creation
"""
from typing import Dict, Type
from .search_strategy import SearchStrategy, TextSearchStrategy, ImageSearchStrategy


class SearchFactory:
    """Factory for creating search strategy instances"""
    _strategies: Dict[str, Type[SearchStrategy]] = {
        'text': TextSearchStrategy,
        'image': ImageSearchStrategy
    }

    @classmethod
    def get_strategy(cls, strategy_type: str, **kwargs) -> SearchStrategy:
        """
        Create and return a search strategy instance
        
        Args:
            strategy_type: Type of search strategy ('text' or 'image')
            **kwargs: Additional arguments to pass to the strategy constructor
        
        Returns:
            An instance of the requested search strategy
        
        Raises:
            ValueError: If strategy_type is not supported
        """
        strategy_class = cls._strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unsupported search strategy: {strategy_type}")
        
        return strategy_class(**kwargs)
