"""
Strategy pattern for search options
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from duckduckgo_search import DDGS
import time
from loguru import logger
import requests
from io import BytesIO
from PIL import Image
import os


class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    def __init__(self, max_retries: int = 5, max_results: int = 5):
        self.max_retries = max_retries
        self.max_results = max_results

    @abstractmethod
    def search(self, query: str) -> Any:
        """Execute the search strategy"""
        pass

    def _retry_operation(self, operation):
        """Template method for retry logic"""
        for i in range(self.max_retries):
            try:
                return operation()
            except Exception as e:
                logger.error(f"Attempt {i + 1} failed: {e}")
                if i < self.max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error("All retries failed.")
                    raise ValueError(e)


class TextSearchStrategy(SearchStrategy):
    """Strategy for text-based search"""
    def search(self, query: str) -> List[Dict[str, str]]:
        with DDGS() as ddgs:
            return self._retry_operation(
                lambda: ddgs.text(
                    query,
                    safesearch='Off',
                    max_results=self.max_results
                )
            )


class ImageSearchStrategy(SearchStrategy):
    """Strategy for image-based search"""
    def search(self, query: str) -> Dict[str, Any]:
        with DDGS() as ddgs:
            return self._retry_operation(
                lambda: ddgs.images(
                    query,
                    safesearch='Off',
                    max_results=self.max_results
                )[0]
            )


class ImageProcessor:
    """Handles image processing and saving"""
    @staticmethod
    def save_search_result(search_result: Dict[str, Any], save_path: str, idx: int, conversation_num: int) -> Tuple[str, str]:
        image_url = search_result['image']
        
        response = requests.get(image_url)
        response.raise_for_status()
        
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        
        save_image_path = os.path.join(
            save_path,
            f'{idx}_{conversation_num}_{search_result.get("position", "0")}.png'
        )
        image.save(save_image_path, format='PNG')
        
        return image_url, save_image_path
