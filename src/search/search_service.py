from typing import List, Dict, Any, Tuple, Optional
from .search_factory import SearchFactory
from .search_config import SearchConfig
from .search_strategy import ImageProcessor


class SearchService:
    """Main service class for handling searches.
    """
    def __init__(self):
        self.config = SearchConfig()
        self.image_processor = ImageProcessor()

    def text_search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform text-based search
        
        Args:
            query: Search query text
            
        Returns:
            List of search results
        """
        strategy = SearchFactory.get_strategy(
            'text',
            max_retries=self.config.get('max_retries'),
            max_results=self.config.get('max_results')
        )
        return strategy.search(query)

    def image_search(self, query: str, save_path: str, idx: int, conversation_num: int) -> Tuple[str, str]:
        """
        Perform image-based search and save the result
        
        Args:
            query: Search query text
            save_path: Path to save the image
            idx: Image index
            conversation_num: Conversation number
            
        Returns:
            Tuple of (image_url, save_path)
        """
        strategy = SearchFactory.get_strategy(
            'image',
            max_retries=self.config.get('max_retries'),
            max_results=self.config.get('max_results')
        )
        result = strategy.search(query)
        return self.image_processor.save_search_result(result, save_path, idx, conversation_num)

    def fine_search(self,
                   query: str,
                   search_type: str,
                   save_path: str,
                   dataset_name: str,
                   idx: int,
                   conversation_num: int) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Perform a detailed search based on type
        
        Args:
            query: Search query
            search_type: Type of search ('text' or 'image')
            save_path: Path to save results
            dataset_name: Name of dataset
            idx: Search index
            conversation_num: Conversation number
            
        Returns:
            Tuple of (list of image results, list of text results)
        """
        search_images = []
        search_texts = []

        if search_type == 'image':
            image_url, save_path = self.image_search(query, save_path, idx, conversation_num)
            search_images.append((image_url, save_path))
        else:
            results = self.text_search(query)
            search_texts.extend([result.get('body', '') for result in results])

        return search_images, search_texts
