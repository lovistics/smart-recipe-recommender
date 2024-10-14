import json
import pandas as pd
import logging
from typing import List, Dict
from recipe_recommender.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for loading and processing recipe data from JSON files.
    """

    def __init__(self, file_paths: List[str]):
        """
        Initialize the DataProcessor with file paths.

        Args:
            file_paths (List[str]): List of paths to JSON files containing recipe data.
        """
        self.file_paths = file_paths
        self.text_preprocessor = TextPreprocessor()

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load data from JSON files and preprocess it.

        Returns:
            pd.DataFrame: Preprocessed DataFrame containing recipe data.
        """
        dfs = []
        for file_path in self.file_paths:
            df = self._load_data(file_path)
            dfs.append(df)
        
        data = pd.concat(dfs, ignore_index=True)
        data['ingredients'] = data['ingredients'].apply(self.text_preprocessor.preprocess_text)
        
        return data

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a single JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            pd.DataFrame: DataFrame containing recipe data from the file.
        """
        logger.info(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df[['title', 'ingredients', 'instructions']].dropna(how='any')
        df['ingredients'] = df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        df['ingredients'] = df['ingredients'].apply(self.text_preprocessor.remove_digits)
        df['ingredients'] = df['ingredients'].apply(self.text_preprocessor.remove_ads)
        logger.info(f"Data loaded from {file_path}. Recipes: {len(df)}")
        return df