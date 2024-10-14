import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict
from recipe_recommender.data_processor import DataProcessor
from recipe_recommender.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class RecipeRecommender:
    """
    Class for recommending recipes based on input ingredients.
    """

    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the RecipeRecommender with a DataProcessor.

        Args:
            data_processor (DataProcessor): An instance of DataProcessor for data loading and processing.
        """
        self.data_processor = data_processor
        self.data = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.text_preprocessor = TextPreprocessor()

    def initialize(self):
        """
        Initialize the recommender by loading and processing data, and vectorizing ingredients.
        """
        logger.info("Initializing RecipeRecommender...")
        self.data = self.data_processor.load_and_preprocess_data()
        self._vectorize_ingredients()
        logger.info("RecipeRecommender initialized successfully.")

    def _vectorize_ingredients(self):
        """
        Vectorize the ingredients using TF-IDF.
        """
        logger.info("Vectorizing ingredients...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['ingredients'])
        logger.info("Ingredients vectorized successfully.")

    def recommend_recipes(self, input_ingredients: str, n: int = 5) -> str:
        """
        Recommend recipes based on input ingredients.

        Args:
            input_ingredients (str): A string of input ingredients.
            n (int): Number of recipes to recommend. Default is 5.

        Returns:
            str: A formatted string containing recommended recipes.
        """
        logger.info(f"Recommending recipes for input: {input_ingredients}")
        preprocessed_input = self.text_preprocessor.preprocess_text(input_ingredients)
        input_vector = self.vectorizer.transform([preprocessed_input])
        cosine_similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[:-n-1:-1]
        
        recommended_recipes = []
        for i, index in enumerate(top_indices, start=1):
            recipe = self.data.iloc[index]
            recipe_dict = {
                "Title": recipe['title'],
                "Ingredients": recipe['ingredients'],
                "Instructions": recipe['instructions']
            }
            recipe_md = self._format_recipe(i, recipe_dict)
            recommended_recipes.append(recipe_md)
        
        return "\n\n" + "\n\n".join(recommended_recipes)

    @staticmethod
    def _format_recipe(index: int, recipe: Dict[str, str]) -> str:
        """
        Format a recipe for output.

        Args:
            index (int): The index of the recipe in the recommendation list.
            recipe (Dict[str, str]): A dictionary containing recipe information.

        Returns:
            str: A formatted string representing the recipe.
        """
        ingredients_list = recipe['Ingredients'].split(',') if isinstance(recipe['Ingredients'], str) else recipe['Ingredients']
        formatted_ingredients = '\n'.join(f"- {ingredient.strip()}" for ingredient in ingredients_list)
        
        instructions_list = recipe['Instructions'].split('.') if isinstance(recipe['Instructions'], str) else recipe['Instructions'].split('\n')
        formatted_instructions = '\n'.join(f"{i+1}. {instruction.strip()}" for i, instruction in enumerate(instructions_list) if instruction.strip())
        
        return f"""
{index}. {recipe['Title']}
Ingredients:
{formatted_ingredients}
Instructions:
{formatted_instructions}
"""