import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gradio as gr
import logging
import yaml
from recipe_recommender.data_processor import DataProcessor
from recipe_recommender.recipe_recommender import RecipeRecommender
from recipe_recommender.utils import download_nltk_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """
    Load configuration from YAML file.
    """
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()

    # Initialize DataProcessor and RecipeRecommender
    data_processor = DataProcessor(config['file_paths'])
    recommender = RecipeRecommender(data_processor)
    recommender.initialize()

    def recommend_wrapper(input_ingredients: str) -> str:
        """
        Wrapper function for recipe recommendation to use with Gradio.
        """
        return recommender.recommend_recipes(input_ingredients)

    # Set up Gradio interface
    iface = gr.Interface(
        fn=recommend_wrapper,
        inputs=gr.Textbox(lines=2, label="Enter Ingredients to see recommended recipes:"),
        outputs="textbox",
        title="Recipe Recommender",
        description="Enter ingredients and get recipe recommendations!"
    )
    
    iface.launch(share=True)

if __name__ == "__main__":
    download_nltk_data()
    main()