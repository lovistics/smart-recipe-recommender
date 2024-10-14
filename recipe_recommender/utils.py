import os
import nltk
import logging

logger = logging.getLogger(__name__)

def download_nltk_data():
    """
    Download required NLTK data.
    """
    logger.info("Downloading NLTK data...")
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
    nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
    nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)
    logger.info("NLTK data downloaded successfully.")
