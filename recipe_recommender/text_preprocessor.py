import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Class for preprocessing text data.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor with NLTK resources.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text by tokenizing, removing stopwords, and lemmatizing.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        words = word_tokenize(text.lower())
        words = [word.translate(str.maketrans('', '', string.punctuation.replace(',', ''))) for word in words]
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    @staticmethod
    def remove_digits(s: str) -> str:
        """
        Remove digits from the input string.

        Args:
            s (str): Input string.

        Returns:
            str: String with digits removed.
        """
        return re.sub(r'\d+', '', s)

    @staticmethod
    def remove_ads(s: str) -> str:
        """
        Remove 'ADVERTISEMENT' from the input string.

        Args:
            s (str): Input string.

        Returns:
            str: String with 'ADVERTISEMENT' removed.
        """
        return re.sub(r'ADVERTISEMENT', '', s)