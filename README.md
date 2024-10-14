# Smart Recipe Recommender

Smart Recipe Recommender is a Python-based application that suggests recipes based on input ingredients. It uses natural language processing and machine learning techniques to process recipe data and provide relevant recommendations.

## Features

- Load and preprocess recipe data from multiple JSON files
- Utilize TF-IDF vectorization for ingredient analysis
- Implement cosine similarity for recipe matching
- Provide a user-friendly interface using Gradio

## Project Structure

```
├── recipe_recommender/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── recipe_recommender.py
│   ├── text_preprocessor.py
│   └── utils.py
├── scripts/
│   └── run_recommender.py
├── data/
│   ├── raw/
│   │   ├── recipes_raw_nosource_ar.json
│   │   ├── recipes_raw_nosource_epi.json
│   │   └── recipes_raw_nosource_fn.json
├── config/
│   └── config.yaml
├── requirements.txt
├── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/lovistics/smart-recipe-recommender.git
   cd smart-recipe-recommender
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your recipe data files (JSON format) are in the `data/raw/` directory.

2. Update the `config/config.yaml` file with the correct file paths if necessary.

3. Run the application:
   ```
   python scripts/run_recommender.py
   ```

4. Open the provided Gradio interface URL in your web browser.

5. Enter ingredients in the text box and click "Submit" to get recipe recommendations.

## Data Format

The application expects JSON files with recipe data in the following format:

```json
{
  "recipe_id": {
    "title": "Recipe Title",
    "ingredients": ["ingredient 1", "ingredient 2", ...],
    "instructions": "Step 1. ... Step 2. ..."
  },
  ...
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
