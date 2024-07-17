import os

# Hugging Face API key and model ID
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"

# Directory containing the data files
DATA_DIR: str = os.path.join(os.path.dirname(__file__), '../data')
