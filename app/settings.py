import os

# Hugging Face API key and model ID
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Directory containing the data files
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
