from dotenv import load_dotenv
import os
import logging

# Try to load environment variable from .env file
try:
    load_dotenv()
except Exception as e:
    raise RuntimeError(
        "Failed to load .env file. Please ensure the .env file is saved with UTF-8 encoding."
    ) from e

# Hugging Face API key and model ID
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"

# Directory containing the data files
DATA_DIR: str = os.path.join(os.path.dirname(__file__), '..', 'data')

# Set up logging
logger = logging.getLogger("uvicorn")

if HUGGINGFACE_API_KEY:
    logger.info(f"Hugging Face API key is in use.")
else:
    logger.warning("Hugging Face API key is NOT in use. Functionality may be limited.")
