import pytest
from app.models import tokenizer as models_tokenizer
from app.retrieval import model as retrieval_model, tokenizer as retrieval_tokenizer
from app.settings import MODEL_ID, HUGGINGFACE_API_KEY
from huggingface_hub import InferenceClient


def test_models_tokenizer_loading():
    """
    Test if the tokenizer in models.py loads correctly.
    """
    try:
        models_tokenizer("This is a test.")
    except Exception as e:
        pytest.fail(f"Models.py tokenizer loading failed: {e}")


def test_retrieval_tokenizer_loading():
    """
    Test if the tokenizer in retrieval.py loads correctly.
    """
    try:
        retrieval_tokenizer("This is a test.")
    except Exception as e:
        pytest.fail(f"Retrieval.py tokenizer loading failed: {e}")


def test_retrieval_model_loading():
    """
    Test if the model in retrieval.py loads correctly.
    """
    try:
        retrieval_model(**retrieval_tokenizer("This is a test.", return_tensors="pt"))
    except Exception as e:
        pytest.fail(f"Retrieval.py model loading failed: {e}")


def test_models_main_model_loading():
    """
    Test if the model specified by MODEL_ID in settings.py can be accessed.
    """
    try:
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
        client.chat_completion(
            messages=[{"role": "user", "content": "Test message"}],
            model=MODEL_ID,
            max_tokens=10
        )
    except Exception as e:
        pytest.fail(f"Main model (models.py) access failed: {e}")
