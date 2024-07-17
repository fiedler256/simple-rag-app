import os
from app.utils import load_documents, approximate_token_count, truncate_text
from transformers import AutoTokenizer


def test_load_documents():
    """
    Test load_documents with the data directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    documents = load_documents(data_dir)
    assert len(documents) > 0


def test_approximate_token_count():
    """
    Test approximate_token_count with a sample text.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    text = "This is a test text."
    token_count = approximate_token_count(text, tokenizer)
    assert token_count > 0


def test_truncate_text():
    """
    Test truncate_text with a sample text and token limit.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    text = "This is a test text."
    truncated_text = truncate_text(text, 3, tokenizer)
    assert len(truncated_text.split()) <= 3
