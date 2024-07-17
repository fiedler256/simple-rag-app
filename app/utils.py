import os
import sys
from transformers import PreTrainedTokenizer
from typing import List


def load_documents(data_dir: str) -> List[str]:
    """
    Load all Markdown documents from the specified directory.

    Args:
        data_dir (str): Path to the directory containing Markdown files.

    Returns:
        List[str]: List of document contents.
    """
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.md'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


def approximate_token_count(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Approximate the number of tokens in a text using the specified tokenizer.

    Args:
        text (str): The text to count tokens in.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for counting tokens.

    Returns:
        int: The approximate number of tokens.
    """
    # Temporarily increase the model max length to avoid truncation
    tokenizer.model_max_length = sys.maxsize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def truncate_text(text: str, max_tokens: int, tokenizer: PreTrainedTokenizer) -> str:
    """
    Truncate the text to the specified number of tokens using the tokenizer.

    Args:
        text (str): The text to truncate.
        max_tokens (int): The maximum number of tokens to keep.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for truncating.

    Returns:
        str: The truncated text.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text
