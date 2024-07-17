import os, sys
from transformers import PreTrainedTokenizer

def load_documents(data_dir):
    """
    Load all Markdown documents from the specified directory.
    """
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.md'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def approximate_token_count(text, tokenizer: PreTrainedTokenizer):
    """
    Approximate the number of tokens in a text using the specified tokenizer.
    """
    tokenizer.model_max_length = sys.maxsize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def truncate_text(text, max_tokens, tokenizer: PreTrainedTokenizer):
    """
    Truncate the text to the specified number of tokens using the tokenizer.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text
