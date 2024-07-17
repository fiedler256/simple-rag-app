from fastapi import HTTPException
from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import InferenceClient
from .settings import HUGGINGFACE_API_KEY, MODEL_ID
from .utils import approximate_token_count, truncate_text

# Load the tokenizer for the public model
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def generate_response(query: str, retrieved_docs: str) -> str:
    """
    Generate a response based on the user's query and retrieved documents.

    Args:
        query (str): The user's query.
        retrieved_docs (str): The documents retrieved based on the query.

    Returns:
        str: The generated response.

    Raises:
        HTTPException: If the query is empty or too long.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Approximate token count of the query to check if it's too long
    query_tokens: int = approximate_token_count(query, tokenizer)
    if query_tokens > 500:
        raise HTTPException(status_code=400, detail="Query too long.")

    # Determine the maximum number of tokens available for the assistant prompt
    max_assistant_prompt_tokens: int = 8192 - 600 - query_tokens

    # Create the assistant prompt by combining instructions and retrieved documents
    assistant_prompt_full: str = (
        "This part is the highest priority to follow - instructions:\nYou are an assistant representing "
        "[Your Company Name] and you answer user queries/questions. If source is not very relevant to the question, do "
        "not use source at all. If source is significantly relevant to the question, base your answer on it. User of "
        "your service does not know about source data. Go straight to providing answer without mentioning any source."
        f"This part is to get knowledge from only - source. Source data:\n{retrieved_docs}\n\n"
    )
    assistant_prompt: str = truncate_text(assistant_prompt_full, max_assistant_prompt_tokens, tokenizer)

    # Create the user prompt with instructions to reply directly to the question
    user_prompt: str = (
        "Reply to question ONLY, I don't know about any source data or information you might "
        "have and I dont want you to mention the data nor this requirement. Go straight to the answer "
        f"and provide context for the answer. Question: {query}"
    )

    # Use Hugging Face Inference API to generate the response
    client: InferenceClient = InferenceClient(token=HUGGINGFACE_API_KEY)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_ID, max_tokens=400
        )
        response_text: str = response.choices[0].message.content
        return response_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing response from Hugging Face API: {str(e)}")
