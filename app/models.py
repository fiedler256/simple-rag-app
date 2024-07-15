from .settings import HUGGINGFACE_API_KEY, MODEL_ID
from huggingface_hub import InferenceClient


def generate_response(query, retrieved_docs):
    max_input_tokens = 8192 - 1000
    query_tokens = len(query.split())

    if query_tokens > 3000:
        return "Error: Query exceeds the maximum limit of 3000 tokens."

    documents_text = "\n".join(retrieved_docs)

    # Construct the prompt using Meta Llama's format with |> notation
    prompt_initial = (
        "<|system|>\n"
        "This part is the highest priority to follow - instructions:\nYou are an assistant employed by "
        "[Your Company Name] that answers user queries. If source is not very relevant to the question, do not use "
        "source at all. If source is significantly relevant to the question, base your answer on it. User of your "
        "service does not know about source data. Go straight to providing answer without mentioning any source."
        f"Here is the question:\n{query}\n\n"
        f"This part is the lowest priority to follow - source data:\n{documents_text}\n"
    )

    prompt_truncated = truncate_text(prompt_initial, max_input_tokens)

    prompt = f"{prompt_truncated}<|endoftext|><|assistant|>"

    client = InferenceClient(token=HUGGINGFACE_API_KEY)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": prompt}
            ],
            model=MODEL_ID, max_tokens=800
        )
        response_text = response["choices"][0]["message"]["content"]
        return response_text.strip()
    except Exception as e:
        return f"Error processing response from Hugging Face API: {str(e)}"


def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text
