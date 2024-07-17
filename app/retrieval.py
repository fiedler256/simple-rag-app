import faiss
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import List, Any
import warnings

# Suppress specific warnings related to Hugging Face Hub file downloads
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load tokenizer and model for embedding generation
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model: PreTrainedModel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(documents: List[str]) -> Any:
    """
    Generate embeddings for a list of documents.

    Args:
        documents (List[str]): List of documents to generate embeddings for.

    Returns:
        numpy.ndarray: Array of embeddings for the documents.

    Raises:
        ValueError: If the document list is empty.
    """
    if not documents:
        raise ValueError("Document list is empty.")
    # Tokenize the documents and get model outputs
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Compute mean of token embeddings to get document embeddings
    embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


def retrieve_documents(query: str, documents: List[str], index: faiss.IndexFlatL2) -> str:
    """
    Retrieve the most relevant documents for the given query using FAISS index.

    Args:
        query (str): The user's query.
        documents (List[str]): The list of documents to search.
        index (faiss.IndexFlatL2): The FAISS index to search in.

    Returns:
        str: The concatenated string of the retrieved documents.
    """
    # Get the embedding for the query
    query_embedding = get_embeddings([query])
    # Search the index for the top 5 most similar documents
    _, indices = index.search(query_embedding, k=5)
    # Extract and return the retrieved documents as a single string
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return "\n".join(retrieved_docs)


def build_faiss_index(documents: List[str]) -> faiss.IndexFlatL2:
    """
    Build a FAISS index for the given list of documents.

    Args:
        documents (List[str]): The list of documents to index.

    Returns:
        faiss.IndexFlatL2: The built FAISS index.
    """
    # Generate embeddings for the documents
    embeddings = get_embeddings(documents)
    dimension = embeddings.shape[1]
    # Create a FAISS index and add the document embeddings to it
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
