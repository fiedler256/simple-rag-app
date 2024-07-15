import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(documents):
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


def retrieve_documents(query, documents, index):
    query_embedding = get_embeddings([query])
    _, indices = index.search(query_embedding, k=5)  # Retrieve top 5 documents
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return "\n".join(retrieved_docs)  # Concatenate retrieved documents with newline separator


def build_faiss_index(documents):
    embeddings = get_embeddings(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
