from fastapi import FastAPI
from pydantic import BaseModel
from .retrieval import retrieve_documents, build_faiss_index
from .models import generate_response
from .settings import DATA_DIR
from .utils import load_documents
from typing import List

app = FastAPI()

# Load documents from the data directory and build the FAISS index
documents: List[str] = load_documents(DATA_DIR)
if not documents:
    raise RuntimeError("No documents found in the data directory.")
index = build_faiss_index(documents)


# Define the request and response models for the query endpoint
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Endpoint to handle user queries and return responses based on document retrieval.
    """
    # Retrieve relevant documents based on the query
    retrieved_docs: str = retrieve_documents(request.query, documents, index)
    # Generate a response using the retrieved documents
    response: str = generate_response(request.query, retrieved_docs)

    return QueryResponse(response=response)
