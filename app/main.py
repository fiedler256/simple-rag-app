from fastapi import FastAPI
from pydantic import BaseModel
from .retrieval import retrieve_documents, build_faiss_index
from .models import generate_response
import os

app = FastAPI()

# Load documents from the data folder
data_dir = os.path.join(os.path.dirname(__file__), '../data')
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith('.md'):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())

# Build the FAISS index
index = build_faiss_index(documents)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    retrieved_docs = retrieve_documents(request.query, documents, index)
    response = generate_response(request.query, retrieved_docs)

    # Ensure response is not None
    if not response:
        response = "No response generated."

    return QueryResponse(response=response)
