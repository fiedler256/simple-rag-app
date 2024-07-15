# Simple RAG Application

## Overview
This is a simple Retrieval-Augmented Generation (RAG) application that integrates a document retrieval system with a large language model (LLM) to generate responses based on retrieved documents.

## Setup and Usage

### Prerequisites
- Docker

### Running the Application

1. **Build the Docker image:**
    ```bash
    docker build -t simple-rag-app .
    ```

2. **Run the Docker container with Llama model:**
    ```bash
    docker run -p 8000:8000 -e HUGGINGFACE_API_KEY=<your-huggingface-api-key> simple-rag-app
    ```
   - Providing Hugging Face API key is optional for using the Hugging Face API. If not set, the Inference API might be used with lower rate limits or not available at all.

### API Endpoints

- `POST /query`
  - Request: `{ "query": "your query here" }`
  - Response: `{ "response": "generated response based on retrieved documents" }`

### Example

To ask a question to the RAG app, send a POST request to `http://localhost:8000/query` with the following JSON body:
```json
{
  "query": "Are you selling in Germany?"
}
