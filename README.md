# Simple RAG Application

## Overview
This is a simple Retrieval-Augmented Generation (RAG) application that integrates a document retrieval system with a large language model (LLM) to generate responses based on retrieved documents.

## Running With Docker

0. **Prerequisites**
    ###### Docker


1. **Clone the repository**
    ```bash
    git clone https://github.com/fiedler256/simple-rag-app.git
    cd simple-rag-app
    ```
   
2. **(Optional) Create a .env file in the project root with your Hugging Face API key:**
    #### On Windows
    ```bash
    echo "#" | Out-File -Encoding utf8 .env; echo "HUGGINGFACE_API_KEY=your_hugging_face_api_key" | Out-File -Encoding utf8 -Append .env
    ```
    #### On Linux
    ```bash
    echo -e "#\nHUGGINGFACE_API_KEY=your_hugging_face_api_key" > .env
    ```

3. **Build the Docker image:**
    ```bash
    docker build -t simple-rag-app .
    ```

4. **Run the Docker container**
    ```bash
    docker run -p 8000:8000 simple-rag-app
    ```

## Running Without Docker

0. **Prerequisites**
    ###### Python 3.10


1. **Clone the repository**
    ```bash
    git clone https://github.com/fiedler256/simple-rag-app.git
    cd simple-rag-app
    ```
   
2. **(Optional) Create a .env file in the project root with your Hugging Face API key:**
    #### On Windows
    ```bash
    echo "#" | Out-File -Encoding utf8 .env; echo "HUGGINGFACE_API_KEY=your_hugging_face_api_key" | Out-File -Encoding utf8 -Append .env
    ```
    #### On Linux
    ```bash
    echo -e "#\nHUGGINGFACE_API_KEY=your_hugging_face_api_key" > .env
    ```

3. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Running the Application**

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

### Running Tests
1. **Pytest**
    ```bash
    pytest -v
    ```

## Usage

### API Endpoints

- `POST /query`
  - Request: `{ "query": "your query here" }`
  - Response: `{ "response": "generated response based on retrieved documents" }`

### Example

To ask a question to the RAG app, send a POST request to `http://localhost:8000/query` with the following structure of a JSON body:
```json
{
  "query": "Are you selling in Germany?"
}
