# RAGLab
# RAG Application

This is a production-ready RAG application built using FastAPI, Streamlit, Milvus, and OpenAI.

## Architecture

- **Frontend:** Streamlit for the UI with chatbot interface and file upload.
- **Backend:** FastAPI for the API with async endpoints.
- **Vector Database:** Milvus for storing embeddings.
- **Embedding Model:** Sentence-transformers/all-mpnet-base-v2 for generating embeddings.
- **LLM:** OpenAI GPT for generating responses.
- **Monitoring:** Attu for visualizing Milvus.

## Project Structure

```
rag-lab/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py
│   │   │   └── dependencies.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   └── background.py
│   │   └── repository/
│   │       ├── __init__.py
│   │       └── milvus.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   └── test_services.py
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   └── sidebar.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── api.py
│   └── Dockerfile
├── config/
│   ├── logging.yaml
│   └── milvus.yaml
├── docker/
│   └── docker-compose.yml
├── docs/
│   ├── setup.md
│   └── api.md
├── logs/
└── README.md
```

## Setup and Deployment

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag-app
    ```

2.  **Install `uv` globally:**
    ```bash
     pip install uv
    ```

3.  **Create a virtual environment:**
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

    ```PowerShell
    .venv/bin/activate
    ```

4. **Install the application requirements:**
    ```bash
    uv pip install -r pyproject.toml
    ```
5.  **Set up environment variables:**
    - Create a `.env` file based on the `.env.example` and add your API keys.

6.  **Run Docker Compose:**
    ```bash
    docker-compose up --build
    ```

7.  **Access the applications:**
    -   Frontend: http://localhost:8501
    -   Attu: http://localhost:3000
    -   Backend: http://localhost:8000/docs (for API Documentation access)

## Features
* Document upload
* Chat interface
* Vector search 
* Real-time chat with Websockets
* Healthcheck endpoint
* API Key security
* Docker based configuration
* Error handling

## API Documentation
You can access the OpenAPI documentation at http://localhost:8000/docs