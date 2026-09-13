# Frontend Application

This is the frontend part of the RAG application, built using Streamlit. It provides a user interface for interacting with the backend service.

## Setup and Usage

### Environment Variables

The frontend uses the following environment variables, which are loaded from the `.env.frontend` file:

- `API_BASE_URL`:  The base URL for the backend API (e.g., `http://backend:8000`).
- `API_KEY`: The API key for accessing the backend API.

### How to Run

1. Ensure you have the application dependencies installed by using `uv`:
    ```bash
    uv pip install -r requirements.in
    ```
2.  Run the streamlit app:
    ```bash
    streamlit run app.py
    ```

## Features
* Document upload
* Chat interface
* Real-time chat with Websockets
* Health check display

## Notes
The `.env.frontend` file should be located in the root directory of the project (`rag-app/`)