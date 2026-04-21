# Hybrid RAG Project

## Overview
The Hybrid RAG (Retrieval-Augmented Generation) project is designed to provide a robust framework for document ingestion, retrieval, and generation using FastAPI. This project integrates various techniques, including semantic chunking, vector storage, BM25 indexing, and advanced generation methods.

## Project Structure
```
hybrid-rag/
├── app/                    # Main application code
│   ├── __init__.py
│   ├── main.py             # FastAPI entry point
│   ├── config.py           # Configuration settings
│   ├── ingestion/           # Document ingestion components
│   │   ├── __init__.py
│   │   ├── chunker.py      # Semantic chunking
│   │   ├── embedder.py     # OpenAI embeddings
│   │   └── loader.py       # Document loaders (PDF, txt, web)
│   ├── retrieval/          # Document retrieval components
│   │   ├── __init__.py
│   │   ├── vector_store.py  # Qdrant operations
│   │   ├── bm25_store.py    # BM25 index
│   │   ├── hybrid.py       # Fusion of both retrieval methods
│   │   └── reranker.py     # Cohere re-ranking
│   ├── generation/         # Document generation components
│   │   ├── __init__.py
│   │   └── chain.py        # RAG chain with Groq
│   └── api/                # API definitions
│       ├── __init__.py
│       ├── routes.py       # API routes
│       └── schemas.py      # Pydantic models
├── tests/                  # Unit tests
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_chain.py
├── data/                   # Sample documents for testing
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image instructions
├── docker-compose.yml      # Docker services configuration
└── README.md               # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd hybrid-rag
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the FastAPI application, execute the following command:
```
uvicorn app.main:app --reload
```
This will start the server at `http://127.0.0.1:8000`.

## API Endpoints
- **POST /ingest**: Ingest documents into the system.
- **POST /query**: Query the system for information.
- **POST /evaluate**: Evaluate the performance of the retrieval and generation.

## Testing
To run the tests, use:
```
pytest tests/
```

## Docker
To build and run the application using Docker, execute:
```
docker-compose up --build
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.