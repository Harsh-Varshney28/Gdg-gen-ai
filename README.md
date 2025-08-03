# RAG Question-Answering Assistant

A Retrieval-Augmented Generation (RAG) system built for the GDG AI/ML Domain recruitment task. This system allows users to upload PDF documents, processes them into a vector database, and provides intelligent question-answering capabilities using open-source models.

## Features

- **PDF Document Ingestion**: Upload and process multiple PDF files
- **Text Chunking**: Intelligent text segmentation with overlap for better context retention  
- **Vector Database**: ChromaDB for efficient similarity search
- **Open-Source Embeddings**: Uses sentence-transformers for document embeddings
- **LLM Integration**: Supports Groq API and HuggingFace models
- **RESTful API**: FastAPI backend for easy integration
- **Web Interface**: Streamlit frontend for user-friendly interaction
- **Deployment Ready**: Configured for cloud deployment

## Architecture

```
Documents → PDF Processing → Text Chunking → Vector Embeddings → ChromaDB
                                                                      ↓
User Query → Embedding → Similarity Search → Context Retrieval → LLM → Response
```

## Project Structure

```
rag-qa-assistant/
├── src/
│   ├── config.py              # Configuration settings
│   ├── document_processor.py   # PDF processing and chunking
│   ├── vector_store.py        # ChromaDB vector database operations
│   ├── llm_client.py          # LLM integration (Groq/HuggingFace)
│   ├── rag_pipeline.py        # Main RAG pipeline
│   ├── api.py                 # FastAPI backend
│   └── app.py                 # Streamlit frontend
├── documents/                 # PDF storage directory
├── tests/                     # Unit tests
├── data/                      # Data files and exports
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-qa-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### 5. Add PDF Documents

Place your PDF files in the `documents/` directory.

### 6. Run the Application

#### Option A: Web Interface (Streamlit)
```bash
streamlit run src/app.py
```

#### Option B: API Server (FastAPI)
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /upload` - Upload PDF documents
- `POST /process` - Process uploaded documents
- `POST /query` - Ask questions about the documents
- `GET /health` - Health check endpoint

## Usage Example

1. **Upload Documents**: Place PDF files in the `documents/` folder
2. **Process Documents**: The system will automatically chunk and embed the documents
3. **Ask Questions**: Query the system about the document contents
4. **Get Answers**: Receive contextually relevant answers based on the document content

## Technologies Used

- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Interactive web application framework
- **ChromaDB**: Vector database for similarity search
- **Sentence Transformers**: Open-source embedding models
- **Groq API**: Fast LLM inference
- **PyMuPDF/PyPDF2**: PDF text extraction
- **LangChain**: LLM application framework

## Deployment

The application is configured for deployment on cloud platforms:

- **AWS**: Use EC2 or ECS for containerized deployment
- **Google Cloud**: Deploy on Cloud Run or Compute Engine
- **Heroku**: Direct deployment with Procfile
- **Docker**: Containerized deployment ready

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is created for the GDG AI/ML Domain recruitment task.

## Contact

For questions or clarifications about this recruitment task, please reach out to the GDG AI/ML Domain team.
