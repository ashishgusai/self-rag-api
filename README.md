# Self-Correcting RAG API 🤖📚

A **Self-Correcting Retrieval-Augmented Generation (Self-RAG)** API built with **FastAPI**, **LangGraph**, **ChromaDB**, and **OpenRouter** (OpenAI Embeddings & Gemini 2.5 Flash).

This API evaluates its own retrieved context, rewrites questions if retrieved context is irrelevant, and checks for AI hallucinations before returning an answer.

---

## 🏗️ Project Architecture

```
.
├── database.py       # Initializes OpenRouter embeddings and ChromaDB vector store
├── graph.py          # Defines the LangGraph agent state machine (Retrieve -> Grade -> Rewrite -> Generate -> Check Hallucinations)
├── main.py           # FastAPI application with endpoints for document upload, chat, and health check
├── requirements.txt  # Python package dependencies
└── sample.txt        # Sample document for testing ingestion
```

---

## 🚀 Getting Started

Follow these step-by-step instructions to get the API running on your local machine.

### 📋 Prerequisites

- **Python 3.10** or higher
- An **OpenRouter API Key** (or an OpenAI-compatible API Key)

---

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd self-rag-api
```

### Step 2: Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the root directory of the project:

```bash
cp .env.example .env   # Or create a new .env file manually
```

Add your OpenRouter API key inside `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🏃 Running the Application

Start the FastAPI server using `uvicorn`:

```bash
uvicorn main:app --reload
```

The server will start running at: **`http://127.0.0.1:8000`**

- **Interactive API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Documentation (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📖 How to Use the API

### 1️⃣ Check API Health & Document Count

**Endpoint:** `GET /health`

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store_docs": 3
}
```

---

### 2️⃣ Ingest / Upload a Document (`.txt`)

Upload a `.txt` file to chunk it and store its vector embeddings in ChromaDB.

**Endpoint:** `POST /api/v1/documents`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.txt"
```

**Response:**
```json
{
  "message": "Successfully ingested sample.txt",
  "chunks_created": 3
}
```

---

### 3️⃣ Ask a Question (Self-RAG Chat)

Ask a question against your uploaded documents. The Self-RAG agent will retrieve context, grade relevance, generate an answer, and check for hallucinations.

**Endpoint:** `POST /api/v1/chat`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is covered in the document?"
  }'
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "execution_trace": [
    "Node 'retrieve' executed.",
    "Node 'grade_documents' executed.",
    "Node 'generate' executed."
  ]
}
```

---

## 🛠️ How Self-RAG Works Under the Hood

1. **Retrieval**: Fetches top matching document chunks from ChromaDB.
2. **Relevance Grading**: Evaluates if retrieved chunks actually pertain to the question.
3. **Query Rewriting**: If document chunks are irrelevant, the agent automatically rewrites the user query and searches ChromaDB again.
4. **Generation**: Generates an answer using Gemini 2.5 Flash conditioned on the retrieved context.
5. **Hallucination Check**: Evaluates if the answer is grounded in facts. If hallucinations are detected, it regenerates the answer (up to 2 retries).
