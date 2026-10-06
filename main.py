# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables early
load_dotenv()

# Initialize Embeddings with OpenRouter
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small" # OpenRouter routing format
)

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

app = FastAPI(
    title="Self-Correcting RAG API",
    description="An agentic API that evaluates its own retrieved context and mitigates hallucinations.",
    version="0.2.0"
)


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    execution_trace: list[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vector_store_docs": vectorstore._collection.count()}

@app.post("/api/v1/documents")
async def upload_document(file: UploadFile = File(...)):
    """Reads a text file, chunks it, and saves embeddings to ChromaDB."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported for now.")
    
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        # Use a text splitter for intelligent chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": file.filename}) for chunk in chunks]
        
        # Generate deterministic IDs for chunk upserting (prevents duplicate entries)
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(documents))]
        
        # Add or update in ChromaDB
        vectorstore.add_documents(documents, ids=ids)
        
        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_created": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    dummy_trace = ["Node 'retrieve' executed", "Node 'generate' executed"]
    return QueryResponse(
        answer=f"Echoing back your question: {request.question}",
        execution_trace=dummy_trace
    )