# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import vectorstore
from graph import app_graph

# Load environment variables early
load_dotenv()

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

    inputs = {"question": request.question}
    events = []
    final_answer = ""

    try:
        # Stream the execution of the graph
        for output in app_graph.stream(inputs):
            # LangGraph yields a dictionary with the node name as the key
            for node_name, state_update in output.items():
                events.append(f"Node '{node_name}' executed.")
                
                if "generation" in state_update:
                    final_answer = state_update["generation"]
        
        return QueryResponse(
            answer=final_answer,
            execution_trace=events
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))