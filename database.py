import os
from dotenv import load_dotenv
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

# Initialize Vectorstore
vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
