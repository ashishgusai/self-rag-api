import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Import the local vector store from database module
from database import vectorstore

load_dotenv()

# Define the State (Memory)
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Initialize the LLM via OpenRouter
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    temperature=0,
    max_tokens=1000
)

# Define the Nodes (Actions)
def retrieve(state: GraphState):
    """Fetches relevant documents from ChromaDB."""
    question = state["question"]
    print(f"---RETRIEVING DOCUMENTS FOR: {question}---")
    
    # Fetch top 3 chunks
    docs = vectorstore.similarity_search(question, k=3)
    return {"documents": [d.page_content for d in docs]}

def generate(state: GraphState):
    """Generates an answer using the LLM and retrieved documents."""
    print("---GENERATING ANSWER---")
    question = state["question"]
    docs = state["documents"]
    
    # Combine all retrieved chunks into a single string
    context = "\n\n".join(docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Use the following retrieved context to answer the user's question. If the context doesn't contain the answer, say 'I don't know based on the provided documents.'\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    # Chain the prompt and the LLM together
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    
    return {"generation": response.content}

# 4. Build the Graph Workflow
workflow = StateGraph(GraphState)

# Add our nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Connect them in a linear path for now
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into an executable application
app_graph = workflow.compile()