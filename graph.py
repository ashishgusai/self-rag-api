import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Import the local vector store from database module
from database import vectorstore
from pydantic import BaseModel, Field

load_dotenv()

# Define the State (Memory)
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retries: int

# Initialize the LLM via OpenRouter
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash",
    temperature=0,
    max_tokens=1000
)

#  Define Grader Schemas (Pydantic)
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# Define the Nodes (Actions)
def retrieve(state: GraphState):
    """Fetches relevant documents from ChromaDB."""
    question = state["question"]
    print(f"---RETRIEVING DOCUMENTS FOR: {question}---")
    
    # Fetch top 3 chunks
    docs = vectorstore.similarity_search(question, k=3)
    return {"documents": [d.page_content for d in docs], "retries": state.get("retries", 0)}

def grade_documents(state: GraphState):
    """Filters retrieved documents based on relevance."""
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    docs = state["documents"]
    
    # Bind the Pydantic schema to the LLM to force JSON output
    structured_grader = llm.with_structured_output(GradeDocuments)
    system = "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords or semantic meaning related to the question, grade it as 'yes'."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", "Document: \n\n {document} \n\n Question: {question}")
    ])
    retrieval_grader = prompt | structured_grader

    filtered_docs = []
    for d in docs:
        score = retrieval_grader.invoke({"question": question, "document": d})
        if score and score.binary_score.lower() == "yes":
            filtered_docs.append(d)
            
    return {"documents": filtered_docs}

def rewrite_question(state: GraphState):
    """Rewrites the question if documents were irrelevant."""
    print("---REWRITING QUESTION---")
    question = state["question"]
    system = "You are an AI optimizing a user query for vector database retrieval. Look at the input and deduce the underlying intent."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", "Initial question: {question} \n Formulate an improved question.")
    ])
    question_rewriter = prompt | llm
    better_question = question_rewriter.invoke({"question": question}).content
    return {"question": better_question}

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

def decide_to_generate(state: GraphState):
    """Routes to generate if documents are relevant, or rewrite if not."""
    filtered_docs = state["documents"]
    if not filtered_docs:
        print("---DECISION: ALL DOCS IRRELEVANT, REWRITING QUESTION---")
        return "rewrite_question"
    print("---DECISION: DOCS RELEVANT, GENERATING---")
    return "generate"

def check_hallucinations(state: GraphState):
    """Checks if the final generation is grounded in facts."""
    print("---CHECKING FOR HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    retries = state["retries"]

    if retries >= 2:
        print("---MAX RETRIES REACHED---")
        return "end"

    structured_grader = llm.with_structured_output(GradeHallucinations)
    system = "You are a grader assessing whether an LLM generation is grounded in / supported by a set of facts. Grade 'yes' if supported, 'no' if hallucinated."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        ("human", "Facts: \n\n {documents} \n\n Generation: {generation}")
    ])
    hallucination_grader = prompt | structured_grader
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    
    if score and score.binary_score.lower() == "yes":
        print("---DECISION: NO HALLUCINATIONS, SENDING TO USER---")
        return "end"
    else:
        print("---DECISION: HALLUCINATION DETECTED, RE-GENERATING---")
        state["retries"] += 1
        return "generate"

# Build the Graph Workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_question", rewrite_question)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Branching logic after grading
workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_generate, 
    {"rewrite_question": "rewrite_question", "generate": "generate"}
)

workflow.add_edge("rewrite_question", "retrieve")

# Loop logic after generation
workflow.add_conditional_edges(
    "generate", 
    check_hallucinations, 
    {"end": END, "generate": "generate"}
)

app_graph = workflow.compile()