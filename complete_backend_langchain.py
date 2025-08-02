"""
Multi-Agent PDF Q&A System using LangChain and LangGraph

Dependencies:
- langchain_community
- langchain_google_genai
- langgraph
- python-dotenv

Make sure to set your Google Gemini API key in a .env file:
GOOGLE_API_KEY=your_api_key_here
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END

# ---------------------------
# PDF Processing Functions
# ---------------------------

def process_pdf(pdf_path):
    """
    Loads a PDF, splits it into chunks, and creates a FAISS vector store with Gemini embeddings.
    Returns the vector store and the list of document chunks.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, chunks

# ---------------------------
# LangGraph Agent Nodes
# ---------------------------

def researcher_node(state):
    """
    Retrieves relevant text chunks from the vector store based on the user's query.
    """
    query = state["query"]
    vectorstore = state["vectorstore"]
    docs = vectorstore.similarity_search(query, k=5)
    retrieved_texts = [doc.page_content for doc in docs]
    state["retrieved_texts"] = retrieved_texts
    return state

def analyst_node(state):
    """
    Analyzes retrieved text and extracts key insights using Gemini LLM.
    """
    query = state["query"]
    retrieved_texts = state["retrieved_texts"]
    llm = state["llm"]

    prompt = (
        "You are an analyst. Given the following retrieved text chunks from a PDF and the user's query, "
        "extract key insights that directly answer the query. "
        "Be concise and focus on the most relevant information.\n\n"
        f"User Query: {query}\n\n"
        f"Retrieved Texts:\n{chr(10).join(retrieved_texts)}\n\n"
        "Key Insights:"
    )
    insights = llm.invoke(prompt).content
    state["insights"] = insights
    return state

def writer_node(state):
    """
    Crafts a concise, professional final answer using Gemini LLM.
    """
    query = state["query"]
    insights = state["insights"]
    llm = state["llm"]

    prompt = (
        "You are a professional writer. Using the key insights below, craft a concise and professional answer "
        "to the user's query. Ensure clarity and completeness.\n\n"
        f"User Query: {query}\n\n"
        f"Key Insights:\n{insights}\n\n"
        "Final Answer:"
    )
    final_answer = llm.invoke(prompt).content
    state["final_answer"] = final_answer
    return state

# ---------------------------
# Main CLI Orchestration
# ---------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        sys.exit(1)

    # Use fixed PDF path
    pdf_path = r"C:\Users\Endead\Downloads\Documents\lekl101-2-9-1.pdf"
    if not os.path.isfile(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    print("Processing PDF and building vector store...")
    vectorstore, _ = process_pdf(pdf_path)

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=1024,
    )

    # Define LangGraph workflow with state_schema
    workflow = StateGraph(dict)  # Use dict as the state schema
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)

    workflow.set_entry_point("researcher")
    graph = workflow.compile()

    print("\nPDF Q&A System Ready.")
    print("Type your question below (or type 'exit' to quit):\n")

    while True:
        query = input(">> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        # Initial state for workflow
        state = {
            "query": query,
            "vectorstore": vectorstore,
            "llm": llm,
        }

        # Stream workflow progression
        print("\n[Researcher] Retrieving relevant text chunks...")
        state = researcher_node(state)
        print(f"Retrieved {len(state['retrieved_texts'])} text chunks.")

        print("\n[Analyst] Extracting key insights...")
        state = analyst_node(state)
        print(f"Key Insights:\n{state['insights']}\n")

        print("[Writer] Crafting final answer...")
        state = writer_node(state)
        print(f"\nFinal Answer:\n{state['final_answer']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()