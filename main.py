"""
Multi-Agent PDF Q&A System using LangChain and LangGraph with Optional Web Search

Dependencies:
- langchain_community
- langchain_google_genai
- langgraph
- python-dotenv
- duckduckgo-search (optional, for web search)

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence

from langgraph.graph import StateGraph, END

# Try to import web search - gracefully handle if not available
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

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

# --- LCEL Chains ---

# Analyst LCEL Chain: Extract insights, then summarize them
def analyst_node(state):
    query = state["query"]
    retrieved_texts = state["retrieved_texts"]
    llm = state["llm"]
    web_search_enabled = state.get("web_search_enabled", False)

    # First LLM call: Extract insights
    extract_prompt = ChatPromptTemplate.from_template(
        "Given the following PDF chunks and user query, extract key insights.\n\n"
        "User Query: {query}\n\n"
        "PDF Chunks:\n{retrieved_texts}\n\n"
        "Key Insights:"
    )
    extract_chain = extract_prompt | llm | StrOutputParser()

    # Second LLM call: Summarize insights
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following insights in 2-3 sentences for clarity.\n\n"
        "Insights:\n{insights}\n\n"
        "Summary:"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    # First, extract insights
    insights = extract_chain.invoke({
        "query": query,
        "retrieved_texts": "\n".join(retrieved_texts)
    })

    # Then, summarize the insights
    summary = summarize_chain.invoke({
        "insights": insights
    })

    state["insights"] = insights
    state["summary"] = summary

    # Optionally, determine if web search is needed (using summary)
    if web_search_enabled and WEB_SEARCH_AVAILABLE:
        needs_web_search = "insufficient" in summary.lower() or "additional" in summary.lower() or "more information" in summary.lower()
        state["needs_web_search"] = needs_web_search
    else:
        state["needs_web_search"] = False

    return state

# Writer LCEL Chain: Compose answer using both insights and summary, and optionally web results
def writer_node(state):
    query = state["query"]
    insights = state["insights"]
    summary = state.get("summary", "")
    web_results = state.get("web_results", "")
    llm = state["llm"]

    if web_results and state.get("needs_web_search", False):
        writer_prompt = ChatPromptTemplate.from_template(
            "Create a comprehensive answer using PDF insights, summary, and web search results.\n\n"
            "User Query: {query}\n\n"
            "PDF Insights:\n{insights}\n\n"
            "Summary:\n{summary}\n\n"
            "Web Search Results:\n{web_results}\n\n"
            "Instructions:\n"
            "1. Start with PDF information\n"
            "2. Supplement with web search\n"
            "3. Indicate sources\n"
            "4. Provide a well-structured answer\n\n"
            "Final Answer:"
        )
        writer_chain = writer_prompt | llm | StrOutputParser()
        final_answer = writer_chain.invoke({
            "query": query,
            "insights": insights,
            "summary": summary,
            "web_results": web_results
        })
    else:
        writer_prompt = ChatPromptTemplate.from_template(
            "Using the PDF insights and summary below, craft a concise and professional answer to the user's query.\n\n"
            "User Query: {query}\n\n"
            "PDF Insights:\n{insights}\n\n"
            "Summary:\n{summary}\n\n"
            "Final Answer:"
        )
        writer_chain = writer_prompt | llm | StrOutputParser()
        final_answer = writer_chain.invoke({
            "query": query,
            "insights": insights,
            "summary": summary
        })

    state["final_answer"] = final_answer
    return state

def web_researcher_node(state):
    """
    Performs web search if additional information is needed and web search is enabled.
    """
    if not state.get("needs_web_search", False) or not WEB_SEARCH_AVAILABLE:
        state["web_results"] = ""
        return state

    query = state["query"]
    search = DuckDuckGoSearchRun()
    search_query = f"{query} famous works themes author literature"

    try:
        print(f"🔍 Searching web for additional information: {search_query}")
        web_results = search.run(search_query)
        state["web_results"] = web_results
        print("✅ Web search completed.")
    except Exception as e:
        state["web_results"] = f"Web search failed: {str(e)}"
        print(f"❌ Web search failed: {str(e)}")

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
        max_output_tokens=2048,
    )

    # Ask user about web search preference
    if WEB_SEARCH_AVAILABLE:
        web_search_choice = input("\n🌐 Enable web search for information not available in PDF? (y/n): ").strip().lower()
        web_search_enabled = web_search_choice in ['y', 'yes', '1', 'true']
        
        if web_search_enabled:
            print("✅ Web search enabled - will search online for additional information when needed.")
        else:
            print("📖 Web search disabled - will only use PDF content.")
    else:
        web_search_enabled = False
        print("⚠️  Web search not available (install 'duckduckgo-search' package to enable).")

    # Define LangGraph workflow with state_schema
    workflow = StateGraph(dict)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("web_researcher", web_researcher_node)
    workflow.add_node("writer", writer_node)

    # Define workflow edges
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "web_researcher")
    workflow.add_edge("web_researcher", "writer")
    workflow.add_edge("writer", END)

    workflow.set_entry_point("researcher")
    graph = workflow.compile()

    system_name = "Enhanced PDF Q&A System" if web_search_enabled else "PDF Q&A System"
    print(f"\n{system_name} Ready (LangChain + LangGraph).")
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
            "web_search_enabled": web_search_enabled,
        }

        # Stream workflow progression
        print("\n[Researcher] Retrieving relevant text chunks from PDF...")
        state = researcher_node(state)
        print(f"Retrieved {len(state['retrieved_texts'])} text chunks.")

        print("\n[Analyst] Analyzing content and extracting key insights...")
        state = analyst_node(state)
        
        if web_search_enabled and state.get("needs_web_search", False):
            print("📝 PDF content incomplete - additional web search recommended.")
        elif web_search_enabled:
            print("✅ PDF content appears sufficient for the query.")

        if web_search_enabled:
            print("\n[Web Researcher] Checking for additional information...")
            state = web_researcher_node(state)

        print("\n[Writer] Crafting final answer...")
        state = writer_node(state)
        print(f"\nKey Insights:\n{state['insights']}\n")
        print(f"Final Answer:\n{state['final_answer']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()