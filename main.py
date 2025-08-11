"""
Multi-Agent PDF Q&A System using LangChain and LangGraph with Optional Web Search and Conversation History
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END

# Try to import web search - gracefully handle if not available
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False

# ---------------------------
# Conversation History Class
# ---------------------------

class ConversationHistory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
    
    def add_interaction(self, query, answer):
        """Add a Q&A pair to history"""
        self.history.append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Keep only the last max_history interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_formatted_history(self):
        """Get formatted conversation history for context"""
        if not self.history:
            return ""
        
        formatted = "Previous Conversation History:\n"
        for i, interaction in enumerate(self.history, 1):
            formatted += f"\nQ{i}: {interaction['query']}\n"
            formatted += f"A{i}: {interaction['answer']}\n"
        formatted += "\n---End of Previous Conversation---\n"
        return formatted
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_recent_context(self, num_recent=3):
        """Get only the most recent interactions for context"""
        if not self.history:
            return ""
        
        recent = self.history[-num_recent:] if len(self.history) >= num_recent else self.history
        formatted = "Recent Conversation Context:\n"
        for interaction in recent:
            formatted += f"Q: {interaction['query']}\nA: {interaction['answer']}\n\n"
        return formatted

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
    Now also considers conversation history for better context retrieval.
    """
    query = state["query"]
    vectorstore = state["vectorstore"]
    conversation_history = state.get("conversation_history", "")
    
    # Combine current query with recent context for better retrieval
    enhanced_query = query
    if conversation_history:
        enhanced_query = f"{conversation_history}\n\nCurrent Question: {query}"
    
    docs = vectorstore.similarity_search(enhanced_query, k=5)
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
    conversation_history = state.get("conversation_history", "")

    # Simple, working approach
    extract_prompt = ChatPromptTemplate.from_template(
        "Given the following PDF chunks, conversation history, and user query, extract key insights.\n\n"
        "{conversation_history}"
        "Current User Query: {query}\n\n"
        "PDF Chunks:\n{retrieved_texts}\n\n"
        "Instructions: Consider the conversation history to provide contextually relevant insights.\n"
        "Key Insights:"
    )
    
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following insights in 2-3 sentences for clarity.\n\n"
        "Insights:\n{insights}\n\n"
        "Summary:"
    )

    # Extract insights first
    extract_chain = extract_prompt | llm | StrOutputParser()
    insights = extract_chain.invoke({
        "query": query,
        "retrieved_texts": "\n".join(retrieved_texts),
        "conversation_history": conversation_history
    })

    # Then summarize
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    summary = summarize_chain.invoke({"insights": insights})

    state["insights"] = insights
    state["summary"] = summary

    # Determine if web search is needed
    if web_search_enabled and WEB_SEARCH_AVAILABLE:
        needs_web_search = "insufficient" in summary.lower() or "additional" in summary.lower() or "more information" in summary.lower()
        state["needs_web_search"] = needs_web_search
    else:
        state["needs_web_search"] = False

    return state

def writer_node(state):
    query = state["query"]
    insights = state["insights"]
    summary = state.get("summary", "")
    web_results = state.get("web_results", "")
    conversation_history = state.get("conversation_history", "")
    llm = state["llm"]

    # Choose appropriate prompt
    if web_results and state.get("needs_web_search", False):
        writer_prompt = ChatPromptTemplate.from_template(
            "Create a comprehensive answer using PDF insights, summary, conversation history, and web search results.\n\n"
            "{conversation_history}"
            "Current User Query: {query}\n\n"
            "PDF Insights:\n{insights}\n\n"
            "Summary:\n{summary}\n\n"
            "Web Search Results:\n{web_results}\n\n"
            "Instructions:\n"
            "1. Consider previous conversation context when answering\n"
            "2. Reference previous answers when relevant (e.g., 'As mentioned earlier...')\n"
            "3. Start with PDF information\n"
            "4. Supplement with web search\n"
            "5. Indicate sources\n"
            "6. Provide a well-structured answer that maintains conversation flow\n\n"
            "Final Answer:"
        )
    else:
        writer_prompt = ChatPromptTemplate.from_template(
            "Using the PDF insights, summary, and conversation history below, craft a concise and professional answer to the user's query.\n\n"
            "{conversation_history}"
            "Current User Query: {query}\n\n"
            "PDF Insights:\n{insights}\n\n"
            "Summary:\n{summary}\n\n"
            "Instructions:\n"
            "1. Consider previous conversation context when answering\n"
            "2. Reference previous answers when relevant (e.g., 'As I mentioned earlier...')\n"
            "3. Maintain conversation flow and context\n"
            "4. Provide a contextually aware\n\n"
            "Final Answer:"
        )

    # Simple chain
    writer_chain = writer_prompt | llm | StrOutputParser()
    
    final_answer = writer_chain.invoke({
        "query": query,
        "insights": insights,
        "summary": summary,
        "web_results": web_results,
        "conversation_history": conversation_history
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
    conversation_history = state.get("conversation_history", "")
    search = DuckDuckGoSearchRun()
    
    # Enhance search query with conversation context
    search_query = f"{query}"
    if conversation_history:
        # Extract key terms from recent conversation for better search
        search_query = f"{query} literature author works themes"

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

    # Initialize conversation history
    conversation_history = ConversationHistory(max_history=10)

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
    # Compile the workflow (do this once after defining the workflow)
    graph = workflow.compile()

    system_name = "Enhanced PDF Q&A System with Memory" if web_search_enabled else "PDF Q&A System with Memory"
    print(f"\n{system_name} Ready (LangChain + LangGraph).")
    print("Type your question below (or type 'exit' to quit, 'clear' to clear history, 'history' to view conversation history):\n")

    while True:
        query = input(">> ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        elif query.lower() == "clear":
            conversation_history.clear_history()
            print("🗑️ Conversation history cleared.")
            continue
        elif query.lower() == "history":
            history = conversation_history.get_formatted_history()
            if history:
                print(f"\n{history}")
            else:
                print("📝 No conversation history yet.")
            continue

        # Get recent conversation context for the current query
        recent_context = conversation_history.get_recent_context(num_recent=3)

        # Initial state for workflow
        state = {
            "query": query,
            "vectorstore": vectorstore,
            "llm": llm,
            "web_search_enabled": web_search_enabled,
            "conversation_history": recent_context,
        }
        
        # Run the workflow
        state = graph.invoke(state)

        # Store this interaction in conversation history
        conversation_history.add_interaction(query, state['final_answer'])

        print(f"\nKey Insights:\n{state['insights']}\n")
        print(f"Final Answer:\n{state['final_answer']}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()