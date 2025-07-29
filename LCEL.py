import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports for Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangChain Core: prompt templates, LCEL utilities, and chat history
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Document loading and vector storage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Modern chain creation
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

class DocumentAssistant:
    def __init__(self):
        # Initialize Google's Gemini model with moderate creativity (temperature=0.7)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.7
        )
        # Initialize Google's embedding model for converting text to vectors
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def load_and_process_docs(self, uploaded_files):
        # List to store all processed document objects
        documents = []
        
        # Process each uploaded file one by one
        for uploaded_file in uploaded_files:
            # Create temporary file path using original filename
            temp_file_path = f"temp_{uploaded_file.name}"
            
            # Write uploaded file content to temporary file on disk
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Choose appropriate loader based on file extension
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    loader = TextLoader(temp_file_path, encoding='utf-8')
                else:
                    # Skip unsupported file types
                    continue
                    
                # Load document content and add to documents list
                documents.extend(loader.load())
            finally:
                # Always clean up temporary file, even if processing fails
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        # Return None if no valid documents were processed
        if not documents:
            return None
            
        # Split large documents into smaller chunks for better processing
        # chunk_size=1000: each chunk max 1000 characters
        # chunk_overlap=200: overlap between chunks to maintain context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create vector database from document chunks using FAISS
        # This enables semantic search through documents
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Return retriever object for querying the vector database
        return vectorstore.as_retriever()
    
    def create_conversational_chain(self, retriever):
        # Initialize session store for managing chat history across conversations
        if 'store' not in st.session_state:
            st.session_state.store = {}

        # Function to get or create chat history for a specific session
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Create comprehensive prompt template with Chain of Thought reasoning
        # System message defines AI behavior and thinking process
        # Placeholder for chat_history maintains conversation context
        # Human message slot for user input
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Use the following step-by-step approach to answer questions:

Step 1: READ - Carefully examine the provided context and identify key information relevant to the question.

Step 2: ANALYZE - Consider the question from multiple angles:
- What specific information is being requested?
- How does the context relate to this question?
- Are there any connections or patterns in the information?

Step 3: SYNTHESIZE - Combine the relevant information to form a comprehensive understanding.

Step 4: RESPOND - Provide a clear, direct answer. Do not start with phrases like "Based on the context" or "According to the document". Just provide the answer naturally.

If the context doesn't contain enough information to answer the question completely, clearly state what information is available and what is missing.

Context: {context}"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        # Create chain that combines retrieved documents with the prompt
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain that first retrieves relevant docs, then answers
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Wrap chain with message history to enable conversation memory
        # Maps input/output keys to maintain proper conversation flow
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain
    
    def create_summary_chain(self, retriever):
        # Function to clean and format AI-generated bullet points
        def format_bullet_points(text):
            # Remove unwanted asterisks and normalize spacing
            text = text.replace('*', '').replace('  ', ' ').strip()
            
            # Split text by bullet points to process each one individually
            lines = text.split('•')
            formatted_lines = []
            
            # Process each line to ensure proper bullet formatting
            for line in lines:
                line = line.strip()
                if line:  # Only process non-empty lines
                    # Add bullet symbol if not already present
                    if not line.startswith('•'):
                        formatted_lines.append(f"• {line}")
                    else:
                        formatted_lines.append(line)
            
            # Join with double line breaks for proper Streamlit markdown rendering
            return '\n\n'.join(formatted_lines)
        
        # Create detailed Chain of Thought prompt for document summarization
        # Guides AI through systematic 5-step summarization process
        summary_prompt = ChatPromptTemplate.from_template("""
You are an expert document summarizer. Follow this systematic approach to create an effective summary:

Step 1: SCAN - Quickly review the entire context to understand the overall scope and main subject matter.

Step 2: IDENTIFY - Locate the most important themes, concepts, and key points throughout the document.
- What are the primary topics discussed?
- What are the main arguments or findings?
- What supporting details are crucial?

Step 3: PRIORITIZE - Rank the identified information by importance and relevance.
- Which points are fundamental to understanding the document?
- What information would be most valuable to someone who hasn't read the full text?

Step 4: SYNTHESIZE - Combine the prioritized information into 10-12 clear, concise bullet points.
- Each bullet point should capture a distinct key concept
- Use • symbol for each point
- Ensure each bullet point is on a separate line
- Maintain logical flow between points

Step 5: REVIEW - Ensure the summary accurately represents the document's main ideas without adding external information.

Format your response as:
• First key point
• Second key point
• Third key point
(and so on...)

Context: {context}

Summary:""")
    
        # Build LCEL chain: retriever → prompt → LLM → parser → formatter
        # This creates a pipeline that processes documents into formatted summaries
        summary_chain = (
            retriever
            | summary_prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(format_bullet_points)
        )
        
        return summary_chain

def main():
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Document Assistant",
        page_icon="📄",
        layout="wide"
    )
    
    # Display main page header and description
    st.title("📄 Document Assistant")
    st.markdown("Upload documents and ask questions or get summaries!")
    
    # Initialize DocumentAssistant instance in session state (persists across reruns)
    if 'assistant' not in st.session_state:
        st.session_state.assistant = DocumentAssistant()
    
    # Create sidebar for file upload functionality
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        # File uploader widget accepting PDF and TXT files
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload one or more PDF or TXT files to analyze"
        )
        
        # Process uploaded files when button is clicked
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                # Show processing spinner while documents are being processed
                with st.spinner("Processing documents..."):
                    try:
                        # Process documents and create retriever for vector search
                        retriever = st.session_state.assistant.load_and_process_docs(uploaded_files)
                        
                        if retriever:
                            # Store retriever and chains in session state for persistence
                            st.session_state.retriever = retriever
                            st.session_state.conv_chain = st.session_state.assistant.create_conversational_chain(retriever)
                            st.session_state.summary_chain = st.session_state.assistant.create_summary_chain(retriever)
                            
                            # Store processed file names for display
                            file_names = [f.name for f in uploaded_files]
                            st.session_state.processed_files = file_names
                            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")
                        else:
                            st.error("❌ No valid documents found!")
                    except Exception as e:
                        # Display any processing errors to user
                        st.error(f"❌ Error processing files: {str(e)}")
        
        # Display list of successfully processed files
        if 'processed_files' in st.session_state:
            st.info("📋 **Processed Files:**")
            for filename in st.session_state.processed_files:
                st.write(f"• {filename}")
    
    # Show message if no documents have been processed yet
    if 'retriever' not in st.session_state:
        st.info("👆 Please upload and process documents using the sidebar to get started!")
        return
    
    # Create main interface with tabs for different functions
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📋 Summarize"])
    
    # Tab 1: Question and Answer functionality
    with tab1:
        st.header("💬 Ask Questions")
        
        # Initialize chat history in session state if not present
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Text input field for user questions
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic of the documents?",
            key="question_input"
        )
        
        # Create two columns for buttons
        col1, col2 = st.columns([1, 4])
        
        # Ask Question button
        with col1:
            ask_button = st.button("🚀 Ask Question", type="primary")
        
        # Clear Chat button
        with col2:
            if st.button("🗑️ Clear Chat"):
                # Reset chat history and conversation memory
                st.session_state.chat_history = []
                if 'store' in st.session_state:
                    st.session_state.store = {}
                st.rerun()  # Refresh page to show cleared state
        
        # Process question when Ask button is clicked
        if ask_button and question.strip():
            try:
                # Show thinking spinner while processing
                with st.spinner("🤔 Thinking..."):
                    # Invoke conversation chain with user input and session ID
                    result = st.session_state.conv_chain.invoke(
                        {"input": question},
                        config={"configurable": {"session_id": "default_session"}}
                    )
                    
                    # Extract answer from chain result
                    response = result["answer"]
                    
                    # Add Q&A pair to chat history for display
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response
                    })
                    
            except Exception as e:
                # Display any errors that occur during processing
                st.error(f"❌ Error: {str(e)}")
        
        # Display conversation history if it exists
        if st.session_state.chat_history:
            st.markdown("---")  # Visual separator
            st.subheader("💬 Conversation")
            
            # Show conversations in reverse order (newest first)
            # Use expanders to save space, expand only the most recent
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {chat['question'][:60]}...", expanded=(i==1)):
                    # Show only answer since question is already in expander title
                    st.markdown(f"**💡 Answer:** {chat['answer']}")
    
    # Tab 2: Document Summarization functionality
    with tab2:
        st.header("📋 Document Summary")
        
        # Generate Summary button
        if st.button("📄 Generate Summary", type="primary"):
            try:
                # Show processing spinner while generating summary
                with st.spinner("📝 Creating summary..."):
                    # Invoke summary chain to generate document summary
                    summary = st.session_state.summary_chain.invoke("Summarize the main points")
                    
                    # Display summary results with formatting
                    st.markdown("### 📋 Summary Results:")
                    st.markdown(summary)
                    
            except Exception as e:
                # Display any errors that occur during summarization
                st.error(f"❌ Error generating summary: {str(e)}")

# Entry point - run main function when script is executed directly
if __name__ == "__main__":
    main()