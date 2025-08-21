# Generative AI Internship

A comprehensive repository showcasing my journey through Generative AI technologies during my internship, featuring practical implementations of LangChain, LangGraph, and various AI-powered applications.

## 🎯 Overview

This repository documents my hands-on exploration of cutting-edge Generative AI frameworks and tools. It includes complete implementations ranging from basic chain operations to complex multi-agent systems, demonstrating proficiency in building production-ready AI applications.


## 📁 Project Structure

### 🔗 LangChain Implementations
- **`Chains.py`** - Core chain implementations including LLMChain, SequentialChain, and SimpleSequentialChain for marketing slogans and LinkedIn post generation
- **`conversations_chains.py`** - Conversation memory management with different memory types (Buffer, Window, Summary, Combined)
- **`qna.py`** - Document Q&A system with RAG (Retrieval Augmented Generation) using FAISS vector store
- **`social_media_agent.py`** - Professional social media content generator with token tracking and LCEL chains
- **`runnable_*.py`** - Collection of LangChain Expression Language (LCEL) examples:
  - **Sequence**: Customer support chat with conversation history
  - **Parallel**: Multi-output generation (greetings, compliments, advice)
  - **Passthrough**: Text analysis with parallel summarization and sentiment analysis
  - **Branch**: Conditional logic flows
  - **Map**: Batch processing operations
  - **Generator**: Streaming response generation

### 🕸️ LangGraph Applications  
- **`basic_graph.py`** - Fundamental graph-based chatbot implementation
- **`customer_support_graph.py`** - Advanced multi-agent customer support system with department routing (billing, technical, shipping) and escalation handling

## ✨ Key Features Implemented

### 🔄 Chain Operations
- **Sequential Processing**: Multi-step workflows for content generation
- **Parallel Execution**: Simultaneous processing of multiple tasks
- **Memory Management**: Conversation context retention across sessions
- **Router Chains**: Intelligent routing based on input classification

### 🤖 Multi-Agent Systems
- **Specialized Agents**: Purpose-built agents for specific domains (customer support, content creation)
- **State Management**: Sophisticated state handling across agent interactions
- **Conditional Routing**: Dynamic workflow paths based on content analysis
- **Escalation Logic**: Intelligent escalation mechanisms for complex scenarios

### 📄 Document Intelligence
- **PDF Processing**: Advanced text extraction and chunking strategies
- **Vector Embeddings**: Semantic search using Google's embedding models
- **RAG Implementation**: Context-aware question answering with source attribution
- **Hybrid Search**: Combination of document retrieval and web search


Through this internship, I've gained hands-on experience with:

- **LangChain Ecosystem**: Mastery of chains, agents, memory, and tools
- **LangGraph Framework**: Building complex multi-agent workflows with state management
- **LLM Integration**: Working with Google Gemini models and prompt engineering
- **Production Patterns**: Error handling, logging, configuration management, and scalable architectures
- **AI Application Design**: From simple chains to complex multi-agent systems

---

*This repository represents my practical journey in Generative AI development, showcasing progression from fundamental concepts to advanced multi-agent systems.*