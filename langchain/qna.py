from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

import tkinter as tk
from tkinter import filedialog

file_path = filedialog.askopenfilename(title="Select your document")
if not file_path:
    print("No file selected. Exiting.")
    exit()

load_dotenv()

# Use the right loader based on file extension
if file_path.lower().endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path)

docs = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Creates embeddings for each chunk and stores them in a FAISS vector database.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embeddings)

# Creates a retriever to search for relevant chunks.
retriever = vectorstore.as_retriever()

# Initializes the Gemini LLM with specified model and temperature.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# Imports the RetrievalQA chain for question answering.
from langchain.chains import RetrievalQA

# Creates a RetrievalQA chain that uses the LLM and retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

question = input("Ask a question about your documents: ")
result = qa_chain.invoke({"query": question})

print("Answer:", result["result"])
