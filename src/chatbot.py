# src/chatbot.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "chatbot")  # default to "chatbot"

# ----------------------------
# 1. Initialize Embeddings
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# 2. Pinecone Vector Store Setup
# ----------------------------
def init_pinecone():
    import pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    return PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

# ----------------------------
# 3. Load and Split PDFs
# ----------------------------
def load_and_split_pdfs(pdf_folder: str, chunk_size=500, chunk_overlap=50):
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# ----------------------------
# 4. Upload PDFs to Pinecone
# ----------------------------
def upload_to_pinecone(pdf_folder: str):
    from langchain_pinecone import PineconeVectorStore
    import pinecone

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    text_chunks = load_and_split_pdfs(pdf_folder)
    texts = [doc.page_content for doc in text_chunks]
    metadatas = [doc.metadata for doc in text_chunks]

    store = PineconeVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    return store

# ----------------------------
# 5. Initialize RAG QA Chain
# ----------------------------
def get_qa_chain(k=3):
    # Load the vector store
    docsearch = init_pinecone()
    
    # Load LLM
    llm = OllamaLLM(model="llama3.2")
    
    # RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": k})
    )
    return qa