from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. Load PDFs from a directory ---
def load_pdf(data_path: str):
    """
    Loads all PDF files from the specified directory.
    
    Args:
        data_path (str): Path to the folder containing PDFs.
        
    Returns:
        List[Document]: List of langchain Document objects.
    """
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF documents from {data_path}")
    return documents


# --- 2. Split documents into smaller text chunks ---
def text_split(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits the loaded documents into smaller chunks for embedding.
    
    Args:
        documents (List[Document]): List of langchain Document objects.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
        
    Returns:
        List[Document]: List of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} text chunks")
    return chunks


# --- 3. Initialize HuggingFace Embeddings ---
def download_hugging_face_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes HuggingFace embedding model.
    
    Args:
        model_name (str): The HuggingFace sentence transformer model to use.
        
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Initialized embeddings model: {model_name}")
    return embeddings