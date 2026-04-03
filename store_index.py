# store_index.py
from src.chatbot import upload_to_pinecone

if __name__ == "__main__":
    pdf_folder = "data/"
    print("Uploading PDFs to Pinecone...")
    upload_to_pinecone(pdf_folder)
    print("Upload complete!")