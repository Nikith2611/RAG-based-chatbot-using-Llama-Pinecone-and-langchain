# RAG Chatbot using Llama 2, Pinecone, and LangChain

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **Meta Llama 2**, and **Pinecone** for semantic document retrieval and context-aware question answering.

This project demonstrates how to build a **document-based AI assistant** that retrieves relevant information from a knowledge source and generates grounded responses using a large language model.

---

## 🚀 Overview

This project implements a **RAG (Retrieval-Augmented Generation)** pipeline where:

- source documents are processed and embedded
- embeddings are stored in **Pinecone**
- relevant chunks are retrieved based on user queries
- retrieved context is passed to **Llama 2**
- the chatbot generates context-aware responses

This architecture is commonly used in **enterprise AI assistants**, **document Q&A systems**, and **knowledge retrieval applications**.

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **Flask**
- **Meta Llama 2**
- **Pinecone**
- **Sentence Transformers / Embeddings**
- **Vector Database**
- **Prompt Engineering**

---

## 📌 Features

- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search using **Pinecone**
- Context-aware chatbot responses
- Modular project structure
- Local LLM inference using **Llama 2**
- Flask-based chatbot UI
- Reusable prompt and helper utilities

---

## 📂 Project Structure

```text
RAG-based-chatbot-using-Llama2-Pinecone-and-langchain/
│
├── Data/
│   └── Medical_book.pdf              # Source knowledge document
│
├── research/
│   └── trials.ipynb                  # Experimentation / testing notebook
│
├── src/
│   ├── __init__.py
│   ├── helper.py                     # Utility/helper functions
│   └── prompt.py                     # Prompt templates / prompt logic
│
├── templates/                        # HTML templates for chatbot UI
│
├── .gitignore
├── LICENSE
├── README.md
├── app.py                            # Main Flask application
├── requirements.txt                  # Python dependencies
└── setup.py                          # Project/package setup
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nikith2611/RAG-based-chatbot-using-Llama2-Pinecone-and-langchain.git
cd RAG-based-chatbot-using-Llama2-Pinecone-and-langchain
```

### 2️⃣ Create and Activate a Conda Environment

```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root and add your Pinecone credentials:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENV=your_pinecone_environment
```

⚠️ Make sure your `.env` file is not committed to GitHub.

---

## 🧠 Model Setup

### Download the Llama 2 Model

Download the following model file:

```text
llama-2-7b-chat.ggmlv3.q4_0.bin
```

**Model Source:**  
Download it from Hugging Face:  
**TheBloke / Llama-2-7B-Chat-GGML**

Place the downloaded model file in the location expected by your application.

---

## 📚 Data Source

This project uses the following knowledge source:

```text
Medical_book.pdf
```

The PDF is processed, chunked, embedded, and indexed into Pinecone to enable document-based question answering.

---

## 📦 Indexing the Documents

Run the indexing script to process the document and store embeddings in Pinecone:

```bash
python store_index.py
```

This step performs the following:

- loads the PDF document
- splits the content into chunks
- generates embeddings
- stores the embeddings in Pinecone for retrieval

Make sure your Pinecone index is configured properly before running this step.

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open your browser and visit:

```text
http://localhost:5000
```

---

## 💬 How It Works

- The user asks a question in the chatbot interface
- The query is converted into embeddings
- Pinecone retrieves the most relevant document chunks
- Retrieved context is passed to Llama 2
- The model generates a grounded response based on the retrieved content

This improves response quality by reducing hallucinations and increasing factual relevance.

