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

```bash
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


