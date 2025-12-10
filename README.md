Document-Based Question Answering System
Using RAG, FAISS, HuggingFace Embeddings & Groq LLM

This project is an AI-powered Document Question Answering System that enables users to upload PDF documents and interact with them through natural language queries. The system is built using the Retrieval-Augmented Generation (RAG) approach to ensure accurate, context-aware answers.

Users can:

Upload multiple PDFs

Convert documents into embeddings

Store them in a FAISS vector database

Retrieve relevant sections using semantic search

Generate precise answers using Groq LLaMA-based LLM

Interact through a modern Streamlit chat interface


🏗️ System Architecture (RAG Pipeline)

This system follows the Retrieval-Augmented Generation (RAG) architecture:

                  ┌────────────────────┐
                  │   User Uploads     │
                  │        PDFs        │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │  PDF Loader Module │
                  │   (PyPDFLoader)    │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │  Text Chunking     │
                  │ (Recursive Split) │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │ HuggingFace        │
                  │ Embedding Model   │
                  └─────────┬──────────┘
                            ↓
                  ┌────────────────────┐
                  │   FAISS Vector     │
                  │   Database         │
                  └─────────┬──────────┘
                            ↓
                 ┌────────────────────┐
                 │  Semantic Search   │
                 │   on FAISS         │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │   Retrieved        │
                 │   Context Chunks   │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │   Groq LLM         │
                 │ (LLaMA 3.3 - 70B) │
                 └─────────┬──────────┘
                           ↓
                 ┌────────────────────┐
                 │  Final Answer Sent │
                 │   to Streamlit UI  │
                 └────────────────────┘
